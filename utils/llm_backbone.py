"""Local-only backbone selection. Model identity is recorded in adapter checkpoints."""
import hashlib
import json
from pathlib import Path
import torch
from transformers import AutoConfig, AutoModel

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = PROJECT_ROOT / 'configs' / 'llm_backbones.json'


def read_profile(name='gpt2', registry=None):
    registry = Path(registry or DEFAULT_REGISTRY).resolve()
    profiles = json.loads(registry.read_text(encoding='utf-8-sig'))['models']
    if name not in profiles:
        raise ValueError(f'Unknown LLM {name!r}; choose from {list(profiles)}')
    return profiles[name]


def resolve_snapshot(path):
    path = Path(path).expanduser().resolve()
    if (path / 'config.json').is_file():
        return path
    ref = path / 'refs' / 'main'
    if ref.is_file():
        candidate = path / 'snapshots' / ref.read_text().strip()
        if (candidate / 'config.json').is_file():
            return candidate
    if not ref.exists():
        snapshots = list((path / 'snapshots').glob('*/config.json'))
        if len(snapshots) == 1:
            return snapshots[0].parent
    raise FileNotFoundError(f'No unambiguous config.json snapshot in {path}')


def resolve_backbone(configs):
    name = getattr(configs, 'llm_name', 'gpt2')
    profile = read_profile(name, getattr(configs, 'llm_registry', None))
    explicit = getattr(configs, 'llm_path', None) or profile.get('local_path')
    if explicit:
        path = Path(explicit).expanduser()
        # Registry paths are project-relative; CLI paths are relative to cwd.
        if not path.is_absolute() and not getattr(configs, 'llm_path', None):
            path = PROJECT_ROOT / path
        return resolve_snapshot(path), profile
    model_id = profile['model_id']
    candidates = [PROJECT_ROOT / 'llm_weights' / name,
                  PROJECT_ROOT / ('models--' + model_id.replace('/', '--'))]
    for candidate in candidates:
        if candidate.exists():
            return resolve_snapshot(candidate), profile
    from huggingface_hub import snapshot_download
    try:
        return resolve_snapshot(snapshot_download(model_id, local_files_only=True)), profile
    except Exception as exc:
        raise FileNotFoundError(
            f'{name}: local weights not found. Set models.{name}.local_path in '
            f'configs/llm_backbones.json or pass --llm_path. No download was attempted.'
        ) from exc


def inspect_backbone(configs):
    path, profile = resolve_backbone(configs)
    cfg = AutoConfig.from_pretrained(str(path), local_files_only=True, trust_remote_code=False)
    if cfg.model_type != profile['model_type']:
        raise ValueError(f"Selected {getattr(configs, 'llm_name', 'gpt2')} expects "
                         f"{profile['model_type']}, but weights are {cfg.model_type}")
    hidden = cfg.hidden_size
    if hidden != profile['hidden_size']:
        raise ValueError(f"Selected profile expects hidden_size={profile['hidden_size']}, "
                         f'but local weights have {hidden}; check model variant/path')
    full_depth = cfg.num_hidden_layers
    if full_depth != profile['full_layers']:
        raise ValueError(f'Expected {profile["full_layers"]} original layers, found {full_depth}')
    depth = getattr(configs, 'llm_layers', None)
    depth = full_depth if depth is None else depth
    if not 1 <= depth <= full_depth:
        raise ValueError(f'llm_layers must be between 1 and {full_depth}')
    weights = sorted(p for p in path.iterdir() if p.suffix in ('.safetensors', '.bin'))
    if not weights:
        raise FileNotFoundError(f'No PyTorch/safetensors weights in {path}')
    identity = {
        'name': getattr(configs, 'llm_name', 'gpt2'), 'source': str(path),
        'model_type': cfg.model_type, 'hidden_size': hidden,
        'full_layers': full_depth, 'used_layers': depth,
        'dtype': getattr(configs, 'llm_dtype', 'float32'),
        'config_sha256': hashlib.sha256((path / 'config.json').read_bytes()).hexdigest(),
        # Manifest catches replacement of local files without hashing multi-GB weights.
        'weight_manifest': [[p.name, p.stat().st_size, p.stat().st_mtime_ns] for p in weights],
    }
    cfg.num_hidden_layers = depth
    cfg.use_cache = False
    cfg.output_attentions = False
    cfg.output_hidden_states = False
    return path, cfg, identity


def load_backbone(configs):
    path, cfg, identity = inspect_backbone(configs)
    precision = getattr(configs, 'llm_dtype', 'float32')
    dtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[precision]
    model = AutoModel.from_pretrained(str(path), config=cfg, local_files_only=True,
                                     trust_remote_code=False, torch_dtype=dtype)
    model.requires_grad_(False)
    model.eval()
    return model, cfg, identity
