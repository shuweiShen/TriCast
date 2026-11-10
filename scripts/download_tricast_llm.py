"""Explicitly download one configured backbone to a user-selected directory."""
import argparse
import json
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download


def main():
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True)
    p.add_argument('--output', required=True, help='Directory where model weights will be downloaded')
    p.add_argument('--registry', default=str(root / 'configs' / 'llm_backbones.json'))
    p.add_argument('--register', action='store_true', help='Write downloaded path into the registry after success')
    args = p.parse_args()
    registry = Path(args.registry)
    content = json.loads(registry.read_text(encoding='utf-8-sig'))
    if args.model not in content['models']:
        p.error(f'Unknown model: {args.model}')
    profile = content['models'][args.model]
    output = Path(args.output).expanduser().resolve()
    files = HfApi().list_repo_files(profile['model_id'])
    weight_pattern = '*.safetensors' if any(f.endswith('.safetensors') for f in files) else '*.bin'
    snapshot_download(repo_id=profile['model_id'], local_dir=str(output),
                      allow_patterns=['*.json', weight_pattern, '*.model', '*.txt', '*.tiktoken'])
    if args.register:
        profile['local_path'] = str(output)
        registry.write_text(json.dumps(content, indent=2) + '\n', encoding='utf-8')
    print(f'Downloaded {args.model} to {output}')


if __name__ == '__main__':
    main()
