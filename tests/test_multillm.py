"""Real tiny local AutoModel fixtures test all three backbone architectures offline."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import torch
from transformers import GPT2Config, GPT2Model, Qwen2Config, Qwen2Model, LlamaConfig, LlamaModel
from models.TriCast import Model
from run_tricast import arguments
from utils.llm_backbone import inspect_backbone


class MultiLLMTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(2)

    def test_local_architectures_and_gradient_flow(self):
        fixtures = [
            ('gpt2', GPT2Config(n_embd=32, n_layer=2, n_head=4, n_positions=64, vocab_size=32), GPT2Model),
            ('qwen2', Qwen2Config(hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                  num_attention_heads=4, num_key_value_heads=2, vocab_size=32,
                                  max_position_embeddings=64), Qwen2Model),
            ('llama', LlamaConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                  num_attention_heads=4, num_key_value_heads=2, vocab_size=32,
                                  max_position_embeddings=64), LlamaModel),
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for family, cfg, cls in fixtures:
                with self.subTest(family=family):
                    path = root / family
                    cls(cfg).save_pretrained(path)
                    registry = root / 'registry.json'
                    registry.write_text(json.dumps({'models': {family: {
                        'model_id': 'unused/offline', 'local_path': str(path),
                        'model_type': family, 'hidden_size': 32, 'full_layers': 2}}}))
                    args = SimpleNamespace(seq_len=24, pred_len=8, enc_in=2, d_model=16,
                        d_ff=16, patch_len=8, n_heads=4, dropout=0., tricast_stage=5,
                        llm_name=family, llm_registry=str(registry), llm_path=None,
                        llm_layers=None, llm_dtype='float32', msdr_scales=[2, 4],
                        msdr_max_lag=2, num_experts=2)
                    for precision in ('float32', 'bfloat16'):
                        args.llm_dtype = precision
                        model = Model(args)
                        x = torch.randn(2, 24, 2)
                        prediction = model(x)
                        self.assertEqual(prediction.shape, (2, 8, 2))
                        prediction.square().mean().backward()
                        self.assertTrue(torch.isfinite(model.to_llm.weight.grad).all())
                        self.assertGreater(model.to_llm.weight.grad.abs().sum().item(), 0)
                        self.assertTrue(all(p.grad is None for p in model.llm_model.parameters()))
                        self.assertEqual(model.backbone_identity['used_layers'], 2)
                        self.assertEqual(model.to_llm.out_features, 32)
                        model.train()
                        self.assertFalse(model.llm_model.training)
                    args.llm_layers = 1
                    self.assertEqual(Model(args).backbone_identity['used_layers'], 1)
                    args.llm_layers = 3
                    with self.assertRaises(ValueError):
                        inspect_backbone(args)
                    args.llm_layers = None
                    profile = json.loads(registry.read_text())
                    profile['models'][family]['hidden_size'] = 99
                    registry.write_text(json.dumps(profile))
                    with self.assertRaises(ValueError):
                        inspect_backbone(args)

    def test_configuration_priority_and_all_experiments(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / 'experiment.json'
            config.write_text(json.dumps({'llm_name': 'qwen2.5-0.5b', 'llm_layers': None,
                                          'batch_size': 2, 'amp': False}))
            args = arguments(['--config', str(config), '--batch_size', '3'])
            self.assertEqual(args.llm_name, 'qwen2.5-0.5b')
            self.assertEqual(args.batch_size, 3)
            self.assertIsNone(args.llm_layers)
            config.write_text(json.dumps({'typo_key': 3}))
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                arguments(['--config', str(config)])
        root = Path(__file__).resolve().parents[1]
        configs = list((root / 'configs' / 'experiments').glob('*.json'))
        self.assertEqual(len(configs), 28)
        for config in configs:
            args = arguments(['--config', str(config)])
            self.assertEqual(args.freq, 't' if args.data.startswith('ETTm') else 'h')
            self.assertIsNone(args.llm_layers)
            self.assertEqual(args.data_path, args.data + '.csv')


if __name__ == '__main__':
    unittest.main()
