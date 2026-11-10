"""Numerical/behavioral checks independent of the pretrained model."""
import unittest
import torch
from layers.TriCast import MSDR, FiLM, FourierCrossAttn, CPRouter, PatchEncoder


class TriCastTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2021)
        torch.set_num_threads(2)

    def test_constant_and_short_inputs(self):
        detector = MSDR()
        for length in [1, 17, 192]:
            prior = detector(torch.ones(2, length, 3))
            self.assertEqual(prior.shape, (2, length, 3))
            self.assertTrue(torch.equal(prior, torch.zeros_like(prior)))

    def test_level_shift_and_sample_isolation(self):
        x = torch.randn(1, 384, 1) * 0.1
        x[:, 192:] += 5
        detector = MSDR()
        prior = detector(x)
        self.assertTrue(torch.isfinite(prior).all())
        self.assertTrue(((prior >= 0) & (prior <= 1)).all())
        self.assertGreater(prior[:, 180:205].mean().item(), prior[:, 70:120].mean().item() + .2)
        other = torch.randn_like(x) * 100
        self.assertTrue(torch.allclose(prior, detector(torch.cat([x, other]))[:1]))

    def test_nonoverlap_padding(self):
        encoder = PatchEncoder(96, 16)
        self.assertEqual(encoder(torch.randn(2, 193, 3)).shape, (6, 3, 16))

    def test_fusion_and_router_conditioning(self):
        time, cp, spectrum = torch.randn(2, 7, 16), torch.randn(2, 7, 16), torch.randn(2, 40, 16)
        film = FiLM(16)
        attention = FourierCrossAttn(16, 4, 0)
        fused = attention(film(time, cp), spectrum)
        self.assertEqual(fused.shape, time.shape)
        router = CPRouter(16, 32, 4, 0)
        hidden = torch.randn(2, 7, 32)
        router(hidden, cp, spectrum)
        weights = router.last_weights.clone()
        router(hidden * 100, cp, spectrum)
        self.assertTrue(torch.equal(weights, router.last_weights))
        router(hidden, cp + 3 * torch.randn_like(cp), spectrum)
        self.assertFalse(torch.allclose(weights, router.last_weights))
        router(hidden, cp, spectrum).square().mean().backward()
        self.assertGreater(router.router.weight.grad.abs().sum().item(), 0)


if __name__ == '__main__':
    unittest.main()
