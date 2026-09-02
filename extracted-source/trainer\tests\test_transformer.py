"""
Numerical gradient checking for the tiny transformer's hand-written
backward pass. This is the real self-verification the spec demands for
a from-scratch backprop implementation — comparing analytic gradients
against finite-difference estimates on several parameters.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
from trainer.algorithms.transformer import TinyTransformer, TinyTransformerConfig


class TestTransformerGradients(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        self.cfg = TinyTransformerConfig(
            vocab_size=12, context_length=6, d_model=8, n_layers=2, n_heads=2, d_ff=16,
            learning_rate=0.01, random_seed=1,
        )
        self.model = TinyTransformer(self.cfg)
        rng = np.random.default_rng(0)
        self.token_ids = rng.integers(0, self.cfg.vocab_size, size=(2, self.cfg.context_length))
        self.targets = rng.integers(0, self.cfg.vocab_size, size=(2, self.cfg.context_length))

    def _assert_close(self, a, b, msg=""):
        # Relative+absolute tolerance is the right comparison for finite
        # difference vs analytic gradients — gradient magnitudes vary a lot
        # across tensors, so a fixed decimal-places check is too strict.
        self.assertTrue(np.isclose(a, b, rtol=2e-2, atol=2e-3),
                         msg=f"{msg}: analytic={a} numeric={b}")

    def _numerical_grad(self, param: np.ndarray, eps=1e-4):
        grad = np.zeros_like(param)
        rng = np.random.default_rng(42)
        flat_indices = list(np.ndindex(param.shape))
        sample = rng.choice(len(flat_indices), size=min(8, len(flat_indices)), replace=False)
        for idx_i in sample:
            idx = flat_indices[idx_i]
            orig = param[idx]
            param[idx] = orig + eps
            loss_plus, _ = self.model.loss_and_backward(self.token_ids, self.targets)
            param[idx] = orig - eps
            loss_minus, _ = self.model.loss_and_backward(self.token_ids, self.targets)
            param[idx] = orig
            grad[idx] = (loss_plus - loss_minus) / (2 * eps)
        return grad, sample, flat_indices

    def test_out_proj_weight_gradient(self):
        loss, grads = self.model.loss_and_backward(self.token_ids, self.targets)
        analytic = grads["out_proj"][0]
        numeric, sample, flat_indices = self._numerical_grad(self.model.out_proj.W)
        for idx_i in sample:
            idx = flat_indices[idx_i]
            self._assert_close(analytic[idx], numeric[idx], f"out_proj.W at {idx}")

    def test_token_embedding_gradient(self):
        loss, grads = self.model.loss_and_backward(self.token_ids, self.targets)
        analytic = grads["token_emb"]
        numeric, sample, flat_indices = self._numerical_grad(self.model.token_emb)
        for idx_i in sample:
            idx = flat_indices[idx_i]
            self._assert_close(analytic[idx], numeric[idx], f"token_emb at {idx}")

    def test_attention_qproj_weight_gradient(self):
        loss, grads = self.model.loss_and_backward(self.token_ids, self.targets)
        analytic = grads["blocks"][0]["attn"]["q_proj"][0]
        numeric, sample, flat_indices = self._numerical_grad(self.model.blocks[0].attn.q_proj.W)
        for idx_i in sample:
            idx = flat_indices[idx_i]
            self._assert_close(analytic[idx], numeric[idx], f"block0.attn.q_proj.W at {idx}")

    def test_ffn_fc1_weight_gradient(self):
        loss, grads = self.model.loss_and_backward(self.token_ids, self.targets)
        analytic = grads["blocks"][1]["ffn"]["fc1"][0]
        numeric, sample, flat_indices = self._numerical_grad(self.model.blocks[1].ffn.fc1.W)
        for idx_i in sample:
            idx = flat_indices[idx_i]
            self._assert_close(analytic[idx], numeric[idx], f"block1.ffn.fc1.W at {idx}")

    def test_layernorm_gamma_gradient(self):
        loss, grads = self.model.loss_and_backward(self.token_ids, self.targets)
        analytic = grads["blocks"][0]["ln1"][0]
        numeric, sample, flat_indices = self._numerical_grad(self.model.blocks[0].ln1.gamma)
        for idx_i in sample:
            idx = flat_indices[idx_i]
            self._assert_close(analytic[idx], numeric[idx], f"block0.ln1.gamma at {idx}")

    def test_loss_decreases_with_training(self):
        losses = []
        for _ in range(40):
            loss = self.model.train_step(self.token_ids, self.targets)
            losses.append(loss)
        self.assertLess(losses[-1], losses[0], "Transformer loss should decrease with real training steps")

    def test_causal_mask_blocks_future_tokens(self):
        """Changing a future token should not change logits for earlier positions (causality)."""
        ids1 = self.token_ids.copy()
        ids2 = self.token_ids.copy()
        ids2[:, -1] = (ids2[:, -1] + 1) % self.cfg.vocab_size  # perturb the LAST token only
        logits1, _ = self.model.forward(ids1)
        logits2, _ = self.model.forward(ids2)
        self.assertTrue(np.allclose(logits1[:, :-1, :], logits2[:, :-1, :], atol=1e-8))
        self.assertFalse(np.allclose(logits1[:, -1, :], logits2[:, -1, :]))

    def test_state_roundtrip_identical_predictions(self):
        logits_before, _ = self.model.forward(self.token_ids)
        state = self.model.get_state()
        restored = TinyTransformer.from_state(state)
        logits_after, _ = restored.forward(self.token_ids)
        self.assertTrue(np.allclose(logits_before, logits_after, atol=1e-8))

    def test_generate_produces_valid_ids(self):
        out = self.model.generate([0, 1, 2], max_new_tokens=5)
        self.assertEqual(len(out), 8)
        self.assertTrue(all(0 <= i < self.cfg.vocab_size for i in out))


if __name__ == "__main__":
    unittest.main(verbosity=2)
