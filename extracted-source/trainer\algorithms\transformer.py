"""
A genuine, from-scratch tiny transformer implemented in NumPy — no autograd
framework. Every operation (embeddings, multi-head causal self-attention,
feed-forward, layer norm, residual connections, output projection, and the
backward pass through all of them) is computed explicitly.

This is intentionally small and conservative-by-default so it can train on
a 6GB CPU-only machine, per the hardware constraint. It is NOT intended to
compete with real LLMs — it's an educational/experimental tiny transformer.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))


def gelu_grad(x):
    eps = 1e-4
    return (gelu(x + eps) - gelu(x - eps)) / (2 * eps)


def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


# ---------------------------------------------------------------- LayerNorm
class LayerNorm:
    def __init__(self, dim, eps=1e-5, rng=None):
        self.gamma = np.ones(dim)
        self.beta = np.zeros(dim)
        self.eps = eps

    def forward(self, x):
        mu = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        std_inv = 1.0 / np.sqrt(var + self.eps)
        x_hat = (x - mu) * std_inv
        out = self.gamma * x_hat + self.beta
        cache = (x_hat, std_inv, x.shape[-1])
        return out, cache

    def backward(self, dout, cache):
        x_hat, std_inv, D = cache
        dgamma = np.sum(dout * x_hat, axis=tuple(range(dout.ndim - 1)))
        dbeta = np.sum(dout, axis=tuple(range(dout.ndim - 1)))

        dx_hat = dout * self.gamma
        dx = (1.0 / D) * std_inv * (
            D * dx_hat
            - np.sum(dx_hat, axis=-1, keepdims=True)
            - x_hat * np.sum(dx_hat * x_hat, axis=-1, keepdims=True)
        )
        return dx, dgamma, dbeta


# -------------------------------------------------------------------- Linear
class Linear:
    def __init__(self, d_in, d_out, rng):
        limit = np.sqrt(2.0 / d_in)
        self.W = rng.normal(0, limit, size=(d_in, d_out))
        self.b = np.zeros(d_out)

    def forward(self, x):
        out = x @ self.W + self.b
        cache = x
        return out, cache

    def backward(self, dout, cache):
        x = cache
        orig_shape = x.shape
        x2 = x.reshape(-1, orig_shape[-1])
        dout2 = dout.reshape(-1, dout.shape[-1])
        dW = x2.T @ dout2
        db = dout2.sum(axis=0)
        dx = (dout2 @ self.W.T).reshape(orig_shape)
        return dx, dW, db


# ---------------------------------------------------- Multi-Head Attention
class MultiHeadSelfAttention:
    def __init__(self, d_model, n_heads, rng):
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.q_proj = Linear(d_model, d_model, rng)
        self.k_proj = Linear(d_model, d_model, rng)
        self.v_proj = Linear(d_model, d_model, rng)
        self.out_proj = Linear(d_model, d_model, rng)

    def _split_heads(self, x):
        B, T, D = x.shape
        return x.reshape(B, T, self.n_heads, self.d_head).transpose(0, 2, 1, 3)  # (B,H,T,Dh)

    def _merge_heads(self, x):
        B, H, T, Dh = x.shape
        return x.transpose(0, 2, 1, 3).reshape(B, T, H * Dh)

    def forward(self, x):
        B, T, D = x.shape
        q, q_cache = self.q_proj.forward(x)
        k, k_cache = self.k_proj.forward(x)
        v, v_cache = self.v_proj.forward(x)

        qh = self._split_heads(q)  # (B,H,T,Dh)
        kh = self._split_heads(k)
        vh = self._split_heads(v)

        scale = 1.0 / np.sqrt(self.d_head)
        scores = np.einsum("bhid,bhjd->bhij", qh, kh) * scale  # (B,H,T,T)

        causal_mask = np.triu(np.ones((T, T)), k=1).astype(bool)
        scores = np.where(causal_mask[None, None, :, :], -1e9, scores)

        attn = softmax(scores, axis=-1)  # (B,H,T,T)
        context = np.einsum("bhij,bhjd->bhid", attn, vh)  # (B,H,T,Dh)
        merged = self._merge_heads(context)  # (B,T,D)

        out, out_cache = self.out_proj.forward(merged)

        cache = {
            "q_cache": q_cache, "k_cache": k_cache, "v_cache": v_cache,
            "qh": qh, "kh": kh, "vh": vh, "attn": attn, "context": context,
            "out_cache": out_cache, "scale": scale, "T": T, "B": B,
        }
        return out, cache

    def backward(self, dout, cache):
        d_merged, dW_out, db_out = self.out_proj.backward(dout, cache["out_cache"])
        # split d_merged back into heads
        B, T = cache["B"], cache["T"]
        d_context = d_merged.reshape(B, T, self.n_heads, self.d_head).transpose(0, 2, 1, 3)  # (B,H,T,Dh)

        attn, vh, qh, kh = cache["attn"], cache["vh"], cache["qh"], cache["kh"]
        scale = cache["scale"]

        # context = attn @ vh
        d_attn = np.einsum("bhid,bhjd->bhij", d_context, vh)  # (B,H,T,T)
        d_vh = np.einsum("bhij,bhid->bhjd", attn, d_context)  # (B,H,T,Dh)

        # softmax backward (per row over last axis)
        d_scores = attn * (d_attn - np.sum(d_attn * attn, axis=-1, keepdims=True))

        # scores = qh @ kh^T * scale (causal-masked positions had -1e9 fixed constant, grad flows as 0 there naturally since attn≈0)
        d_qh = np.einsum("bhij,bhjd->bhid", d_scores, kh) * scale
        d_kh = np.einsum("bhij,bhid->bhjd", d_scores, qh) * scale

        d_q = self._merge_heads(d_qh)
        d_k = self._merge_heads(d_kh)
        d_v = self._merge_heads(d_vh)

        dx_q, dW_q, db_q = self.q_proj.backward(d_q, cache["q_cache"])
        dx_k, dW_k, db_k = self.k_proj.backward(d_k, cache["k_cache"])
        dx_v, dW_v, db_v = self.v_proj.backward(d_v, cache["v_cache"])

        dx = dx_q + dx_k + dx_v
        grads = {
            "q_proj": (dW_q, db_q), "k_proj": (dW_k, db_k),
            "v_proj": (dW_v, db_v), "out_proj": (dW_out, db_out),
        }
        return dx, grads


# ---------------------------------------------------------- Feed-forward
class FeedForward:
    def __init__(self, d_model, d_ff, rng):
        self.fc1 = Linear(d_model, d_ff, rng)
        self.fc2 = Linear(d_ff, d_model, rng)

    def forward(self, x):
        h, fc1_cache = self.fc1.forward(x)
        a = gelu(h)
        out, fc2_cache = self.fc2.forward(a)
        cache = {"fc1_cache": fc1_cache, "h": h, "fc2_cache": fc2_cache}
        return out, cache

    def backward(self, dout, cache):
        da, dW2, db2 = self.fc2.backward(dout, cache["fc2_cache"])
        dh = da * gelu_grad(cache["h"])
        dx, dW1, db1 = self.fc1.backward(dh, cache["fc1_cache"])
        grads = {"fc1": (dW1, db1), "fc2": (dW2, db2)}
        return dx, grads


# ------------------------------------------------------------- Transformer block
class TransformerBlock:
    def __init__(self, d_model, n_heads, d_ff, rng):
        self.ln1 = LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_heads, rng)
        self.ln2 = LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, rng)

    def forward(self, x):
        ln1_out, ln1_cache = self.ln1.forward(x)
        attn_out, attn_cache = self.attn.forward(ln1_out)
        x2 = x + attn_out

        ln2_out, ln2_cache = self.ln2.forward(x2)
        ffn_out, ffn_cache = self.ffn.forward(ln2_out)
        x3 = x2 + ffn_out

        cache = {"ln1_cache": ln1_cache, "attn_cache": attn_cache,
                  "ln2_cache": ln2_cache, "ffn_cache": ffn_cache}
        return x3, cache

    def backward(self, dx3, cache):
        # x3 = x2 + ffn_out
        dx2_res = dx3
        dffn_out = dx3
        dln2_out, ffn_grads = self.ffn.backward(dffn_out, cache["ffn_cache"])
        dx2_ln, dgamma2, dbeta2 = self.ln2.backward(dln2_out, cache["ln2_cache"])
        dx2 = dx2_res + dx2_ln

        # x2 = x + attn_out
        dx_res = dx2
        dattn_out = dx2
        dln1_out, attn_grads = self.attn.backward(dattn_out, cache["attn_cache"])
        dx_ln, dgamma1, dbeta1 = self.ln1.backward(dln1_out, cache["ln1_cache"])
        dx = dx_res + dx_ln

        grads = {
            "ln1": (dgamma1, dbeta1), "attn": attn_grads,
            "ln2": (dgamma2, dbeta2), "ffn": ffn_grads,
        }
        return dx, grads


@dataclass
class TinyTransformerConfig:
    vocab_size: int
    context_length: int = 32
    d_model: int = 64
    n_layers: int = 2
    n_heads: int = 4
    d_ff: int = 128
    learning_rate: float = 0.001
    random_seed: int = 42


class TinyTransformer:
    def __init__(self, cfg: TinyTransformerConfig):
        self.cfg = cfg
        rng = np.random.default_rng(cfg.random_seed)

        self.token_emb = rng.normal(0, 0.02, size=(cfg.vocab_size, cfg.d_model))
        self.pos_emb = rng.normal(0, 0.02, size=(cfg.context_length, cfg.d_model))

        self.blocks = [TransformerBlock(cfg.d_model, cfg.n_heads, cfg.d_ff, rng) for _ in range(cfg.n_layers)]
        self.ln_final = LayerNorm(cfg.d_model)
        self.out_proj = Linear(cfg.d_model, cfg.vocab_size, rng)

        self._adam_state = {}
        self._t = 0

    # --------------------------------------------------------------- forward
    def forward(self, token_ids: np.ndarray):
        """token_ids: (B, T) int array. Returns logits (B, T, vocab_size) + cache."""
        B, T = token_ids.shape
        x = self.token_emb[token_ids] + self.pos_emb[:T][None, :, :]

        block_caches = []
        for block in self.blocks:
            x, cache = block.forward(x)
            block_caches.append(cache)

        x_final, ln_final_cache = self.ln_final.forward(x)
        logits, out_cache = self.out_proj.forward(x_final)

        cache = {
            "token_ids": token_ids, "block_caches": block_caches,
            "ln_final_cache": ln_final_cache, "out_cache": out_cache, "T": T,
        }
        return logits, cache

    def loss_and_backward(self, token_ids: np.ndarray, targets: np.ndarray):
        """Real cross-entropy loss + full manual backprop through the whole model."""
        logits, cache = self.forward(token_ids)
        B, T, V = logits.shape
        probs = softmax(logits, axis=-1)

        eps = 1e-12
        onehot = np.zeros_like(probs)
        b_idx, t_idx = np.meshgrid(np.arange(B), np.arange(T), indexing="ij")
        onehot[b_idx, t_idx, targets] = 1.0
        loss = -np.mean(np.sum(onehot * np.log(np.clip(probs, eps, 1)), axis=-1))

        dlogits = (probs - onehot) / (B * T)

        dx_final, dW_out, db_out = self.out_proj.backward(dlogits, cache["out_cache"])
        dx, dgamma_f, dbeta_f = self.ln_final.backward(dx_final, cache["ln_final_cache"])

        block_grads = []
        for block, block_cache in zip(reversed(self.blocks), reversed(cache["block_caches"])):
            dx, g = block.backward(dx, block_cache)
            block_grads.append(g)
        block_grads.reverse()

        # gradient w.r.t. embeddings
        d_token_emb = np.zeros_like(self.token_emb)
        np.add.at(d_token_emb, cache["token_ids"], dx)
        d_pos_emb = np.zeros_like(self.pos_emb)
        d_pos_emb[:cache["T"]] += dx.sum(axis=0)

        grads = {
            "token_emb": d_token_emb, "pos_emb": d_pos_emb,
            "blocks": block_grads,
            "ln_final": (dgamma_f, dbeta_f),
            "out_proj": (dW_out, db_out),
        }
        return float(loss), grads

    # -------------------------------------------------------------- optimizer
    def _adam_update(self, key, param, grad, lr, beta1=0.9, beta2=0.999, eps=1e-8):
        if key not in self._adam_state:
            self._adam_state[key] = {"m": np.zeros_like(param), "v": np.zeros_like(param)}
        state = self._adam_state[key]
        state["m"] = beta1 * state["m"] + (1 - beta1) * grad
        state["v"] = beta2 * state["v"] + (1 - beta2) * (grad ** 2)
        m_hat = state["m"] / (1 - beta1 ** self._t)
        v_hat = state["v"] / (1 - beta2 ** self._t)
        param -= lr * m_hat / (np.sqrt(v_hat) + eps)

    def apply_gradients(self, grads):
        self._t += 1
        lr = self.cfg.learning_rate

        self._adam_update("token_emb", self.token_emb, grads["token_emb"], lr)
        self._adam_update("pos_emb", self.pos_emb, grads["pos_emb"], lr)

        dgamma_f, dbeta_f = grads["ln_final"]
        self._adam_update("ln_final.gamma", self.ln_final.gamma, dgamma_f, lr)
        self._adam_update("ln_final.beta", self.ln_final.beta, dbeta_f, lr)

        dW_out, db_out = grads["out_proj"]
        self._adam_update("out_proj.W", self.out_proj.W, dW_out, lr)
        self._adam_update("out_proj.b", self.out_proj.b, db_out, lr)

        for i, (block, g) in enumerate(zip(self.blocks, grads["blocks"])):
            dgamma1, dbeta1 = g["ln1"]
            self._adam_update(f"b{i}.ln1.gamma", block.ln1.gamma, dgamma1, lr)
            self._adam_update(f"b{i}.ln1.beta", block.ln1.beta, dbeta1, lr)

            for name in ["q_proj", "k_proj", "v_proj", "out_proj"]:
                dW, db = g["attn"][name]
                layer = getattr(block.attn, name)
                self._adam_update(f"b{i}.attn.{name}.W", layer.W, dW, lr)
                self._adam_update(f"b{i}.attn.{name}.b", layer.b, db, lr)

            dgamma2, dbeta2 = g["ln2"]
            self._adam_update(f"b{i}.ln2.gamma", block.ln2.gamma, dgamma2, lr)
            self._adam_update(f"b{i}.ln2.beta", block.ln2.beta, dbeta2, lr)

            dW1, db1 = g["ffn"]["fc1"]
            self._adam_update(f"b{i}.ffn.fc1.W", block.ffn.fc1.W, dW1, lr)
            self._adam_update(f"b{i}.ffn.fc1.b", block.ffn.fc1.b, db1, lr)
            dW2, db2 = g["ffn"]["fc2"]
            self._adam_update(f"b{i}.ffn.fc2.W", block.ffn.fc2.W, dW2, lr)
            self._adam_update(f"b{i}.ffn.fc2.b", block.ffn.fc2.b, db2, lr)

    def train_step(self, token_ids, targets):
        loss, grads = self.loss_and_backward(token_ids, targets)
        self.apply_gradients(grads)
        return loss

    # -------------------------------------------------------------- inference
    def generate(self, prompt_ids: List[int], max_new_tokens: int, temperature: float = 1.0,
                 rng: Optional[np.random.Generator] = None) -> List[int]:
        rng = rng or np.random.default_rng(0)
        ids = list(prompt_ids)
        for _ in range(max_new_tokens):
            context = ids[-self.cfg.context_length:]
            x = np.array([context])
            logits, _ = self.forward(x)
            last_logits = logits[0, -1] / max(temperature, 1e-6)
            probs = softmax(last_logits[None, :], axis=-1)[0]
            next_id = int(rng.choice(len(probs), p=probs))
            ids.append(next_id)
        return ids

    def n_parameters(self) -> int:
        n = self.token_emb.size + self.pos_emb.size
        n += self.ln_final.gamma.size + self.ln_final.beta.size
        n += self.out_proj.W.size + self.out_proj.b.size
        for block in self.blocks:
            n += block.ln1.gamma.size + block.ln1.beta.size
            n += block.ln2.gamma.size + block.ln2.beta.size
            for layer in [block.attn.q_proj, block.attn.k_proj, block.attn.v_proj, block.attn.out_proj,
                          block.ffn.fc1, block.ffn.fc2]:
                n += layer.W.size + layer.b.size
        return n

    # -------------------------------------------------------------- state I/O
    def get_state(self) -> Dict[str, Any]:
        state = {
            "cfg": self.cfg.__dict__,
            "token_emb": self.token_emb.tolist(),
            "pos_emb": self.pos_emb.tolist(),
            "ln_final": [self.ln_final.gamma.tolist(), self.ln_final.beta.tolist()],
            "out_proj": [self.out_proj.W.tolist(), self.out_proj.b.tolist()],
            "blocks": [],
            "t": self._t,
        }
        for block in self.blocks:
            b_state = {
                "ln1": [block.ln1.gamma.tolist(), block.ln1.beta.tolist()],
                "ln2": [block.ln2.gamma.tolist(), block.ln2.beta.tolist()],
                "ffn_fc1": [block.ffn.fc1.W.tolist(), block.ffn.fc1.b.tolist()],
                "ffn_fc2": [block.ffn.fc2.W.tolist(), block.ffn.fc2.b.tolist()],
            }
            for name in ["q_proj", "k_proj", "v_proj", "out_proj"]:
                layer = getattr(block.attn, name)
                b_state[f"attn_{name}"] = [layer.W.tolist(), layer.b.tolist()]
            state["blocks"].append(b_state)
        return state

    @staticmethod
    def from_state(state: Dict[str, Any]) -> "TinyTransformer":
        cfg = TinyTransformerConfig(**state["cfg"])
        model = TinyTransformer(cfg)
        model.token_emb = np.array(state["token_emb"])
        model.pos_emb = np.array(state["pos_emb"])
        model.ln_final.gamma = np.array(state["ln_final"][0])
        model.ln_final.beta = np.array(state["ln_final"][1])
        model.out_proj.W = np.array(state["out_proj"][0])
        model.out_proj.b = np.array(state["out_proj"][1])
        for block, b_state in zip(model.blocks, state["blocks"]):
            block.ln1.gamma = np.array(b_state["ln1"][0])
            block.ln1.beta = np.array(b_state["ln1"][1])
            block.ln2.gamma = np.array(b_state["ln2"][0])
            block.ln2.beta = np.array(b_state["ln2"][1])
            block.ffn.fc1.W = np.array(b_state["ffn_fc1"][0])
            block.ffn.fc1.b = np.array(b_state["ffn_fc1"][1])
            block.ffn.fc2.W = np.array(b_state["ffn_fc2"][0])
            block.ffn.fc2.b = np.array(b_state["ffn_fc2"][1])
            for name in ["q_proj", "k_proj", "v_proj", "out_proj"]:
                layer = getattr(block.attn, name)
                layer.W = np.array(b_state[f"attn_{name}"][0])
                layer.b = np.array(b_state[f"attn_{name}"][1])
        model._t = state["t"]
        return model
