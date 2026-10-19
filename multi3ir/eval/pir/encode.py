"""Single-vector text encoders for the external benches: encode_queries / encode_docs -> (N, D) unit rows."""
import numpy as np
import torch
import torch.nn.functional as F

from multi3ir import paths as _P


class Qwen3VLEncoder:
    """Qwen3-VL-Embedding-8B, optionally with fine-tuned weights (--base-ckpt)."""
    QUERY_INSTRUCTION = "Retrieve passages relevant to the query."

    def __init__(self, base_ckpt=None, device="cuda", max_length=512, batch_size=8):
        from multi3ir.training.backbones.qwen import Qwen3VLTextSplitEncoder, load_embedder, make_text_tokenizer
        self.name = "qwen3vl-8b-" + (base_ckpt.rsplit("/", 2)[-2] if base_ckpt else "raw")
        emb = load_embedder(max_length=max_length)
        if base_ckpt:
            ck = torch.load(base_ckpt, map_location="cpu", weights_only=False)
            miss, unex = emb.model.load_state_dict(ck["state"] if "state" in ck else ck, strict=False)
            print(f"[base] {base_ckpt} (missing={len(miss)}, unexpected={len(unex)})", flush=True)
        emb.model = emb.model.to(device)
        self.device, self.batch_size = device, batch_size
        self.tok_q = make_text_tokenizer(emb, instruction=self.QUERY_INSTRUCTION)
        self.tok_d = make_text_tokenizer(emb, instruction=None)
        self.enc = Qwen3VLTextSplitEncoder(emb).to(device).eval()

    @torch.no_grad()
    def _encode(self, texts, tok):
        out = []
        for s in range(0, len(texts), self.batch_size):
            inputs = {k: v.to(self.device) for k, v in tok(texts[s:s + self.batch_size]).items()}
            out.append(self.enc.encode_full(inputs).float().cpu().numpy())
        return np.concatenate(out)

    def encode_queries(self, texts): return self._encode(texts, self.tok_q)
    def encode_docs(self, texts): return self._encode(texts, self.tok_d)


class MMEmbedEncoder:
    """nvidia/MM-Embed, text only (mmembed venv)."""
    name = "mm-embed"
    QUERY_INSTRUCTION = "Given a question, retrieve passages that answer the question"

    def __init__(self, device="cuda", max_length=2048, batch_size=4):
        from transformers import AutoModel
        self.model = AutoModel.from_pretrained("nvidia/MM-Embed", trust_remote_code=True, torch_dtype=torch.bfloat16,
                                               attn_implementation="sdpa").to(device).eval()
        self.max_length, self.batch_size = max_length, batch_size

    @torch.no_grad()
    def _encode(self, texts, is_query):
        out = []
        for s in range(0, len(texts), self.batch_size):
            kw = {"is_query": True, "instruction": self.QUERY_INSTRUCTION} if is_query else {}
            res = self.model.encode([{"txt": t} for t in texts[s:s + self.batch_size]], max_length=self.max_length, **kw)
            h = res["hidden_states"] if isinstance(res, dict) else res
            out.append(F.normalize(h, dim=-1).float().cpu().numpy())
        return np.concatenate(out)

    def encode_queries(self, texts): return self._encode(texts, True)
    def encode_docs(self, texts): return self._encode(texts, False)


class GMEEncoder:
    """GME-Qwen2-VL-7B through vLLM embed, text only."""
    name = "gme-qwen2vl-7b"
    INSTRUCTION = "Retrieve images or text relevant to the user's query."

    def __init__(self, max_length=4096, gpu_mem=0.90, **_):
        from transformers import AutoTokenizer
        from vllm import LLM
        model = str(_P.CKPTS / "gme_qwen2vl_7b_clean")
        self.tok = AutoTokenizer.from_pretrained(model, trust_remote_code=False)
        self.llm = LLM(model=model, task="embed", trust_remote_code=False, max_model_len=max_length,
                       dtype="bfloat16", gpu_memory_utilization=gpu_mem)
        self.max_length = max_length

    def _wrap(self, text):
        ids = self.tok(text, max_length=self.max_length - 32, truncation=True, add_special_tokens=False)["input_ids"]
        return (f"<|im_start|>system\n{self.INSTRUCTION}<|im_end|>\n<|im_start|>user\n"
                f"{self.tok.decode(ids, skip_special_tokens=False)}<|im_end|>\n<|im_start|>assistant\n<|endoftext|>")

    def _encode(self, texts):
        emb = np.asarray([o.outputs.embedding for o in self.llm.embed([self._wrap(t) for t in texts])], dtype=np.float32)
        n = np.linalg.norm(emb, axis=1, keepdims=True)
        n[n == 0] = 1
        return emb / n

    encode_queries = encode_docs = _encode


class BM25Encoder:
    """Scores directly: score_matrix(queries, docs) -> (Nq, Nd)."""
    name = "bm25"

    def score_matrix(self, queries, docs):
        from rank_bm25 import BM25Okapi
        bm25 = BM25Okapi([d.split() for d in docs])
        return np.stack([bm25.get_scores(q.split()) for q in queries])


ENCODERS = {"qwen3vl-raw": lambda device, base_ckpt: Qwen3VLEncoder(None, device),
            "qwen3vl-base": lambda device, base_ckpt: Qwen3VLEncoder(base_ckpt, device),
            "mm-embed": lambda device, base_ckpt: MMEmbedEncoder(device),
            "gme": lambda device, base_ckpt: GMEEncoder(),
            "bm25": lambda device, base_ckpt: BM25Encoder()}


def load_encoder(name, device="cuda", base_ckpt=None):
    if name == "qwen3vl-base" and not base_ckpt:
        raise ValueError("qwen3vl-base needs --base-ckpt")
    return ENCODERS[name](device, base_ckpt)
