"""PIR (Zhao et al. 2024), BEIR-style per domain; `python -m multi3ir.eval.pir.data` downloads it."""
import json
from collections import defaultdict

from multi3ir import paths as _P

ROOT = _P.OTHER_BENCH / "pir"          # raw/<domain>/{queries.jsonl,corpus.jsonl,qrels.tsv}  runs/
DOMAINS = ["ambigqa", "story", "perspectrum", "exfever", "allsides", "agnews"]
HF_PREFIX = "trumancai/perspective-information-retrieval"


def _meta(r):
    return json.loads(r["meta"]) if isinstance(r.get("meta"), str) else (r.get("meta") or {})


def load_domain(domain):
    """-> (queries: qid -> {text, meta}, corpus: did -> {title, text, meta}, qrels: qid -> {did: score})."""
    base = ROOT / "raw" / domain
    queries = {r["_id"]: {"text": r["text"], "meta": _meta(r)} for r in map(json.loads, open(base / "queries.jsonl"))}
    corpus = {r["_id"]: {"title": r.get("title", ""), "text": r["text"], "meta": _meta(r)}
              for r in map(json.loads, open(base / "corpus.jsonl"))}
    qrels = defaultdict(dict)
    with open(base / "qrels.tsv") as f:
        assert f.readline().strip().split("\t") == ["query-id", "corpus-id", "score"]
        for line in f:
            qid, did, score = line.rstrip("\n").split("\t")
            qrels[qid][did] = int(score)
    return queries, corpus, dict(qrels)


def main():
    from datasets import load_dataset
    for domain in DOMAINS:
        repo, dst = f"{HF_PREFIX}-{domain}", ROOT / "raw" / domain
        dst.mkdir(parents=True, exist_ok=True)
        for cfg in ("queries", "corpus"):
            ds = load_dataset(repo, cfg)
            ds = ds[list(ds.keys())[0]]
            with (dst / f"{cfg}.jsonl").open("w") as f:
                for r in ds:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
        qrels = load_dataset(repo, split="test")
        with (dst / "qrels.tsv").open("w") as f:
            f.write("query-id\tcorpus-id\tscore\n")
            for r in qrels:
                f.write(f"{r['query-id']}\t{r['corpus-id']}\t{int(r['score'])}\n")
        print(f"  {domain:12s}  qrels={len(qrels)}")


if __name__ == "__main__":
    main()
