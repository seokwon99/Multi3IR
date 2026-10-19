"""BeRDS retrieval corpus: union of BM25 top-k Wikipedia (DPR-100w) passages over every perspective.

  envs/pyserini21/bin/python -m multi3ir.eval.berds.corpus --top-k 100 --splits dev test   # Java 21 in PATH
"""
import argparse
import json
import logging
import os
import time
from collections import defaultdict
from pathlib import Path

from multi3ir import paths as _P
from multi3ir.eval.berds.data import DOMAINS, ROOT, load_domain

log = logging.getLogger("berds.corpus")
INDEX = ROOT / "index/lucene-index.wikipedia-dpr-100w.20210120.d1b9e6"
JAR = "lib/python3.10/site-packages/pyserini/resources/jars/anserini-0.21.0-fatjar.jar"   # inside the pyserini venv


class LuceneSearcher:
    """anserini SimpleSearcher through jnius (pyserini's wrapper drags in torch/transformers)."""

    def __init__(self, index_path):
        os.environ.setdefault("CLASSPATH", str(Path(_P.env_python("pyserini")).parents[1] / JAR))
        from jnius import autoclass
        self._impl = autoclass("io.anserini.search.SimpleSearcher")(str(index_path))
        self._list = autoclass("java.util.ArrayList")

    @property
    def num_docs(self):
        return self._impl.get_total_num_docs()

    @staticmethod
    def _hit(h):
        doc = h.lucene_document
        raw = doc.get("raw")
        if raw:
            try:
                obj = json.loads(raw)
                text, title = obj.get("contents") or obj.get("text") or "", obj.get("title", "")
            except json.JSONDecodeError:
                text, title = raw, ""
        else:
            text, title = doc.get("contents") or "", doc.get("title") or ""
        return {"docid": str(h.docid), "score": float(h.score), "text": text, "title": title}

    def batch_search(self, queries, qids, k=100, threads=8):
        jq, ji = self._list(), self._list()
        for q in queries:
            jq.add(q)
        for i in qids:
            ji.add(i)
        res = self._impl.batch_search(jq, ji, int(k), int(threads))
        return {key: [self._hit(h) for h in (res.get(key) or [])] for key in qids}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-k", type=int, default=100)
    ap.add_argument("--splits", nargs="+", default=["dev", "test"])
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    out_dir = ROOT / "corpus"
    out_dir.mkdir(parents=True, exist_ok=True)

    items = [(dom, spl, f"{r['id']}-p{i}", p) for spl in args.splits for dom in DOMAINS
             for r in load_domain(dom, spl) for i, p in enumerate(r["perspectives"])]
    log.info("index %s; perspectives to query: %d", INDEX, len(items))
    searcher = LuceneSearcher(INDEX)
    docs, hits_per_doc, t0 = {}, defaultdict(int), time.time()
    with (out_dir / "perspective_bm25_top.jsonl").open("w") as fh:
        for s in range(0, len(items), args.batch):
            batch = items[s:s + args.batch]
            qids = [f"{dom}-{spl}-{qid}" for dom, spl, qid, _ in batch]
            res = searcher.batch_search([p for *_, p in batch], qids, k=args.top_k, threads=args.threads)
            for (dom, spl, qid, persp), key in zip(batch, qids):
                for h in res[key]:
                    docs.setdefault(h["docid"], {"docid": h["docid"], "title": h["title"], "text": h["text"]})
                    hits_per_doc[h["docid"]] += 1
                fh.write(json.dumps({"domain": dom, "split": spl, "persp_id": qid, "perspective": persp,
                                     "top": [{"docid": h["docid"], "score": h["score"]} for h in res[key]]},
                                    ensure_ascii=False) + "\n")
            if (s // args.batch) % 10 == 0:
                log.info("  %d/%d  %.1f q/s  unique docs=%d", s + len(batch), len(items),
                         (s + len(batch)) / max(time.time() - t0, 1e-3), len(docs))
    with (out_dir / "corpus.jsonl").open("w") as f:
        for doc in docs.values():
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
    dist = defaultdict(int)
    for c in hits_per_doc.values():
        dist[c] += 1
    (out_dir / "stats.json").write_text(json.dumps({
        "index": str(INDEX), "top_k": args.top_k, "splits": args.splits, "n_perspectives_queried": len(items),
        "n_unique_docs": len(docs), "hit_count_dist_top10": sorted(dist.items())[:10]}, indent=2))
    log.info("done: %d docs -> %s", len(docs), out_dir)


if __name__ == "__main__":
    main()
