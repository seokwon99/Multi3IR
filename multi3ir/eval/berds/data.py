"""BeRDS (Chen & Choi 2024) questions + perspectives; `python -m multi3ir.eval.berds.data` downloads them."""
import json

from multi3ir import paths as _P

ROOT = _P.OTHER_BENCH / "berds"        # raw/ corpus/ index/ encodings/ runs/ judge_cache*.jsonl
DOMAINS = ["arguana", "kialo", "opinionqa"]
SPLITS = ["dev", "test"]


def load_domain(name, split="test"):
    """rows: {"id", "question", "perspectives", "type"}."""
    return [json.loads(l) for l in open(ROOT / "raw" / f"{name}.{split}.jsonl")]


def main():
    from huggingface_hub import hf_hub_download
    (ROOT / "raw").mkdir(parents=True, exist_ok=True)
    for d in DOMAINS:
        for s in SPLITS:
            local = hf_hub_download(repo_id="timchen0618/BERDS", filename=f"{d}.{s}.jsonl",
                                    repo_type="dataset", local_dir=str(ROOT / "raw"))
            print(f"  {d}.{s}: {sum(1 for _ in open(local))} rows -> {local}")


if __name__ == "__main__":
    main()
