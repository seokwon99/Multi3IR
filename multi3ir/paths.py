"""Every location the code reads or writes. No other module builds an absolute path.

ROOT is this checkout ($MULTI3IR_ROOT overrides). Data lives under ROOT/results and
virtualenvs under ROOT/envs; on a shared machine both may be symlinks elsewhere.
"""
import os
from pathlib import Path

ROOT = Path(os.environ.get("MULTI3IR_ROOT") or Path(__file__).resolve().parents[1])
RESULTS = ROOT / "results"
BENCH = RESULTS / "bench"                # Multi³IR-Bench: train/ eval/ corpus/ document_embedding/ query_embedding/ precompute/
OUTPUTS = RESULTS / "outputs"            # benchmark-construction artifacts, one dir per stage
CKPTS = RESULTS / "ckpts"                # checkpoints, mined negatives, hidden-state caches
LOGS = RESULTS / "logs"                  # run logs (logs/training/<run>.log, logs/wandb)
OTHER_BENCH = RESULTS / "other_bench"    # BeRDS / PIR raw dumps, encodings, runs
HUMAN = RESULTS / "human_annotate"       # annotation sheets + judge verdicts
ENVS = ROOT / "envs"

# benchmark files
TRAIN_QUERIES = BENCH / "train/queries.jsonl"
TRAIN_POSITIVES = BENCH / "train/positives_gold.jsonl"
HELDOUT_UIDS = BENCH / "train/heldout_uids.txt"
EVAL_QUERIES = BENCH / "eval/queries_1039.jsonl"
EVAL_GOLD = BENCH / "eval/positives_specific.jsonl"
DOC_EMB = {"qwen": BENCH / "document_embedding/qwen3vl8b",
           "mmembed": BENCH / "document_embedding/mmembed",
           "gme": BENCH / "document_embedding/gme_qwen2vl_7b"}

# construction stages (multi3ir/data)
STAGE1 = OUTPUTS / "stage1_raw_data"
STAGE2 = OUTPUTS / "stage2_perspective_extraction"
PERSPECTIVES = STAGE2 / "perspective_extracted/final/all_perspectives_filtered_min4.jsonl"
STAGE3_C4 = OUTPUTS / "stage3_retrieval/c4"
STAGE3_SERPER = OUTPUTS / "stage3_retrieval/serper"
STAGE4 = OUTPUTS / "stage4_retrieval_filter"
DOMAIN = OUTPUTS / "domain_classification"

# one virtualenv per backbone (incompatible transformers pins); MULTI3IR_VENV_<NAME> overrides
VENVS = {"qwen": ENVS / "qwen3vl_embed", "mmembed": ENVS / "mmembed_env/.venv",
         "gme": ENVS / "gme_reference/.venv", "pyserini": ENVS / "pyserini21"}
QWEN_SRC = ENVS / "qwen3vl_embedding_repo/src"   # upstream Qwen3-VL-Embedding checkout (imported, not installed)


def env_python(backend: str) -> str:
    venv = Path(os.environ.get(f"MULTI3IR_VENV_{backend.upper()}") or VENVS[backend])
    py = venv / "bin/python"
    if not py.exists():
        raise FileNotFoundError(f"{backend} venv missing: {venv} (see README, Installation)")
    return str(py)


if __name__ == "__main__":
    for name in ("ROOT", "BENCH", "OUTPUTS", "CKPTS", "LOGS", "QWEN_SRC"):
        p = globals()[name]
        print(f"{name:9} {p}  {'ok' if p.exists() else 'MISSING'}")
    for b in VENVS:
        try:
            print(f"venv {b:9} {env_python(b)}")
        except FileNotFoundError:
            print(f"venv {b:9} MISSING")
