"""Stage 2a: extract distinct perspectives from each question's answers with gpt-5-mini (OpenAI Batch API).

    python -m multi3ir.data.stage2_extract build   --src <stage1 dir> --out <dir>      # writes <dir>/<model>_batch_inputs.jsonl
    # upload the batch input to OpenAI, download the results, then
    python -m multi3ir.data.stage2_extract parse   --src <stage1 dir> --out <dir> --batch-output <results.jsonl>
    python -m multi3ir.data.stage2_extract direct  --src <stage1 dir> --out <dir>      # synchronous API instead of Batch

Output: <dir>/<model>_<date>.jsonl, one question per line with perspectives = [[summary, [(fact, link), ...]], ...].
"""
import argparse
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

PROMPT = """You are an expert at analyzing complex questions and their answers to identify distinct perspectives and information needs.

Given a question and a list of answers, extract the distinct perspectives (facets) covered in the answers as independent sub-questions. Each sub-question should be:
1. Self-contained: understandable without the original question
2. Search-friendly: suitable as a query to a search engine to find supporting evidence
3. Specific: targeting exactly one perspective or facet from the answers
4. Non-redundant: each sub-question should cover a distinct aspect

Do NOT generate sub-questions about information not present in the answers.
Do NOT generate overly broad or vague sub-questions.
Do NOT simply rephrase the original question.
Keep any relevant external links from the source answers.

Structure your response using the format below, with one <perspective> block per distinct viewpoint:
```
<perspective>
{{Short descriptive summary of the perspective as a search-friendly sub-question}}
    - {{Supporting fact or quote}} [source: {{URL if available, otherwise "NaN"}}]
    - ...
</perspective>
```

<question>
{Question}
</question>

{Answers}"""


def html_to_text(content):
    soup = BeautifulSoup(content, "html.parser")
    for a in soup.find_all("a", href=True):                  # keep links as "text (url)"
        a.replace_with(f"{a.get_text(strip=True)} ({a['href']})")
    return soup.get_text(separator=" ", strip=True)


def load_questions(base_path, n_per_category, seed):
    """Every *_filtered.jsonl under base_path (category = grandparent dir), sampled per category."""
    frames = []
    for root, _, files in os.walk(base_path):
        for file in files:
            if file.endswith(".jsonl"):
                df = pd.read_json(os.path.join(root, file), lines=True)
                df["category"] = Path(root).parent.name
                df["uid"] = df.index.astype(str) + "_" + file.replace(".jsonl", "")
                frames.append(df)
    if not frames:
        raise FileNotFoundError(f"no .jsonl under {base_path}")
    df = pd.concat(frames, ignore_index=True)
    if n_per_category:
        df = df.groupby("category", group_keys=False).apply(
            lambda x: x.sample(n=min(len(x), n_per_category), random_state=seed))
    for col in ("Title", "Body"):
        df[col] = df[col].apply(html_to_text)
    df["Answers"] = df["Answers"].apply(lambda answers: [html_to_text(a) for a in answers])
    df["Question"] = df["Title"] + " " + df["Body"]
    return df


def build_prompts(df):
    return [PROMPT.format(Question=r["Question"], Answers="\n\n".join(
        f"<answer {i+1}>\n{a}\n</answer {i+1}>" for i, a in enumerate(r["Answers"]))) for _, r in df.iterrows()]


def parse_perspectives(text):
    """<perspective> blocks -> [(summary, [(fact, link|None), ...]), ...]."""
    out = []
    for block in re.findall(r"<perspective>\s*(.*?)\s*</perspective>", text, re.DOTALL):
        lines = [l.strip() for l in block.splitlines() if l.strip()]
        facts = []
        for line in lines[1:]:
            line = line.lstrip("- ").strip()
            m = re.search(r"\[source:\s*(https?://[^\]]+)\]", line)
            facts.append((re.sub(r"\s*\[source:.*?\]", "", line).strip(), m.group(1)) if m else (line, None))
        out.append((lines[0], facts))
    return out


def complete(client, model, prompt):
    try:
        r = client.chat.completions.create(model=model, messages=[{"role": "user", "content": prompt}],
                                           max_completion_tokens=1024, temperature=0.7, n=1)
        return r.choices[0].message.content
    except Exception as e:
        print(f"  API error: {type(e).__name__}", flush=True)
        return ""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["build", "parse", "direct"])
    ap.add_argument("--src", required=True, help="stage-1 output root (category/site/*_filtered.jsonl)")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--model", default="gpt-5-mini-2025-08-07")
    ap.add_argument("--n-per-category", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--batch-output", default=None, help="parse: downloaded Batch API results jsonl")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    df = load_questions(args.src, args.n_per_category, args.seed)
    prompts = build_prompts(df)
    print(f"[{args.mode}] {len(df):,} questions", flush=True)
    if args.mode == "build":
        path = os.path.join(args.out, f"{args.model}_batch_inputs.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for uid, p in zip(df["uid"], prompts):
                f.write(json.dumps({"custom_id": uid, "method": "POST", "url": "/v1/chat/completions",
                                    "body": {"model": args.model, "messages": [{"role": "user", "content": p}]}},
                                   ensure_ascii=False) + "\n")
        print(f"[build] -> {path}")
        return
    if args.mode == "parse":
        if not args.batch_output:
            raise SystemExit("parse needs --batch-output")
        rows = [json.loads(l) for l in open(args.batch_output, encoding="utf-8")]
        if len(rows) != len(df):
            raise RuntimeError(f"{len(rows)} batch results for {len(df)} questions")
        df["perspectives"] = [parse_perspectives(r["response"]["body"]["choices"][0]["message"]["content"]) for r in rows]
    else:
        from openai import OpenAI
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        with ThreadPoolExecutor() as ex:
            outputs = list(ex.map(lambda p: complete(client, args.model, p), prompts))
        df["perspectives"] = [parse_perspectives(o) for o in outputs]
    path = os.path.join(args.out, f"{args.model}_{datetime.now():%Y%m%d}.jsonl")
    df.to_json(path, orient="records", lines=True)
    print(f"[{args.mode}] -> {path}")


if __name__ == "__main__":
    main()
