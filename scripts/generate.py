"""Run 600 generations (100 prompts × 6 langs) via the local ollama HTTP API.

Generator: llama3.1:8b-instruct-q4_K_M (Q4_K_M GGUF). T=0, max=512, no system prompt.
"""

import argparse
import csv
import json
import os
import time
from pathlib import Path

import requests
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "behaviors_multilingual.csv"
OUT = ROOT / "results" / "generations.jsonl"

OLLAMA = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
MODEL = "llama3.1:8b-instruct-q4_K_M"
LANGS = ["en", "hi", "sw", "bn", "yo", "tl"]


def chat_once(prompt, model=MODEL, max_tokens=512, timeout=300):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {
            "temperature": 0.0,
            "top_p": 1.0,
            "seed": 42,
            "num_predict": max_tokens,
        },
    }
    r = requests.post(f"{OLLAMA}/api/chat", json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()


def already_done(out_path):
    """Resume support: skip (behavior_id, language) pairs already in OUT."""
    done = set()
    if not out_path.exists():
        return done
    with open(out_path) as f:
        for line in f:
            try:
                obj = json.loads(line)
                done.add((obj["behavior_id"], obj["language"]))
            except Exception:
                continue
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-tokens", type=int, default=512)
    args = ap.parse_args()

    rows = list(csv.DictReader(open(SRC)))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    done = already_done(OUT)
    print(f"resuming with {len(done)} already-completed pairs", flush=True)

    pairs = [(r, lang) for r in rows for lang in LANGS]
    pairs = [(r, l) for (r, l) in pairs if (r["behavior_id"], l) not in done]
    print(f"running {len(pairs)} generations on {MODEL}", flush=True)

    t0 = time.time()
    with open(OUT, "a") as f:
        for r, lang in tqdm(pairs, desc="generate"):
            prompt = r[lang]
            try:
                resp = chat_once(prompt, max_tokens=args.max_tokens)
                gen = resp.get("message", {}).get("content", "")
                finish = resp.get("done_reason", resp.get("done", "unknown"))
                notes = ""
            except Exception as e:
                gen = ""
                finish = "error"
                notes = repr(e)
            obj = {
                "behavior_id": r["behavior_id"],
                "language": lang,
                "prompt": prompt,
                "generation": gen,
                "finish_reason": finish,
                "notes": notes,
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            f.flush()
    print(f"done in {(time.time()-t0)/60:.1f} min -> {OUT}")


if __name__ == "__main__":
    main()
