"""Translate sampled HarmBench behaviors into target languages.

Primary: facebook/nllb-200-distilled-1.3B with num_beams=4 on MPS.
QC pass: Helsinki-NLP/opus-mt-en-{xx} where the per-language model exists,
otherwise a second NLLB pass with num_beams=8.
chrF (sacrebleu) computed between the two translations as an agreement signal.
"""

import argparse
import csv
import gc
import time
from pathlib import Path

import torch
from sacrebleu.metrics import CHRF
from tqdm import tqdm
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    pipeline,
)

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "sampled_behaviors.csv"
OUT = ROOT / "data" / "behaviors_multilingual.csv"

# NLLB-200 BCP-47-ish codes
NLLB_LANG = {
    "hi": "hin_Deva",
    "sw": "swh_Latn",
    "bn": "ben_Beng",
    "yo": "yor_Latn",
    "tl": "tgl_Latn",
}

# Helsinki opus-mt model availability per target lang.
# en-bn / en-yo do not exist as standalone Helsinki models on the hub at the time
# of writing; for those we use a second NLLB pass with different beams.
HELSINKI = {
    "hi": "Helsinki-NLP/opus-mt-en-hi",
    "sw": "Helsinki-NLP/opus-mt-en-sw",
    "tl": "Helsinki-NLP/opus-mt-en-tl",
    # bn, yo: no per-language model -> use second NLLB pass
}

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "mps" else torch.float32

NLLB_ID = "facebook/nllb-200-distilled-1.3B"


def build_prompt_text(row):
    """HarmBench convention: prepend ContextString for contextual rows."""
    ctx = (row.get("ContextString") or "").strip()
    beh = row["Behavior"].strip()
    if ctx:
        return f"{ctx}\n\n{beh}"
    return beh


def load_nllb():
    print(f"[NLLB] loading on {DEVICE} ({DTYPE})", flush=True)
    tok = AutoTokenizer.from_pretrained(NLLB_ID)
    model = AutoModelForSeq2SeqLM.from_pretrained(NLLB_ID, torch_dtype=DTYPE).to(DEVICE)
    model.eval()
    return tok, model


@torch.inference_mode()
def nllb_translate_batch(tok, model, texts, tgt_code, num_beams, max_new_tokens=512):
    tok.src_lang = "eng_Latn"
    enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(DEVICE)
    forced_bos = tok.convert_tokens_to_ids(tgt_code)
    out = model.generate(
        **enc,
        forced_bos_token_id=forced_bos,
        num_beams=num_beams,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    return tok.batch_decode(out, skip_special_tokens=True)


def nllb_translate_all(tok, model, texts, tgt_code, num_beams, batch_size=4):
    out = []
    for i in tqdm(range(0, len(texts), batch_size), desc=f"nllb b={num_beams} {tgt_code}"):
        chunk = texts[i : i + batch_size]
        out.extend(nllb_translate_batch(tok, model, chunk, tgt_code, num_beams))
    return out


def helsinki_translate(model_id, texts, batch_size=8):
    print(f"[Helsinki] {model_id} on {DEVICE}", flush=True)
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id, torch_dtype=DTYPE).to(DEVICE)
    model.eval()
    out = []
    with torch.inference_mode():
        for i in tqdm(range(0, len(texts), batch_size), desc=f"helsinki {model_id.split('-')[-1]}"):
            chunk = texts[i : i + batch_size]
            enc = tok(chunk, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(DEVICE)
            gen = model.generate(**enc, num_beams=4, max_new_tokens=512, do_sample=False)
            out.extend(tok.batch_decode(gen, skip_special_tokens=True))
    del model, tok
    gc.collect()
    if DEVICE == "mps":
        torch.mps.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=4)
    args = ap.parse_args()

    rows = list(csv.DictReader(open(SRC)))
    en_texts = [build_prompt_text(r) for r in rows]
    print(f"loaded {len(rows)} behaviors", flush=True)

    chrf = CHRF()

    # Primary: NLLB beams=4 for all 5 langs
    tok, nllb = load_nllb()
    primary = {}
    for lang, code in NLLB_LANG.items():
        t0 = time.time()
        primary[lang] = nllb_translate_all(tok, nllb, en_texts, code, num_beams=4, batch_size=args.batch_size)
        print(f"  primary {lang} done in {time.time()-t0:.1f}s", flush=True)

    # QC: NLLB beams=8 for the langs without Helsinki coverage
    qc = {}
    for lang in ["bn", "yo"]:
        t0 = time.time()
        qc[lang] = nllb_translate_all(tok, nllb, en_texts, NLLB_LANG[lang], num_beams=8, batch_size=args.batch_size)
        print(f"  qc nllb-b8 {lang} done in {time.time()-t0:.1f}s", flush=True)

    # Free NLLB before loading Helsinki models
    del nllb, tok
    gc.collect()
    if DEVICE == "mps":
        torch.mps.empty_cache()

    # QC: Helsinki where available
    for lang, model_id in HELSINKI.items():
        try:
            qc[lang] = helsinki_translate(model_id, en_texts, batch_size=8)
        except Exception as e:
            print(f"  helsinki {lang} failed ({e!r}); falling back to NLLB beams=8", flush=True)
            tok2, nllb2 = load_nllb()
            qc[lang] = nllb_translate_all(tok2, nllb2, en_texts, NLLB_LANG[lang], num_beams=8, batch_size=args.batch_size)
            del tok2, nllb2
            gc.collect()
            if DEVICE == "mps":
                torch.mps.empty_cache()

    # Per-prompt chrF between primary and qc
    chrfs = {lang: [] for lang in NLLB_LANG}
    for lang in NLLB_LANG:
        for p, q in zip(primary[lang], qc[lang]):
            score = chrf.sentence_score(p, [q]).score / 100.0
            chrfs[lang].append(score)

    # Write output
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        ["behavior_id", "category", "en"]
        + list(NLLB_LANG.keys())
        + [f"{lang}_chrf" for lang in NLLB_LANG]
        + ["low_agreement_langs"]
    )
    flagged_per_lang = {lang: 0 for lang in NLLB_LANG}
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for i, r in enumerate(rows):
            low = []
            for lang in NLLB_LANG:
                if chrfs[lang][i] < 0.3:
                    low.append(lang)
                    flagged_per_lang[lang] += 1
            row_out = {
                "behavior_id": r["BehaviorID"],
                "category": r["FunctionalCategory"],
                "en": en_texts[i],
                **{lang: primary[lang][i] for lang in NLLB_LANG},
                **{f"{lang}_chrf": f"{chrfs[lang][i]:.4f}" for lang in NLLB_LANG},
                "low_agreement_langs": ",".join(low),
            }
            w.writerow(row_out)
    print(f"\nwrote {OUT}")
    print("low-agreement (chrF<0.3) per language:", flagged_per_lang)


if __name__ == "__main__":
    main()
