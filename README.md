# HarmBench × Global South languages — small empirical study

End-to-end pipeline that measures whether a current open LLM refuses harmful
prompts less reliably in five Global South languages than in English. All
inference runs locally on Apple Silicon (M-series, 18–36GB unified memory).

## Stack

- **Translator:** `facebook/nllb-200-distilled-1.3B` (HF transformers on MPS), QC pass with `Helsinki-NLP/opus-mt-en-{hi,sw,tl}` plus second NLLB pass with different beam settings for `bn`/`yo`.
- **Generator:** `llama3.1:8b-instruct-q4_K_M` via ollama.
- **Refusal classifier:** `llama-guard3:1b` via ollama.
- **Stats:** Wilson 95% CIs (`statsmodels`).

## Layout

```
harmbench_multilingual/
  HarmBench/                    # cloned repo (MIT)
  scripts/
    sample_behaviors.py
    translate_behaviors.py
    generate.py
    classify.py
    report.py
  data/
    sampled_behaviors.csv       # 100 stratified behaviors, seed=42
    behaviors_multilingual.csv  # en + 5 translations + per-lang chrF
  results/
    generations.jsonl
    classifications.jsonl
    refusal_rates.csv
    methodology.md
  Makefile
  README.md
```

## One-shot reproduction

```bash
brew install ollama         # one-time
make all
```

`make all` performs: `deps` → clone HarmBench → `ollama pull` (generator + guard) → `sample` → `translate` → `generate` → `classify` → `report`.

`generate` and `classify` are **resumable**: re-running picks up where they left off (existing `(behavior_id, language)` pairs in the JSONL files are skipped).

## Per-step commands

```bash
make sample            # data/sampled_behaviors.csv
make translate         # data/behaviors_multilingual.csv (~30–45 min)
make generate          # results/generations.jsonl (~2–3h, 600 calls)
make classify          # results/classifications.jsonl (~20–40 min)
make report            # results/refusal_rates.csv + results/methodology.md
make clean-results     # wipe just the results, keep deps and translations
```

## Notes

- HarmBench is MIT-licensed; the public `harmbench_behaviors_text_all.csv` is what we sample from.
- Behaviors are adversarial by design and are passed verbatim — no paraphrasing or softening.
- ollama applies each model's chat template via `/api/chat`; no custom prompting beyond the user message.
- Memory budget: peak RSS stays under 12GB because translator → generator → classifier are loaded sequentially (translator unloaded before `make generate`; ollama swaps between generator and guard at the model-tag boundary).
