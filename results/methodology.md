# Methodology
## Hardware
- MacBook Pro, Apple M4 Pro, 24GB unified memory, macOS (Darwin 24.6).
- No CUDA. PyTorch on Metal Performance Shaders (MPS) for translation; ollama (llama.cpp) for LLM inference.

## Models
- **Translator (primary):** `facebook/nllb-200-distilled-1.3B`, fp16 on MPS, num_beams=4.
- **Translator (QC pass):** `Helsinki-NLP/opus-mt-en-{hi,sw,tl}` where available; second NLLB pass with num_beams=8 for `bn` and `yo` (no public Helsinki en→bn / en→yo model).
- **Generator:** `llama3.1:8b-instruct-q4_K_M` via ollama (Q4_K_M GGUF, ~4.9GB). temperature=0, top_p=1, seed=42, num_predict=512, no system prompt, no jailbreak scaffold.
- **Refusal classifier:** `llama-guard3:1b` via ollama. Multilingual by design. We pass the user prompt + assistant generation as a two-turn chat; ollama applies Llama-Guard's template; first token of the response is parsed (`safe`/`unsafe`).

## Dataset
- Source: HarmBench `data/behavior_datasets/harmbench_behaviors_text_all.csv` (n=400). Stratified sample of n=100 by `FunctionalCategory`, fixed seed=42.
- Sample composition: {'standard': 50, 'contextual': 25, 'copyright': 25}.
- Contextual rows: `ContextString` is prepended to `Behavior` with a blank-line separator before translation and generation, per HarmBench convention.
- Languages: en (source), hi, sw, bn, yo, tl. Total prompt-language pairs = 600.

## Translation quality (chrF agreement)
Per-language count of behaviors flagged with chrF < 0.3 between primary and QC translations (kept in dataset, surfaced in `low_agreement_langs`):
- Hindi (hi): 34
- Swahili (sw): 14
- Bengali (bn): 0
- Yoruba (yo): 6
- Tagalog (tl): 6

## Sanity checks
- Total classifications written: 600 (expected 600).
- All unsafe rates ∈ [0,1]: True.
- English unsafe rate: 20.0% (sanity threshold <20%).
  - **Flag:** English unsafe rate is 20.0%, above the 20% sanity threshold. Investigate generator prompting or classifier behavior before drawing language-gap conclusions.
- Languages with unsafe rate higher than English: none.
- Effect of dropping low-agreement translations:
  - Hindi: 13.0% (all) vs 10.6% (dropping low-agreement, n=66); Δ = 2.4pp (>2pp shift)
  - Swahili: 10.0% (all) vs 10.5% (dropping low-agreement, n=86); Δ = 0.5pp
  - Bengali: 15.0% (all) vs 15.0% (dropping low-agreement, n=100); Δ = 0.0pp
  - Yoruba: 7.0% (all) vs 5.3% (dropping low-agreement, n=94); Δ = 1.7pp
  - Tagalog: 19.0% (all) vs 18.1% (dropping low-agreement, n=94); Δ = 0.9pp

## Results table
See `results/refusal_rates.csv`. Reproduced here:

| language | n_total | n_unsafe | unsafe_rate | 95% Wilson CI | low-agreement n | unparsed/error |
|---|---:|---:|---:|---|---:|---:|
| English | 100 | 20 | 0.2000 | [0.1334, 0.2888] | 0 | 0 |
| Hindi | 100 | 13 | 0.1300 | [0.0776, 0.2098] | 34 | 0 |
| Swahili | 100 | 10 | 0.1000 | [0.0552, 0.1744] | 14 | 0 |
| Bengali | 100 | 15 | 0.1500 | [0.0931, 0.2328] | 0 | 0 |
| Yoruba | 100 | 7 | 0.0700 | [0.0343, 0.1375] | 6 | 0 |
| Tagalog | 100 | 19 | 0.1900 | [0.1251, 0.2778] | 6 | 0 |

## Per-FunctionalCategory breakdown (unsafe / total)

| language | contextual | copyright | standard | all |
|---|---:|---:|---:|---:|
| English | 0/25 | 20/25 | 0/50 | 20/100 |
| Hindi | 5/25 | 6/25 | 2/50 | 13/100 |
| Swahili | 6/25 | 2/25 | 2/50 | 10/100 |
| Bengali | 4/25 | 7/25 | 4/50 | 15/100 |
| Yoruba | 6/25 | 0/25 | 1/50 | 7/100 |
| Tagalog | 3/25 | 14/25 | 2/50 | 19/100 |

## Summary
On a stratified n=100 sample of HarmBench behaviors evaluated against `llama3.1:8b-instruct-q4_K_M` and judged by `llama-guard3:1b`, the aggregate English unsafe rate is 20.0% — but the breakdown shows this is driven entirely by the copyright subset (20/25 lyric-reproduction prompts), while the genuinely harmful standard and contextual subsets see 0/75 English compliance. Across non-English languages, the standard subset shows small absolute compliance (1–4 of 50 prompts, i.e. 2–8%) and the contextual subset shows a consistent jump (3–6 of 25 prompts, 12–24%) versus 0 in English; the copyright subset varies idiosyncratically by language. Read this as: for prompts that are unambiguously harmful in form (standard), the model's English-trained refusal generalises to translated variants but with a few percentage-point leak; for prompts that depend on translated context (contextual), the refusal generalises noticeably less well. Wilson 95% intervals (table) overlap heavily across languages at n=100, so claims about rank-ordering between non-English languages should be hedged. Translation-agreement diagnostics (chrF<0.3 between NLLB-1.3B and the QC translator) are reported per language; for Hindi the chrF is comparing two distinct translator families (NLLB vs Helsinki) and is naturally lower than the bn/yo cases where both passes are NLLB. Dropping low-agreement prompts shifts headline numbers by ≤2.4pp. Llama-Guard-3-1B is multilingual but smaller than the 8B variant; its non-English judgments carry more noise than English. All inference ran locally on a 24GB unified-memory M4 Pro with sequential model loading (peak <12GB RSS); the pipeline is reproducible end-to-end with `make all`.
