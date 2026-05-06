PY := .venv/bin/python
PIP := .venv/bin/pip
OLLAMA := /opt/homebrew/bin/ollama
GEN_MODEL := llama3.1:8b-instruct-q4_K_M
GUARD_MODEL := llama-guard3:1b

.PHONY: all setup deps harmbench ollama-models sample translate generate classify report clean-results

all: setup sample translate generate classify report

setup: deps harmbench ollama-models

deps: .venv/.installed

.venv/.installed:
	python3 -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install torch transformers sentencepiece sacremoses sacrebleu pandas statsmodels requests tqdm
	@touch $@

harmbench:
	@if [ ! -d HarmBench ]; then git clone --depth 1 https://github.com/centerforaisafety/HarmBench.git HarmBench; fi

ollama-models:
	@command -v $(OLLAMA) >/dev/null || (echo "ollama not found. Run: brew install ollama"; exit 1)
	@pgrep -x ollama >/dev/null || ($(OLLAMA) serve >/tmp/ollama.log 2>&1 & sleep 3)
	$(OLLAMA) pull $(GEN_MODEL)
	$(OLLAMA) pull $(GUARD_MODEL)

sample: data/sampled_behaviors.csv
data/sampled_behaviors.csv: scripts/sample_behaviors.py
	$(PY) scripts/sample_behaviors.py

translate: data/behaviors_multilingual.csv
data/behaviors_multilingual.csv: data/sampled_behaviors.csv scripts/translate_behaviors.py
	$(PY) scripts/translate_behaviors.py

generate: results/generations.jsonl
results/generations.jsonl: data/behaviors_multilingual.csv scripts/generate.py
	$(PY) scripts/generate.py

classify: results/classifications.jsonl
results/classifications.jsonl: results/generations.jsonl scripts/classify.py
	$(PY) scripts/classify.py

report: results/refusal_rates.csv results/methodology.md
results/refusal_rates.csv results/methodology.md: results/classifications.jsonl scripts/report.py
	$(PY) scripts/report.py

clean-results:
	rm -f results/generations.jsonl results/classifications.jsonl results/refusal_rates.csv results/methodology.md
