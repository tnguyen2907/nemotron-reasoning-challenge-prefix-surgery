# Prefix Surgery

This repository implements my **prefix surgery** submission to the NVIDIA
Nemotron Model Reasoning Challenge.

The idea is simple: when a model's reasoning is correct until one bad step, keep
the correct prefix and train only a repaired continuation. I tested that idea on
the competition's symbol-cipher task against a control that rewrites the whole
solution from scratch.

The experiment did not improve the target task, but prefix surgery caused much
less forgetting than the clean rewrite. The method, results, and failure
analysis are described in [WRITEUP.md](WRITEUP.md).

## Repository layout

```text
nemotron_reasoning/
  symbol_cipher.py     deterministic symbol-cipher solver
  trace_surgeon.py     adaptive prefix cut and Arm A/Arm B trace generation
  corpus_build.py      corpus admission, replay anchors, and verification
  training.py          rank-32 completion-only LoRA training
  inference.py         vLLM evaluation
  eval_analysis.py     full and held-out comparison tables
  metric.py            answer extraction and scoring
  prompts.py           competition prompt formatting
  make_splits.py       deterministic held-out split
  train_worker.py      distributed training entry point
notebooks/
  full_workflow.ipynb  complete Modal workflow
requirements.txt       local notebook dependencies
WRITEUP.md             solution story, results, and sources
```

## Run the workflow

Install the local dependencies, authenticate Modal and KaggleHub, and place the
competition `train.csv` at `data/train.csv`:

```bash
pip install -r requirements.txt
modal token new
```

Open `notebooks/full_workflow.ipynb` and run it from top to bottom. The notebook:

1. downloads and validates the baseline adapter;
2. generates baseline predictions;
3. solves the symbol-cipher rows;
4. builds and verifies Arm A and Arm B;
5. trains both adapters concurrently; and
6. evaluates both adapters on all 9,500 rows and the held-out split.

Training uses two H200s per arm, with both arms running concurrently for a peak
of four H200s. Evaluation uses one H200 per arm for a peak of two H200s. A full
run takes roughly 8-12 hours depending on image startup, queueing, and
throughput.
