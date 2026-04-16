# Albanian Editor AI

Multi-task Albanian text analysis model. Performs 6 editorial checks in a
single inference pass and returns a combined JSON result.

## Tasks
1. **Grammar** — morphosyntactic compliance
2. **Writing Style** — register detection and consistency
3. **Formatting** — paragraphing and punctuation
4. **Brand Compliance** — scored against injected client guidelines
5. **Marketing Compliance** — scored against marketing best practices
6. **Structure** — key point coverage and elaboration linearity

## Quick Start
```bash
# 1. Set up workspace (already done if you ran setup_workspace.sh)
bash setup_workspace.sh

# 2. Install dependencies
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 3. Read the data guide before doing anything else
cat docs/DATA_GUIDE.md
```

## Pipeline Order
```
seed articles
    ↓
scripts/generate_synthetic.py    # generate raw training data
    ↓
[human review → data/gold/]      # most critical step
    ↓
scripts/augment.py               # expand to 2000+ examples
    ↓
scripts/validate_schema.py       # gate check before training
    ↓
scripts/split_dataset.py         # train/val/test splits
    ↓
[training — see training/configs/]
    ↓
[GGUF export — see docs/EXPORT.md]
    ↓
inference/server.py              # serve on CPU
```

## Docs
- `docs/DATA_GUIDE.md` — annotation rules and workflow
- `docs/EXPORT.md` — how to convert to GGUF for CPU deployment
- `docs/RLHF_GUIDE.md` — DPO preference fine-tuning
- `scripts/SPECS.md` — full specification for every script

## Schema version
Current: `schemas/training_example_v1.json` v1.0.0
