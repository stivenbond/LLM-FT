# Albanian Analysis Model Ecosystem

This repository contains the end-to-end data pipeline, evaluation scripts, RLHF infrastructure, and inference server for the **Albanian Analysis Model**. 

This model is designed to evaluate Albanian text across four registers (editorial, informational, marketing, and classical) and score them on grammar, writing style, formatting, brand compliance, marketing compliance, and structure.

The data generated here is specifically formatted to train models like **Gemma 4:e4b**, featuring native support for **Chain of Thought (CoT)** reasoning traces using `<think>` blocks.

## Project Structure

- `data/`: Contains seed articles, augmented data, synthetic JSON scaffolds, and final train/val/test splits.
- `prompts/`: Albanian-language prompt templates for the data generation teacher models.
- `schemas/`: The JSON schema (`training_example_v1.json`) defining the strict output structure.
- `scripts/`: The core data pipeline:
  - `validate_schema.py`: Validates and auto-fixes JSON outputs against the schema.
  - `generate_synthetic.py`: Calls teacher models (Groq/Anthropic/OpenAI) to generate data, or scaffolds empty JSONs for manual chat interfaces.
  - `augment.py`: Programmatically expands the dataset via error injection, register swaps, etc.
  - `split_dataset.py`: Generates stratified data splits for training.
  - `build_prompt.py`: Assembles the exact Gemma prompt format, including CoT instructions, and parses outputs.
  - `run_pipeline.py`: A master script that orchestrates the entire data pipeline.
- `inference/`: FastAPI server for production deployment.
  - `server.py`: Supports SSE streaming, multi-model loading from `models_config.json`, and API key auth.
  - `output_validator.py`: A safety net that fuzzy-matches and repairs malformed model outputs before they reach the client.
- `evals/`: Scripts to calculate per-task metrics (MAE, KL Divergence, etc.) against a test set.
- `rlhf/`: Pipeline for collecting user feedback and compiling Direct Preference Optimization (DPO) pairs.
- `tests/`: Pytest suite for core pipeline logic.

## Getting Started

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   *Note: If you plan to run the inference server, ensure you have `llama-cpp-python` installed, potentially with hardware acceleration enabled (e.g., cuBLAS or Metal).*

2. **Configure Environment**
   Copy `.env.example` to `.env` and populate your API keys:
   ```bash
   cp .env.example .env
   # Edit .env with your keys
   ```

## Running the Data Pipeline

You can run the entire data pipeline (Validation → Augmentation → Splitting) in one go:
```bash
python scripts/run_pipeline.py
```

Alternatively, you can open `Pipeline.ipynb` in Google Colab or Jupyter to run the steps interactively.

## Running the Inference Server

The server requires a downloaded GGUF model. Update `ALBANIAN_EDITOR_MODEL_PATH` in your `.env` file, then run:
```bash
uvicorn inference.server:app --host 0.0.0.0 --port 8000
```

### Server Endpoints
- `POST /analyze`: Submit an article for analysis. Supports SSE streaming if `"stream": true` is passed in the JSON body. Requires `x-api-key` header.
- `POST /feedback`: Submit negative user feedback from your frontend/PWA for future RLHF training.
- `GET /health`: Check server status and loaded models.
- `GET /schema`: Returns the JSON schema expected from the model.

## Running Tests

To verify the core components:
```bash
pytest tests/
```
