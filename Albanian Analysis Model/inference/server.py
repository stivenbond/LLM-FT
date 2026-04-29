from fastapi import FastAPI, HTTPException, Security, Depends, Request
from fastapi.security.api_key import APIKeyHeader
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Any, Dict
import os
import json
import time
import asyncio
from pathlib import Path
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup imports from other scripts
repo_root = Path(__file__).parent.parent
sys.path.append(str(repo_root / "scripts"))
from build_prompt import build_prompt, parse_response
from output_validator import validate_and_repair

app = FastAPI(title="Albanian Editor AI API")

# Security
API_KEY = os.environ.get("API_KEY")
api_key_header = APIKeyHeader(name="x-api-key", auto_error=False)

async def get_api_key(api_key_header: str = Security(api_key_header)):
    if not API_KEY:
        return api_key_header # No key configured, allow pass
    if api_key_header == API_KEY:
        return api_key_header
    raise HTTPException(status_code=403, detail="Could not validate API key")

# Global state for Multi-Model Support
class ModelState:
    models = {}  # model_id -> Llama instance
    startup_time = time.time()
    last_inference = {} # model_id -> timestamp
    config = {}

class AnalyzeRequest(BaseModel):
    model_id: str = "albanian_editor_v1"
    article_text: str = Field(..., min_length=100)
    brand_guidelines: str = ""
    key_points: List[str] = []
    client_id: Optional[str] = None
    stream: bool = False

class FeedbackRequest(BaseModel):
    session_id: str
    client_id: str
    input: Dict[str, Any]
    model_output: Dict[str, Any]
    feedback: Dict[str, Any]

@app.on_event("startup")
def load_models():
    config_path = os.environ.get("MODELS_CONFIG_PATH", str(repo_root / "inference" / "models_config.json"))
    if not Path(config_path).exists():
        print(f"WARNING: Config file not found at {config_path}")
        return
        
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            ModelState.config = json.load(f)
            
        from llama_cpp import Llama
        
        # We can eagerly load the default model or lazily load models. 
        # For this server, let's load all configured models if they exist.
        for model_id, cfg in ModelState.config.items():
            path = cfg.get("model_path")
            if path and Path(path).exists():
                try:
                    ModelState.models[model_id] = Llama(
                        model_path=path,
                        n_ctx=cfg.get("n_ctx", 4096),
                        n_threads=max(1, os.cpu_count() - 1)
                    )
                    ModelState.last_inference[model_id] = None
                    print(f"Model {model_id} loaded successfully from {path}.")
                except Exception as e:
                    print(f"Failed to load model {model_id}: {e}")
            else:
                print(f"Model file for {model_id} not found at {path}.")
    except ImportError:
        print("WARNING: llama-cpp-python not installed. Cannot load models.")
    except Exception as e:
        print(f"Error initializing models: {e}")

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "models_loaded": list(ModelState.models.keys()),
        "uptime_seconds": int(time.time() - ModelState.startup_time),
        "last_inference_timestamps": ModelState.last_inference
    }

@app.get("/schema")
def get_schema():
    schema_path = repo_root / "schemas" / "training_example_v1.json"
    if schema_path.exists():
        with open(schema_path, "r", encoding="utf-8") as f:
            return json.load(f)
    raise HTTPException(status_code=404, detail="Schema not found")

@app.post("/analyze")
async def analyze(req: AnalyzeRequest, api_key: str = Depends(get_api_key)):
    model_id = req.model_id
    if model_id not in ModelState.models:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not loaded or configured.")
        
    llm = ModelState.models[model_id]
    cfg = ModelState.config.get(model_id, {})
    
    prompt = build_prompt(
        article_text=req.article_text,
        brand_guidelines=req.brand_guidelines,
        key_points=req.key_points
    )
    
    ModelState.last_inference[model_id] = time.time()
    
    temperature = cfg.get("temperature", 0.1)

    if req.stream:
        # Server-Sent Events (SSE) Streaming
        async def event_generator():
            try:
                for output in llm(prompt, max_tokens=2048, temperature=temperature, stop=["<end_of_turn>"], stream=True):
                    token = output['choices'][0]['text']
                    yield f"data: {json.dumps({'token': token})}\n\n"
                    # Small sleep to allow async event loop to yield
                    await asyncio.sleep(0.001)
                yield "data: [DONE]\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
                
        return StreamingResponse(event_generator(), media_type="text/event-stream")
    else:
        # Synchronous full response
        response = llm(
            prompt,
            max_tokens=2048,
            temperature=temperature,
            stop=["<end_of_turn>"]
        )
        
        response_text = response['choices'][0]['text']
        parsed = parse_response(response_text)
        
        if parsed.get("parse_error"):
            return parsed
            
        repaired, modifications = validate_and_repair(parsed, key_points_empty=not req.key_points)
        return repaired

@app.post("/feedback")
async def collect_feedback(req: FeedbackRequest, api_key: str = Depends(get_api_key)):
    """Receives feedback from the PWA and saves it for RLHF DPO processing."""
    rlhf_dir = repo_root / "data" / "rlhf" / "collected"
    rlhf_dir.mkdir(parents=True, exist_ok=True)
    
    ts = int(time.time() * 1000)
    filename = f"{req.session_id}_{ts}.json"
    filepath = rlhf_dir / filename
    
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(req.dict(), f, indent=2, ensure_ascii=False)
        
    return {"status": "success", "message": "Feedback saved for RLHF processing."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
