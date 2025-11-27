from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import logging
import torch
import sys
import os
import uuid
import numpy as np
from PIL import Image
import io
import json
import cv2

# Add root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from src.fusion_model import FusionModel
from src.visual_extractor import VisualExtractor, get_visual_embedding
from src.text_model import TextModel, get_text_embedding, Tokenizer
from src.ocr_tesseract import extract_ocr_data
from src.llm_integration import LLMIntegration
from src.interpretability import GradCAM, overlay_heatmap, gather_ocr_evidence
from src.preprocessing import get_transforms
from src.utils import setup_logger

# --- Setup ---
logger = setup_logger("MemeAPI")
app = FastAPI(title="Meme Deception Detection API", version="2.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static Mount for Heatmaps
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
HEATMAP_DIR = os.path.join(STATIC_DIR, "heatmaps")
os.makedirs(HEATMAP_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# --- Global Model State ---
MODELS = {}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_models():
    logger.info("Loading models...")
    
    # 1. Visual
    visual_ext = VisualExtractor(model_name='mobilenet_v2')
    visual_ext.to(DEVICE)
    visual_ext.eval()
    MODELS['visual'] = visual_ext
    
    # 2. Text
    # Initialize tokenizer with basic settings or load from file
    tokenizer = Tokenizer(num_words=10000)
    # In prod, we should load a fitted tokenizer: Tokenizer.load('data/tokenizer.pkl')
    # For now, we keep it raw/empty, assuming it handles OOV
    MODELS['tokenizer'] = tokenizer
    
    text_model = TextModel(vocab_size=10001, embedding_dim=128, hidden_dim=128)
    text_model.to(DEVICE)
    text_model.eval()
    MODELS['text'] = text_model

    # 3. Fusion (usa embeddings pré-calculados como em train.py/evaluate.py)
    config = {'num_classes': 2, 'text_output_dim': 256}
    fusion_model = FusionModel(config)  # use_precomputed_embeddings = True
    fusion_ckpt = "models/fusion_model.pth"
    if os.path.exists(fusion_ckpt):
        fusion_model.load_state_dict(torch.load(fusion_ckpt, map_location=DEVICE))
    fusion_model.to(DEVICE)
    fusion_model.eval()
    MODELS['fusion'] = fusion_model
    
    # 4. LLM
    MODELS['llm'] = LLMIntegration(mock_mode=False) # Will fallback to mock if no key
    
    logger.info("Models loaded.")

@app.on_event("startup")
async def startup_event():
    load_models()

# --- Endpoints ---

@app.get("/health")
def health_check():
    return {"status": "ok", "device": str(DEVICE), "models_loaded": list(MODELS.keys())}

@app.post("/infer")
async def infer(
    file: UploadFile = File(...),
    platform: str = Form("unknown")
):
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(400, detail="Invalid file type")
    
    request_id = str(uuid.uuid4())
    logger.info(f"Processing request {request_id} from platform {platform}")
    
    try:
        # 1. Read Image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        np_image = np.array(image)
        
        # 2. Pipeline: OCR
        ocr_result = extract_ocr_data(np_image)
        text_extracted = ocr_result['full_text']
        ocr_stats = [
            ocr_result['stats']['mean_conf'], 
            ocr_result['stats']['std_conf'], 
            len([w for w in ocr_result['words'] if w['conf'] < 50]) # low conf count
        ]
        ocr_evidence = gather_ocr_evidence(ocr_result['words'])
        
        # 3. Feature Extraction
        # Visual Embedding
        visual_emb = get_visual_embedding(image, model=MODELS['visual'], device=DEVICE)
        # Text Embedding
        text_emb = get_text_embedding(text_extracted, MODELS['tokenizer'], MODELS['text'], device=DEVICE)
        
        # Convert to tensors
        t_v_emb = torch.tensor([visual_emb], dtype=torch.float32).to(DEVICE)
        t_t_emb = torch.tensor([text_emb], dtype=torch.float32).to(DEVICE)
        t_ocr = torch.tensor([ocr_stats], dtype=torch.float32).to(DEVICE)
        
        # 4. Fusion Prediction
        with torch.no_grad():
            logits = MODELS['fusion'](t_v_emb, t_t_emb, t_ocr)
            probs = torch.softmax(logits, dim=1)
            score, pred_idx = torch.max(probs, 1)
            label = "Enganoso/Suspeito" if pred_idx.item() == 1 else "Autêntico"
            score_val = float(score.item())
            
        # 5. LLM Verification
        llm_meta = {
            "platform": platform,
            "metrics": "N/A (Auto-detected)", # Could extract via OCR regex
            "visual_cues": []
        }
        llm_response = MODELS['llm'].analyze(text_extracted, llm_meta)
        
        # 6. Interpretability: Grad-CAM
        # We need access to the CNN features layer. 
        # VisualExtractor exposes .backbone.features
        # Warning: GradCAM needs gradients, but we are in no_grad context usually for inference.
        # We need to re-run forward pass with grad enabled for just the visual part to get heatmap
        # Or instantiate GradCAM once.
        
        heatmap_url = None
        try:
            # Enable grad specifically for CAM
            with torch.enable_grad():
                target_layer = MODELS['visual'].features[-1] # Last conv layer of MobileNetV2 features
                cam = GradCAM(MODELS['visual'], target_layer)
                
                # Transform input
                transform = get_transforms(mode='eval')
                img_tensor = transform(image).unsqueeze(0).to(DEVICE)
                img_tensor.requires_grad = True
                
                # Generate heatmap (using class index 0 or max)
                mask = cam(img_tensor)
                
                # Overlay
                heatmap_img = overlay_heatmap(np_image, mask)
                
                # Save
                heatmap_filename = f"heatmap_{request_id}.png"
                heatmap_path = os.path.join(HEATMAP_DIR, heatmap_filename)
                cv2_img = cv2.cvtColor(heatmap_img, cv2.COLOR_RGB2BGR)
                import cv2
                cv2.imwrite(heatmap_path, cv2_img)
                
                # URL generation (assuming localhost or domain)
                heatmap_url = f"/static/heatmaps/{heatmap_filename}"
                
        except Exception as e:
            logger.error(f"Grad-CAM failed: {e}")
            heatmap_url = "Error generating heatmap"

        response = {
            "request_id": request_id,
            "label": label,
            "confidence_score": score_val,
            "ocr_text": text_extracted,
            "ocr_evidence": ocr_evidence['low_confidence_words'],
            "llm_explanation": llm_response,
            "heatmap_url": heatmap_url
        }
        
        return JSONResponse(content=response)

    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
