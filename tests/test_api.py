import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from src.api.main import app
import os
import io
from PIL import Image
import numpy as np

client = TestClient(app)

def create_dummy_image():
    file = io.BytesIO()
    image = Image.new('RGB', (100, 100), color='red')
    image.save(file, 'jpeg')
    file.name = 'test.jpg'
    file.seek(0)
    return file

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

@patch("src.api.main.extract_ocr_data")
@patch("src.api.main.get_visual_embedding")
@patch("src.api.main.get_text_embedding")
@patch("src.api.main.LLMIntegration")
def test_infer_endpoint(mock_llm, mock_text_emb, mock_vis_emb, mock_ocr):
    # Mock responses
    mock_ocr.return_value = {
        'full_text': "Fake news",
        'stats': {'mean_conf': 90.0, 'std_conf': 5.0},
        'words': [{'text': 'Fake', 'conf': 90.0, 'bbox': (0,0,10,10)}]
    }
    
    mock_vis_emb.return_value = np.zeros(1280, dtype=np.float32)
    mock_text_emb.return_value = np.zeros(128, dtype=np.float32) # Dim matches TextModel default
    
    # Mock LLM instance behavior
    llm_instance = mock_llm.return_value
    llm_instance.analyze.return_value = {
        "label": "suspeito",
        "score": 0.9,
        "explanation": "Mock explanation",
        "issues": []
    }
    
    # Prepare Request
    img_file = create_dummy_image()
    
    # We also need to mock the global MODELS in api/main because they are loaded at startup
    # TestClient triggers startup event, so MODELS are populated.
    # However, inside the endpoint, we use global MODELS.
    # The patch decorator above mocks the functions called, but the FusionModel forward pass 
    # is called on MODELS['fusion'].
    # Since models are loaded real-time in startup, inference might fail if inputs don't match dimensions exactly.
    # Let's rely on the fact that the real models are loaded (untrained) and will produce SOME output.
    # We just need to ensure the mocked intermediate functions return data shapes that don't crash PyTorch.
    
    # FusionModel expects tensors. 
    # In api/main: 
    # t_v_emb = torch.tensor([visual_emb]) -> [1, 1280]
    # t_t_emb = torch.tensor([text_emb]) -> [1, 128] (Needs to match TextModel output dim)
    # Our TextModel in api/main is init with hidden=128, output is linear(hidden*2, 256)? 
    # Wait, let's check src/text_model.py.
    # TextModel __init__ default output_dim=256.
    # So our mock_text_emb should return 256 dim vector.
    mock_text_emb.return_value = np.zeros(256, dtype=np.float32)
    
    response = client.post(
        "/infer",
        files={"file": ("test.jpg", img_file, "image/jpeg")},
        data={"platform": "twitter"}
    )
    
    assert response.status_code == 200
    data = response.json()
    
    assert "label" in data
    assert "confidence_score" in data
    assert "heatmap_url" in data
    assert data["ocr_text"] == "Fake news"
    assert data["llm_explanation"]["label"] == "suspeito"

