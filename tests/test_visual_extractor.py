import pytest
import torch
import numpy as np
import os
from src.visual_extractor import VisualExtractor, get_visual_embedding, save_embeddings
from PIL import Image

def test_visual_extractor_mobilenet():
    model = VisualExtractor(model_name='mobilenet_v2', pretrained=False)
    dummy_input = torch.randn(2, 3, 224, 224)
    output = model(dummy_input)
    assert output.shape == (2, 1280)

def test_visual_extractor_efficientnet():
    model = VisualExtractor(model_name='efficientnet_b0', pretrained=False)
    dummy_input = torch.randn(2, 3, 224, 224)
    output = model(dummy_input)
    assert output.shape == (2, 1280)

def test_get_visual_embedding():
    # Teste com PIL Image dummy
    dummy_img = Image.new('RGB', (100, 100), color='red')
    embedding = get_visual_embedding(dummy_img)
    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (1280,)
    # Verifica se está normalizado (norma aprox 1)
    norm = np.linalg.norm(embedding)
    assert np.isclose(norm, 1.0, atol=1e-5)

def test_save_embeddings(tmp_path):
    # Setup: criar diretório com imagens dummy
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    
    img1 = Image.new('RGB', (50, 50), color='blue')
    img1.save(img_dir / "img1.jpg")
    
    img2 = Image.new('RGB', (50, 50), color='green')
    img2.save(img_dir / "img2.png")
    
    output_file = tmp_path / "embeddings.npy"
    
    # Executar
    save_embeddings(str(img_dir), str(output_file), model_name='mobilenet_v2')
    
    # Verificar
    assert os.path.exists(output_file)
    data = np.load(output_file, allow_pickle=True).item()
    
    assert len(data['filenames']) == 2
    assert data['embeddings'].shape == (2, 1280)

