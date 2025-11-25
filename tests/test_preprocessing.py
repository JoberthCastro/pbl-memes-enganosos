import pytest
import numpy as np
import cv2
import os
import shutil
from src.preprocessing import (
    normalize_intensity, 
    equalize_histogram, 
    denoise, 
    resize_keep_aspect, 
    pipeline_preprocess
)

@pytest.fixture
def dummy_image():
    # Cria uma imagem aleatória 100x100 RGB
    return np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

def test_normalize_intensity(dummy_image):
    # Cria imagem com range baixo
    low_contrast = (dummy_image // 2).astype(np.uint8)
    norm = normalize_intensity(low_contrast)
    assert norm.max() > low_contrast.max()
    assert norm.min() <= low_contrast.min()
    assert norm.shape == dummy_image.shape

def test_equalize_histogram(dummy_image):
    eq = equalize_histogram(dummy_image)
    assert eq.shape == dummy_image.shape
    assert eq.dtype == np.uint8

def test_denoise(dummy_image):
    # Adiciona ruído
    noise = np.random.normal(0, 25, dummy_image.shape).astype(np.uint8)
    noisy_img = cv2.add(dummy_image, noise)
    
    denoised = denoise(noisy_img)
    assert denoised.shape == dummy_image.shape
    # Difícil testar qualidade visual programaticamente sem métricas complexas,
    # mas verificamos se rodou e retornou imagem válida.
    assert denoised is not None

def test_resize_keep_aspect(dummy_image):
    target_size = (200, 200)
    resized = resize_keep_aspect(dummy_image, target_size)
    assert resized.shape == (200, 200, 3)
    
    # Teste retangular
    rect_img = np.zeros((50, 100, 3), dtype=np.uint8)
    resized_rect = resize_keep_aspect(rect_img, (200, 200))
    assert resized_rect.shape == (200, 200, 3)
    # Verifica padding (bordas pretas)
    # A imagem original 50x100 (1:2) vai virar 100x200 dentro de 200x200
    # Deve ter padding vertical
    # Meio da imagem não deve ser preto (assumindo imagem branca no meio se fosse o caso, mas é zeros)
    # Vamos testar com imagem branca
    white_rect = np.full((50, 100, 3), 255, dtype=np.uint8)
    resized_white = resize_keep_aspect(white_rect, (200, 200))
    
    # Centro deve ser branco
    assert np.all(resized_white[100, 100] == 255)
    # Topo deve ser preto (padding)
    assert np.all(resized_white[10, 100] == 0)

def test_pipeline_preprocess(tmp_path):
    # Cria imagem temporária
    img_path = tmp_path / "test.jpg"
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.imwrite(str(img_path), img)
    
    output_dir = tmp_path / "processed"
    save_path, processed_img = pipeline_preprocess(str(img_path), output_dir=str(output_dir))
    
    assert os.path.exists(save_path)
    assert processed_img.shape == (224, 224, 3)

