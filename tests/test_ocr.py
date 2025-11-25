import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from src.ocr_tesseract import extract_ocr_data, visualize_ocr_heatmap, extract_text

@pytest.fixture
def mock_tesseract_data():
    """
    Retorna estrutura de dados simulando pytesseract.image_to_data.
    """
    return {
        'level': [1, 5, 5, 5],
        'page_num': [1, 1, 1, 1],
        'block_num': [1, 1, 1, 1],
        'par_num': [1, 1, 1, 1],
        'line_num': [1, 1, 1, 1],
        'word_num': [0, 1, 2, 3],
        'left': [0, 10, 50, 100],
        'top': [0, 10, 10, 10],
        'width': [200, 30, 30, 30],
        'height': [50, 20, 20, 20],
        'conf': ['-1', '95.0', '20.0', '60.0'], # High, Low (filtered), Medium
        'text': ['', 'HighConf', 'LowConf', 'MedConf']
    }

@patch('src.ocr_tesseract.pytesseract.image_to_data')
def test_extract_ocr_data(mock_image_to_data, mock_tesseract_data):
    mock_image_to_data.return_value = mock_tesseract_data
    
    # Imagem dummy RGB
    dummy_img = np.zeros((100, 200, 3), dtype=np.uint8)
    
    # Teste com threshold 30 (padrão)
    # LowConf (20.0) deve ser filtrado
    result = extract_ocr_data(dummy_img, confidence_threshold=30)
    
    words = result['words']
    texts = [w['text'] for w in words]
    
    assert 'HighConf' in texts
    assert 'MedConf' in texts
    assert 'LowConf' not in texts
    assert len(words) == 2
    
    # Verificar Stats
    # Confidences: 95.0, 60.0 -> Mean: 77.5
    assert result['stats']['mean_conf'] == 77.5
    
    # Teste Full Text
    assert result['full_text'] == "HighConf MedConf"

@patch('src.ocr_tesseract.pytesseract.image_to_data')
def test_extract_text_legacy(mock_image_to_data, mock_tesseract_data):
    # Testar o wrapper de compatibilidade
    mock_image_to_data.return_value = mock_tesseract_data
    dummy_img = np.zeros((100, 200, 3), dtype=np.uint8)
    
    text = extract_text(dummy_img)
    assert "HighConf" in text

def test_visualize_ocr_heatmap(tmp_path):
    # Criar imagem dummy
    img_path = tmp_path / "test_img.jpg"
    dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
    import cv2
    cv2.imwrite(str(img_path), dummy_img)
    
    words = [
        {'text': 'Test', 'conf': 90.0, 'bbox': (10, 10, 20, 20)}
    ]
    
    output_path = tmp_path / "heatmap.png"
    saved_path = visualize_ocr_heatmap(str(img_path), words, str(output_path))
    
    assert os.path.exists(saved_path)

