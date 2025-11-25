import pytest
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from src.interpretability import GradCAM, overlay_heatmap, gather_ocr_evidence

# --- Mock Model for GradCAM ---
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Linear(16 * 16 * 16, 2) # Assuming input 32x32

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def test_grad_cam():
    model = SimpleCNN()
    model.eval()
    
    # Target layer: last conv layer in features sequential
    target_layer = model.features[0] 
    
    grad_cam = GradCAM(model, target_layer)
    
    # Dummy input [1, 3, 32, 32]
    x = torch.randn(1, 3, 32, 32, requires_grad=True)
    
    heatmap = grad_cam(x, class_idx=0)
    
    assert heatmap is not None
    # Shape should match pooled feature map dimensions (32x32 input -> 32x32 output for conv w/ padding)
    # But we access output BEFORE pooling in hook if we hooked the conv layer specifically.
    # Wait, target_layer is Conv2d.
    # Input 32x32 -> Conv(3,16) -> 32x32 (padding=1).
    # So heatmap should be 32x32 (spatial dims of activation).
    assert heatmap.shape == (32, 32)
    assert np.max(heatmap) <= 1.0 + 1e-5
    assert np.min(heatmap) >= 0.0

def test_gather_ocr_evidence():
    words = [
        {'text': 'Good', 'conf': 90, 'bbox': (0,0,10,10)},
        {'text': 'Bad', 'conf': 20, 'bbox': (10,10,10,10)},
        {'text': 'Avg', 'conf': 60, 'bbox': (20,20,10,10)}
    ]
    
    evidence = gather_ocr_evidence(words, low_conf_threshold=50)
    
    assert evidence['count'] == 1
    assert evidence['low_confidence_words'][0]['text'] == 'Bad'
    assert evidence['ratio'] == 1/3

def test_overlay_heatmap():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    heatmap = np.random.rand(10, 10).astype(np.float32)
    
    overlay = overlay_heatmap(img, heatmap)
    
    assert overlay.shape == (100, 100, 3)

