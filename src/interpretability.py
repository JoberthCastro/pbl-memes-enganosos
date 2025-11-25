import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

class GradCAM:
    def __init__(self, model, target_layer):
        """
        Implementação do Grad-CAM para PyTorch.
        Args:
            model: Modelo PyTorch (CNN).
            target_layer: A camada convolucional alvo para visualização.
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Hooks para capturar gradientes e ativações
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        # grad_output é uma tupla, pegamos o primeiro elemento
        self.gradients = grad_output[0]

    def __call__(self, x, class_idx=None):
        """
        Gera o heatmap do Grad-CAM.
        Args:
            x: Tensor de entrada [1, C, H, W]
            class_idx: Índice da classe alvo. Se None, usa a predição máxima.
        """
        # Forward pass
        self.model.zero_grad()
        output = self.model(x)
        
        if class_idx is None:
            class_idx = torch.argmax(output, dim=1)
            
        # Backward pass
        score = output[0, class_idx]
        score.backward()
        
        # Global Average Pooling dos gradientes
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        
        # Ponderar ativações pelos gradientes
        activations = self.activations[0]
        for i in range(activations.shape[0]):
            activations[i, :, :] *= pooled_gradients[i]
            
        # Média dos canais ponderados -> Heatmap 2D
        heatmap = torch.mean(activations, dim=0).cpu().detach()
        
        # ReLU no heatmap (apenas features positivas interessam)
        heatmap = F.relu(heatmap)
        
        # Normalizar
        if torch.max(heatmap) != 0:
            heatmap /= torch.max(heatmap)
            
        return heatmap.numpy()

def overlay_heatmap(img, heatmap, alpha=0.4, colormap=cv2.COLORMAP_JET):
    """
    Sobrepõe o heatmap na imagem original.
    Args:
        img: Imagem original (PIL ou Numpy RGB)
        heatmap: Heatmap 2D (numpy float)
    """
    if isinstance(img, Image.Image):
        img = np.array(img)
        
    # Resize heatmap para tamanho da imagem
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    
    # Converter para RGB (0-255)
    heatmap = np.uint8(255 * heatmap)
    
    # Aplicar colormap
    heatmap = cv2.applyColorMap(heatmap, colormap)
    
    # Sobrepor
    # img deve estar em BGR se for usado com cv2, ou RGB se PIL->np
    # Assumindo RGB (padrão PIL) -> converter heatmap BGR->RGB para display correto
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    overlay = cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)
    
    return overlay

def gather_ocr_evidence(ocr_words, low_conf_threshold=50):
    """
    Analisa palavras do OCR e retorna evidências de problemas (baixa confiança).
    Args:
        ocr_words: Lista de dicts [{'text', 'conf', 'bbox'}]
        low_conf_threshold: Limiar para considerar palavra suspeita/mal detectada.
    Returns:
        dict: {
            'low_confidence_words': list,
            'count': int,
            'ratio': float
        }
    """
    low_conf_list = []
    total_words = len(ocr_words)
    
    if total_words == 0:
        return {'low_confidence_words': [], 'count': 0, 'ratio': 0.0}
        
    for w in ocr_words:
        if w['conf'] < low_conf_threshold:
            low_conf_list.append({
                'text': w['text'],
                'conf': float(w['conf']),
                'bbox': w['bbox']
            })
            
    return {
        'low_confidence_words': low_conf_list,
        'count': len(low_conf_list),
        'ratio': len(low_conf_list) / total_words
    }

