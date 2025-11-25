import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import MobileNet_V2_Weights, EfficientNet_B0_Weights
from PIL import Image
import numpy as np
import os
from tqdm import tqdm

# Import local sem prefixo de pacote para funcionar com `python src/train.py`
from preprocessing import get_transforms

class VisualExtractor(nn.Module):
    def __init__(self, model_name='mobilenet_v2', pretrained=True):
        """
        Extrai features visuais usando MobileNetV2 ou EfficientNetB0.
        
        Args:
            model_name: 'mobilenet_v2' ou 'efficientnet_b0'
            pretrained: Se deve usar pesos do ImageNet
        """
        super(VisualExtractor, self).__init__()
        
        self.model_name = model_name
        
        if model_name == 'mobilenet_v2':
            weights = MobileNet_V2_Weights.DEFAULT if pretrained else None
            self.backbone = models.mobilenet_v2(weights=weights)
            # features output: [B, 1280, 7, 7]
            self.features = self.backbone.features
            self.output_dim = 1280
            
        elif model_name == 'efficientnet_b0':
            weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
            self.backbone = models.efficientnet_b0(weights=weights)
            # features output: [B, 1280, 7, 7]
            self.features = self.backbone.features
            self.output_dim = 1280
            
        else:
            raise ValueError(f"Model {model_name} not supported")

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        """
        x: Tensor de imagens [Batch, 3, 224, 224]
        """
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x

def get_visual_embedding(img_input, model=None, device=None):
    """
    Retorna o vetor de embedding normalizado para uma única imagem.
    
    Args:
        img_input: Pode ser caminho (str), PIL Image ou Tensor
        model: Instância de VisualExtractor carregada (opcional)
        device: 'cpu' ou 'cuda'
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    if model is None:
        model = VisualExtractor()
        model.to(device)
        model.eval()
        
    transform = get_transforms(mode='eval')
    
    # Tratar input
    if isinstance(img_input, str):
        img = Image.open(img_input).convert('RGB')
        img_tensor = transform(img).unsqueeze(0)
    elif isinstance(img_input, Image.Image):
        img_tensor = transform(img_input).unsqueeze(0)
    elif isinstance(img_input, torch.Tensor):
        img_tensor = img_input
        if img_tensor.dim() == 3:
            img_tensor = img_tensor.unsqueeze(0)
    else:
        raise ValueError("Invalid input type for get_visual_embedding")
        
    img_tensor = img_tensor.to(device)
    
    with torch.no_grad():
        features = model(img_tensor)
        # Normalização L2
        features = torch.nn.functional.normalize(features, p=2, dim=1)
        
    return features.cpu().numpy().flatten()

def save_embeddings(dataset_dir, out_file='embeddings.npy', model_name='mobilenet_v2'):
    """
    Processa um diretório de imagens e salva os embeddings e filenames.
    
    Args:
        dataset_dir: Diretório contendo imagens (recursivo)
        out_file: Caminho para salvar o .npy
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = VisualExtractor(model_name=model_name)
    model.to(device)
    model.eval()
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    files = []
    for root, _, filenames in os.walk(dataset_dir):
        for f in filenames:
            if os.path.splitext(f)[1].lower() in image_extensions:
                files.append(os.path.join(root, f))
                
    embeddings = []
    processed_files = []
    
    print(f"Processing {len(files)} images...")
    
    for fpath in tqdm(files):
        try:
            emb = get_visual_embedding(fpath, model=model, device=device)
            embeddings.append(emb)
            processed_files.append(fpath)
        except Exception as e:
            print(f"Error processing {fpath}: {e}")
            
    embeddings = np.array(embeddings)
    
    # Salvar embeddings e filenames
    np.save(out_file, {'embeddings': embeddings, 'filenames': processed_files})
    print(f"Saved embeddings shape {embeddings.shape} to {out_file}")

if __name__ == "__main__":
    # Teste simples
    model = VisualExtractor(model_name='mobilenet_v2')
    dummy = torch.randn(1, 3, 224, 224)
    out = model(dummy)
    print(f"MobileNetV2 Output: {out.shape}") # [1, 1280]
    
    model_eff = VisualExtractor(model_name='efficientnet_b0')
    out_eff = model_eff(dummy)
    print(f"EfficientNetB0 Output: {out_eff.shape}") # [1, 1280]
