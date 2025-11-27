import os
import cv2
import numpy as np
import torchvision.transforms as transforms
from pathlib import Path

# --- Updated Data Augmentation ---
def get_transforms(mode='train'):
    if mode == 'train':
        return transforms.Compose([
            transforms.Resize((224, 224)),
            # Removed HorizontalFlip because it mirrors text, making it unreadable/unnatural for OCR-like tasks
            # transforms.RandomHorizontalFlip(), 
            
            # Geometric distortions (simulate bad screenshots/photos)
            transforms.RandomRotation(degrees=5),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
            
            # Visual distortions (simulate compression/lighting)
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            transforms.RandomGrayscale(p=0.1),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            
            # Add noise (optional, custom lambda if needed, but GaussianBlur helps)
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

# --- Existing Helpers Preserved ---

def normalize_intensity(img):
    """
    Normaliza a intensidade dos pixels usando Min-Max scaling para 0-255.
    Melhora o contraste linearmente.
    """
    return cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

def equalize_histogram(img):
    """
    Aplica equalização de histograma.
    Se colorida, converte para YUV e equaliza o canal Y (luminância).
    """
    if len(img.shape) == 2: # Grayscale
        return cv2.equalizeHist(img)
    
    # Convert to YUV
    img_yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    # Equalize the histogram of the Y channel
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
    # Convert the YUV image back to RGB format
    return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)

def denoise(img):
    """
    Remove ruído usando Fast Non-Local Means Denoising.
    """
    # h: parameter deciding filter strength. Higher h means better denoising but also more detail removal
    return cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

def resize_keep_aspect(img, target_size):
    """
    Redimensiona a imagem mantendo a proporção e aplicando padding (letterbox).
    target_size: (width, height)
    """
    h, w = img.shape[:2]
    target_w, target_h = target_size
    
    # Calcula scale
    scale = min(target_w/w, target_h/h)
    new_w, new_h = int(w*scale), int(h*scale)
    
    # Resize
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Canvas preta
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    
    # Centralizar
    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2
    
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    return canvas

def pipeline_preprocess(path_or_array, output_dir="data/processed", output_name=None):
    """
    Pipeline completo: Leitura -> RGB -> Denoise -> Equalização -> Normalização -> Resize -> Salvar.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    filename = output_name if output_name else "processed_image.jpg"
    
    # 1. Leitura e Conversão
    if isinstance(path_or_array, (str, Path)):
        img = cv2.imread(str(path_or_array))
        if img is None:
            raise FileNotFoundError(f"Could not read image: {path_or_array}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if not output_name:
            filename = os.path.basename(path_or_array)
    elif isinstance(path_or_array, np.ndarray):
        img = path_or_array
        # Se for BGR (comum no OpenCV), assumimos que quem passa array já tratou ou sabe.
        # Mas por segurança, se vier do cv2.imread externo, é BGR.
        # Aqui assumimos entrada RGB se for array, ou responsabilidade do caller.
        # Vamos assumir RGB para consistência interna.
    else:
        raise ValueError("Input must be file path or numpy array")
        
    # 2. Denoise
    img = denoise(img)
    
    # 3. Equalização
    img = equalize_histogram(img)
    
    # 4. Normalização
    img = normalize_intensity(img)
    
    # 5. Resize (Padronização para o modelo)
    img = resize_keep_aspect(img, (224, 224))
    
    # 6. Gravação
    save_path = os.path.join(output_dir, filename)
    # Converter para BGR para salvar com OpenCV
    cv2.imwrite(save_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    
    return save_path, img
