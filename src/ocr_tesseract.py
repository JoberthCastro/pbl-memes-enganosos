import pytesseract
from pytesseract import Output
import cv2
import numpy as np
import argparse
import os
from pathlib import Path

# --------------------------------------------------------------------------------------
# Configuração automática do caminho do executável do Tesseract no Windows
# --------------------------------------------------------------------------------------
# Se o Tesseract estiver instalado no caminho padrão (C:\Program Files\Tesseract-OCR),
# este bloco configura o pytesseract para usá-lo automaticamente.
#
# Caso esteja em outro lugar, você pode ajustar a variável `TESSERACT_WINDOWS_PATH`
# abaixo para o caminho correto do seu tesseract.exe.
TESSERACT_WINDOWS_PATH = r"C:\Users\jober\Desktop\tesseract\tesseract.exe"
if os.name == "nt":
    # Define o caminho explicitamente; se estiver diferente na sua máquina,
    # basta ajustar a string acima.
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_WINDOWS_PATH

# Configuração padrão do Tesseract
# --psm 6: Assume um único bloco de texto uniforme.
# --oem 3: Default engine mode.
DEFAULT_CONFIG = r'--psm 6 --oem 3'

def extract_ocr_data(image_input, confidence_threshold=30, config=DEFAULT_CONFIG):
    """
    Extrai texto, bounding boxes e estatísticas de confiança da imagem.
    
    Args:
        image_input (str, Path, or np.ndarray): Caminho da imagem ou array numpy.
        confidence_threshold (int): Confiança mínima (0-100) para considerar uma palavra.
        config (str): String de configuração do Tesseract.
        
    Returns:
        dict: {
            'full_text': str,
            'words': list of dicts [{'text': str, 'conf': float, 'bbox': (x, y, w, h)}],
            'stats': {'mean_conf': float, 'std_conf': float}
        }
    """
    # Tratamento de entrada
    if isinstance(image_input, (str, Path)):
        if not os.path.exists(str(image_input)):
             raise FileNotFoundError(f"Image not found: {image_input}")
        img = cv2.imread(str(image_input))
        if img is None:
            raise ValueError(f"Could not read image at {image_input}")
        # Tesseract funciona melhor com RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif isinstance(image_input, np.ndarray):
        img = image_input
    else:
        raise ValueError("Invalid input type. Expected path or numpy array.")

    # Executar Tesseract
    # output_type=Output.DICT retorna um dicionário com listas para cada campo
    try:
        data = pytesseract.image_to_data(img, config=config, output_type=Output.DICT, lang='eng+por')
    except pytesseract.TesseractNotFoundError:
        print("Tesseract not found. Please install tesseract-ocr.")
        return {'full_text': "", 'words': [], 'stats': {'mean_conf': 0.0, 'std_conf': 0.0}}

    words = []
    confidences = []
    
    n_boxes = len(data['level'])
    
    for i in range(n_boxes):
        text = data['text'][i].strip()
        
        # Ignorar texto vazio
        if not text:
            continue
            
        # Confiança vem como string ou int/float
        try:
            conf = float(data['conf'][i])
        except (ValueError, TypeError):
            conf = -1.0
            
        # Confiança -1 geralmente indica blocos layout sem texto específico reconhecido ainda
        if conf < confidence_threshold:
            continue
            
        (x, y, w, h) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
        
        words.append({
            'text': text,
            'conf': conf,
            'bbox': (x, y, w, h)
        })
        confidences.append(conf)

    # Estatísticas
    if confidences:
        mean_conf = np.mean(confidences)
        std_conf = np.std(confidences)
    else:
        mean_conf = 0.0
        std_conf = 0.0
        
    full_text = " ".join([w['text'] for w in words])
    
    return {
        'full_text': full_text,
        'words': words,
        'stats': {'mean_conf': float(mean_conf), 'std_conf': float(std_conf)}
    }

def visualize_ocr_heatmap(image_input, words, output_path="ocr_heatmap.png"):
    """
    Desenha bounding boxes nas palavras detectadas, coloridas por confiança.
    Verde > 80, Amarelo > 50, Vermelho < 50.
    """
    if isinstance(image_input, (str, Path)):
        img = cv2.imread(str(image_input))
        if img is None:
            raise ValueError("Could not load image for visualization")
    elif isinstance(image_input, np.ndarray):
        # Assume BGR para OpenCV desenhar corretamente. 
        # Se a entrada for RGB (do pipeline anterior), converte para BGR.
        # Como não temos certeza absoluta da origem, assumimos que array numpy precisa ser tratado com cuidado.
        # Uma heurística simples: se vier do `extract_ocr_data` que converteu pra RGB, aqui precisaríamos voltar.
        # Vamos assumir que array input é RGB e converter para BGR para salvar.
        img = cv2.cvtColor(image_input, cv2.COLOR_RGB2BGR)
    else:
        raise ValueError("Invalid input for visualization")
    
    vis_img = img.copy()
    
    for word in words:
        x, y, w, h = word['bbox']
        conf = word['conf']
        
        # Cores BGR
        if conf > 80:
            color = (0, 255, 0)     # Green
        elif conf > 50:
            color = (0, 255, 255)   # Yellow
        else:
            color = (0, 0, 255)     # Red
            
        cv2.rectangle(vis_img, (x, y), (x + w, y + h), color, 2)
        # Texto pequeno com valor da confiança
        label = f"{int(conf)}"
        cv2.putText(vis_img, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    cv2.imwrite(output_path, vis_img)
    return output_path

# Função alias para manter compatibilidade com código antigo (se houver chamadas diretas a extract_text)
def extract_text(image_path: str) -> str:
    result = extract_ocr_data(image_path)
    return result['full_text']

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OCR Analysis Tool for Memes")
    parser.add_argument("image_path", help="Path to input image")
    parser.add_argument("--output", default="ocr_heatmap.png", help="Output path for visualization")
    parser.add_argument("--thresh", type=int, default=30, help="Confidence threshold (0-100)")
    
    args = parser.parse_args()
    
    try:
        print(f"Running OCR on {args.image_path}...")
        result = extract_ocr_data(args.image_path, confidence_threshold=args.thresh)
        
        print("\n--- Text Extracted ---")
        print(result['full_text'])
        
        print("\n--- Statistics ---")
        print(f"Word Count: {len(result['words'])}")
        print(f"Mean Confidence: {result['stats']['mean_conf']:.2f}")
        print(f"Std Deviation: {result['stats']['std_conf']:.2f}")
        
        out = visualize_ocr_heatmap(args.image_path, result['words'], args.output)
        print(f"\nVisualization saved to: {out}")
        
    except Exception as e:
        print(f"Error: {e}")
