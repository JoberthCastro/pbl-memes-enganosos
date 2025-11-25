import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import argparse
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from torch.utils.data import DataLoader
from PIL import Image

# Imports locais
from src.fusion_model import FusionModel
from src.visual_extractor import VisualExtractor, get_visual_embedding
from src.text_model import TextModel, get_text_embedding, Tokenizer
from src.ocr_tesseract import extract_ocr_data
from src.preprocessing import get_transforms
from src.utils import CONFIG

def evaluate(dataset_dir, model_path, output_dir='reports', device='cpu'):
    """
    Avalia o modelo de fusão em um dataset e gera relatórios detalhados.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # --- 1. Setup Models ---
    print("Loading models...")
    # Visual
    visual_ext = VisualExtractor(model_name='mobilenet_v2')
    visual_ext.to(device)
    visual_ext.eval()
    
    # Text
    tokenizer = Tokenizer(num_words=10000) 
    # Em prod, carregar tokenizer treinado: Tokenizer.load('data/tokenizer.pkl')
    
    text_model = TextModel(vocab_size=10001, embedding_dim=128, hidden_dim=128)
    text_model.to(device)
    text_model.eval()
    
    # Fusion
    config = {'num_classes': 2, 'text_output_dim': 256}
    model = FusionModel(config, visual_extractor=visual_ext, text_model=text_model)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print(f"Warning: Model checkpoint not found at {model_path}. Using random weights.")
    
    model.to(device)
    model.eval()
    
    # --- 2. Load Data ---
    # Assumindo estrutura: dataset_dir/authentic e dataset_dir/manipulated
    files = []
    labels = [] # 0: Auth, 1: Manipulated
    
    for label_name, label_idx in [('authentic', 0), ('manipulated', 1)]:
        path = os.path.join(dataset_dir, label_name)
        if os.path.exists(path):
            for f in os.listdir(path):
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    files.append(os.path.join(path, f))
                    labels.append(label_idx)
    
    if not files:
        print("No images found for evaluation.")
        return

    print(f"Evaluating on {len(files)} images...")
    
    results = []
    y_true = []
    y_pred = []
    y_prob = []
    
    # --- 3. Inference Loop ---
    transform = get_transforms(mode='eval')
    
    for fpath, true_label in tqdm(zip(files, labels), total=len(files)):
        try:
            # Load Image
            img = Image.open(fpath).convert("RGB")
            np_img = np.array(img)
            
            # OCR
            ocr_res = extract_ocr_data(np_img)
            text = ocr_res['full_text']
            ocr_mean_conf = ocr_res['stats']['mean_conf']
            
            # Features
            # Visual
            img_t = transform(img).unsqueeze(0).to(device)
            
            # Text
            seqs = tokenizer.texts_to_sequences([text])
            tokens = seqs[0] if seqs else []
            if len(tokens) < 100: tokens += [0]*(100-len(tokens))
            else: tokens = tokens[:100]
            text_t = torch.tensor([tokens], dtype=torch.long).to(device)
            
            # OCR Stats
            ocr_stats_vec = [
                ocr_res['stats']['mean_conf'], 
                ocr_res['stats']['std_conf'], 
                len([w for w in ocr_res['words'] if w['conf'] < 50])
            ]
            ocr_t = torch.tensor([ocr_stats_vec], dtype=torch.float32).to(device)
            
            # Predict
            with torch.no_grad():
                logits = model(img_t, text_t, ocr_t)
                probs = torch.softmax(logits, dim=1)
                score, pred_idx = torch.max(probs, 1)
                
            pred_label = pred_idx.item()
            prob_score = score.item()
            
            # Store
            y_true.append(true_label)
            y_pred.append(pred_label)
            y_prob.append(prob_score if pred_label == 1 else 1 - prob_score) # Prob of class 1
            
            results.append({
                "filename": os.path.basename(fpath),
                "true_label": true_label,
                "pred_label": pred_label,
                "probability": round(prob_score, 4),
                "ocr_mean_conf": round(ocr_mean_conf, 2),
                "text_snippet": text[:30]
            })
            
        except Exception as e:
            print(f"Error processing {fpath}: {e}")

    # --- 4. Metrics & Reports ---
    
    # Save CSV
    df = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, "evaluation_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"Detailed results saved to {csv_path}")
    
    # Metrics
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    cm = confusion_matrix(y_true, y_pred)
    
    metrics = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "confusion_matrix": cm.tolist()
    }
    
    # Save JSON
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)
        
    # --- 5. Plots ---
    
    # Confusion Matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Auth', 'Fake'], yticklabels=['Auth', 'Fake'])
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
    plt.close()
    
    # Confidence Histogram
    plt.figure(figsize=(8, 5))
    sns.histplot(data=df, x='probability', hue='true_label', bins=20, kde=True)
    plt.title("Model Confidence Distribution")
    plt.xlabel("Confidence Score")
    plt.savefig(os.path.join(output_dir, "confidence_dist.png"))
    plt.close()
    
    # --- 6. Markdown Report ---
    report_md = f"""
# Evaluation Report

## Metrics
- **Accuracy**: {acc:.4f}
- **Precision**: {prec:.4f}
- **Recall**: {rec:.4f}
- **F1 Score**: {f1:.4f}

## Visualizations
### Confusion Matrix
![Confusion Matrix](confusion_matrix.png)

### Confidence Distribution
![Confidence Distribution](confidence_dist.png)

## Classification Report
```
{classification_report(y_true, y_pred, target_names=['Authentic', 'Manipulated'])}
```
    """
    
    with open(os.path.join(output_dir, "report.md"), "w") as f:
        f.write(report_md)
        
    print("Evaluation complete. Check 'reports/' folder.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/raw", help="Dataset directory (containing authentic/manipulated folders)")
    parser.add_argument("--model", default="models/fusion_model.pth", help="Path to saved model")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    evaluate(args.data, args.model, device=device)
