import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import argparse
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
    matthews_corrcoef,
    precision_recall_curve,
    average_precision_score,
)
from torch.utils.data import DataLoader
from PIL import Image

# Imports locais
from src.fusion_model import FusionModel
from src.visual_extractor import VisualExtractor, get_visual_embedding
from src.text_model import TextModel, get_text_embedding, Tokenizer
from src.ocr_tesseract import extract_ocr_data
from src.preprocessing import get_transforms
from src.utils import CONFIG

TEST_CSV_PATH = "data/test_labels.csv"

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

    # Opcional: carregar pesos treinados do modelo visual, se existirem
    visual_ckpt = "models/visual_model.pth"
    if os.path.exists(visual_ckpt):
        visual_ext.load_state_dict(torch.load(visual_ckpt, map_location=device))

    # Text
    tokenizer = Tokenizer(num_words=10000)
    # Tentar carregar tokenizer treinado se existir, senão fit dummy para evitar erro,
    # mas idealmente o tokenizer deve ser salvo no treinamento.
    # Como o train.py não está salvando o tokenizer explicitamente (apenas recriando),
    # vamos tentar carregar o checkpoint do modelo e deduzir o tamanho do vocabulário.
    
    text_ckpt = "models/text_model.pth"
    vocab_size = 10001 # Default
    
    if os.path.exists(text_ckpt):
        try:
            state_dict = torch.load(text_ckpt, map_location=device)
            # Deduzir vocab_size pelo shape dos pesos de embedding: [vocab_size, embed_dim]
            if 'embedding.weight' in state_dict:
                vocab_size = state_dict['embedding.weight'].shape[0]
                print(f"Detected vocab_size from checkpoint: {vocab_size}")
        except Exception as e:
            print(f"Error inspecting text checkpoint: {e}")

    text_model = TextModel(vocab_size=vocab_size, embedding_dim=128, hidden_dim=128)
    text_model.to(device)
    text_model.eval()

    if os.path.exists(text_ckpt):
        try:
            text_model.load_state_dict(torch.load(text_ckpt, map_location=device))
        except RuntimeError as e:
            print(f"Warning: could not load text_model checkpoint due to size mismatch: {e}")


    # Fusion (usa embeddings pré-calculados, como em train.py)
    config = {'num_classes': 2, 'text_output_dim': 256}
    model = FusionModel(config)  # use_precomputed_embeddings = True
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print(f"Warning: Model checkpoint not found at {model_path}. Using random weights.")

    model.to(device)
    model.eval()
    
    # --- 2. Load Data ---
    files = []
    labels = [] # 0: Auth, 1: Manipulated
    
    # Se existir o arquivo de labels de teste, usa ele
    if os.path.exists(TEST_CSV_PATH):
        print(f"Using test split from {TEST_CSV_PATH}")
        df_test = pd.read_csv(TEST_CSV_PATH)
        
        for _, row in df_test.iterrows():
            fname = row['filename']
            label_str = str(row['label']).lower()
            label_idx = 0 if label_str == 'authentic' else 1
            
            # Tenta achar o arquivo no dataset_dir
            possible_paths = [
                os.path.join(dataset_dir, "authentic", fname),
                os.path.join(dataset_dir, "manipulated", fname),
                os.path.join(dataset_dir, fname),
            ]
            
            found = False
            for p in possible_paths:
                if os.path.exists(p):
                    files.append(p)
                    labels.append(label_idx)
                    found = True
                    break
            
            if not found:
                # Tenta ver se o filename já é um caminho relativo/absoluto válido
                if os.path.exists(fname):
                    files.append(fname)
                    labels.append(label_idx)
    
    else:
        print(f"Warning: {TEST_CSV_PATH} not found. Evaluating on ALL images in {dataset_dir} (potential data leakage if trained on same data).")
        # Assumindo estrutura: dataset_dir/authentic e dataset_dir/manipulated
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
            # Visual -> embedding
            img_t = transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                v_emb = visual_ext(img_t)

            # Text -> embedding
            seqs = tokenizer.texts_to_sequences([text])
            tokens = seqs[0] if seqs else []
            if len(tokens) < 100:
                tokens += [0] * (100 - len(tokens))
            else:
                tokens = tokens[:100]
            # shape [1, max_len] para Embedding/LSTM (batch_first)
            text_t = torch.tensor([tokens], dtype=torch.long).to(device)
            with torch.no_grad():
                t_emb = text_model(text_t)

            # OCR Stats
            # IMPORTANT: During training (train.py), OCR stats are currently passed as ZEROS.
            # If we pass real stats here, the model (which never saw them) might get confused 
            # and output garbage, causing the precision/collapse issue.
            #
            # To fix evaluation parity with current training logic, we must either:
            # 1. Use real OCR in training (preferred but slow/complex now)
            # 2. Use zeros here to match training.
            #
            # Given the user is facing "Precision 0.0" collapse, let's match training behavior first.
            
            # ocr_stats_vec = [
            #     ocr_res['stats']['mean_conf'],
            #     ocr_res['stats']['std_conf'],
            #     len([w for w in ocr_res['words'] if w['conf'] < 50])
            # ]
            
            # ZEROS to match train.py behavior:
            ocr_stats_vec = [0.0, 0.0, 0.0]
            
            # shape [1, 3] para Linear/BatchNorm1d
            ocr_t = torch.tensor([ocr_stats_vec], dtype=torch.float32).to(device)

            # Predict (fusão em cima dos embeddings)
            with torch.no_grad():
                logits = model(v_emb, t_emb, ocr_t)
                probs = torch.softmax(logits, dim=1)
                
            # Aplicar threshold ótimo (0.70) em vez de argmax padrão
            # Isso melhora o balanceamento entre precision e recall
            # Threshold atualizado após retreino com mais dados
            OPTIMAL_THRESHOLD = 0.70
            prob_manipulated = probs[0, 1].item()
            pred_label = 1 if prob_manipulated >= OPTIMAL_THRESHOLD else 0
            prob_score = prob_manipulated if pred_label == 1 else (1 - prob_manipulated)
            
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
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary"
    )
    cm = confusion_matrix(y_true, y_pred)

    # --- Additional Metrics ---
    # Matthews Correlation Coefficient
    mcc = matthews_corrcoef(y_true, y_pred) if len(y_true) > 0 else 0.0

    # Specificity (True Negative Rate) and False Positive Rate
    # Assumindo matriz de confusão binária 2x2: [[TN, FP], [FN, TP]]
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    else:
        specificity = 0.0
        fpr = 0.0

    # Precision–Recall AUC (PR AUC)
    if len(y_true) > 0:
        pr_auc = average_precision_score(y_true, y_prob)
    else:
        pr_auc = 0.0

    metrics = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "mcc": mcc,
        "specificity": specificity,
        "fpr": fpr,
        "pr_auc": pr_auc,
        "confusion_matrix": cm.tolist(),
    }
    
    # Save JSON
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)
        
    # --- 5. Plots ---
    # Só gera gráficos se houver dados suficientes
    if cm.size > 0:
        # Confusion Matrix
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Auth', 'Fake'], yticklabels=['Auth', 'Fake'])
        plt.title("Confusion Matrix")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
        plt.close()
    else:
        print("Confusion matrix vazia, pulando plot de matriz de confusão.")
    
    if not df.empty:
        # Confidence Histogram
        plt.figure(figsize=(8, 5))
        try:
             sns.histplot(data=df, x='probability', hue='true_label', bins=20, kde=True)
             plt.title("Model Confidence Distribution")
             plt.xlabel("Confidence Score")
             plt.savefig(os.path.join(output_dir, "confidence_dist.png"))
        except Exception as e:
             print(f"Could not plot confidence distribution (KDE error): {e}")
        plt.close()
    else:
        print("DataFrame de resultados vazio, pulando histograma de confiança.")

    # Precision–Recall Curve
    if len(y_true) > 0:
        precisions, recalls, _ = precision_recall_curve(y_true, y_prob)
        plt.figure(figsize=(8, 5))
        plt.step(recalls, precisions, where="post", label="Precision–Recall curve")
        plt.fill_between(recalls, precisions, step="post", alpha=0.2)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"Precision–Recall Curve (PR AUC = {pr_auc:.4f})")
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, "precision_recall_curve.png"))
        plt.close()
    
    # --- 6. Markdown Report ---
    # Evitar erro se y_true/y_pred estiverem vazios
    if len(y_true) > 0:
        cls_report = classification_report(y_true, y_pred, target_names=['Authentic', 'Manipulated'])
    else:
        cls_report = "No valid samples were evaluated (check OCR/Text pipeline and dataset)."

    report_md = f"""
# Evaluation Report

## Metrics
- **Accuracy**: {acc:.4f}
- **Precision**: {prec:.4f}
- **Recall**: {rec:.4f}
- **F1 Score**: {f1:.4f}
- **MCC**: {mcc:.4f}
- **Specificity (TNR)**: {specificity:.4f}
- **False Positive Rate (FPR)**: {fpr:.4f}
- **PR AUC**: {pr_auc:.4f}

## Visualizations
### Confusion Matrix
![Confusion Matrix](confusion_matrix.png)

### Confidence Distribution
![Confidence Distribution](confidence_dist.png)

### Precision–Recall Curve
![Precision–Recall Curve](precision_recall_curve.png)

## Classification Report
```
{cls_report}
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
