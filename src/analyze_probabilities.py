"""
Análise da distribuição de probabilidades do modelo.
Ajuda a entender por que o threshold padrão não funciona.
"""
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, roc_curve, auc
import os

from src.fusion_model import FusionModel
from src.visual_extractor import VisualExtractor, get_visual_embedding
from src.text_model import TextModel, get_text_embedding, Tokenizer
from src.ocr_tesseract import extract_ocr_data
from src.preprocessing import get_transforms
from PIL import Image

TEST_CSV_PATH = "data/test_labels.csv"
DATA_DIR = "data/raw"
MODEL_DIR = "models"

def analyze_probability_distribution(dataset_dir, model_path, device='cpu', output_dir='reports'):
    """
    Analisa a distribuição de probabilidades e encontra threshold ótimo.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("🔍 Carregando modelos...")
    # Carregar modelos (mesmo código do evaluate.py)
    visual_ext = VisualExtractor(model_name='mobilenet_v2')
    visual_ext.to(device)
    visual_ext.eval()
    
    visual_ckpt = os.path.join(MODEL_DIR, "visual_model.pth")
    if os.path.exists(visual_ckpt):
        visual_ext.load_state_dict(torch.load(visual_ckpt, map_location=device))
    
    tokenizer = Tokenizer(num_words=10000)
    text_model = TextModel(vocab_size=10001, embedding_dim=128, hidden_dim=128)
    text_model.to(device)
    text_model.eval()
    
    text_ckpt = os.path.join(MODEL_DIR, "text_model.pth")
    if os.path.exists(text_ckpt):
        try:
            text_model.load_state_dict(torch.load(text_ckpt, map_location=device))
        except RuntimeError:
            pass
    
    config = {'num_classes': 2, 'text_output_dim': 256}
    fusion_model = FusionModel(config)
    if os.path.exists(model_path):
        fusion_model.load_state_dict(torch.load(model_path, map_location=device))
    fusion_model.to(device)
    fusion_model.eval()
    
    # Carregar dados
    if not os.path.exists(TEST_CSV_PATH):
        print(f"❌ {TEST_CSV_PATH} não encontrado!")
        return
    
    test_df = pd.read_csv(TEST_CSV_PATH)
    transform = get_transforms(mode='eval')
    
    y_true = []
    y_scores = []
    results = []
    
    print("📊 Processando amostras...")
    for _, row in test_df.iterrows():
        fname = row['filename']
        label_str = str(row['label']).lower()
        label_idx = 0 if label_str == 'authentic' else 1
        
        # Encontrar arquivo
        possible_paths = [
            os.path.join(dataset_dir, "authentic", fname),
            os.path.join(dataset_dir, "manipulated", fname),
            os.path.join(dataset_dir, fname),
        ]
        
        fpath = None
        for p in possible_paths:
            if os.path.exists(p):
                fpath = p
                break
        
        if not fpath:
            continue
        
        try:
            img = Image.open(fpath).convert("RGB")
            np_img = np.array(img)
            
            # OCR
            ocr_res = extract_ocr_data(np_img)
            text = ocr_res['full_text']
            ocr_stats_vec = [0.0, 0.0, 0.0]
            
            # Features
            img_t = transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                v_emb = visual_ext(img_t)
            
            seqs = tokenizer.texts_to_sequences([text])
            tokens = seqs[0] if seqs else []
            if len(tokens) < 100:
                tokens += [0] * (100 - len(tokens))
            else:
                tokens = tokens[:100]
            text_t = torch.tensor([tokens], dtype=torch.long).to(device)
            with torch.no_grad():
                t_emb = text_model(text_t)
            
            ocr_t = torch.tensor([ocr_stats_vec], dtype=torch.float32).to(device)
            
            # Predição
            with torch.no_grad():
                logits = fusion_model(v_emb, t_emb, ocr_t)
                probs = torch.softmax(logits, dim=1)
                prob_manipulated = probs[0, 1].item()
                prob_authentic = probs[0, 0].item()
            
            y_true.append(label_idx)
            y_scores.append(prob_manipulated)
            results.append({
                'filename': fname,
                'true_label': 'authentic' if label_idx == 0 else 'manipulated',
                'prob_authentic': prob_authentic,
                'prob_manipulated': prob_manipulated,
            })
            
        except Exception as e:
            print(f"Erro processando {fpath}: {e}")
            continue
    
    if len(y_true) == 0:
        print("❌ Nenhuma amostra processada!")
        return
    
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    
    print(f"\n📈 Estatísticas das Probabilidades:")
    print(f"   Média: {y_scores.mean():.4f}")
    print(f"   Mediana: {np.median(y_scores):.4f}")
    print(f"   Desvio Padrão: {y_scores.std():.4f}")
    print(f"   Min: {y_scores.min():.4f}")
    print(f"   Max: {y_scores.max():.4f}")
    print(f"   Q1 (25%): {np.percentile(y_scores, 25):.4f}")
    print(f"   Q3 (75%): {np.percentile(y_scores, 75):.4f}")
    
    # Separar por classe
    authentic_scores = y_scores[y_true == 0]
    manipulated_scores = y_scores[y_true == 1]
    
    print(f"\n📊 Por Classe:")
    print(f"   Authentic - Média: {authentic_scores.mean():.4f}, Mediana: {np.median(authentic_scores):.4f}")
    print(f"   Manipulated - Média: {manipulated_scores.mean():.4f}, Mediana: {np.median(manipulated_scores):.4f}")
    
    # Plotar distribuição
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(authentic_scores, bins=20, alpha=0.7, label='Authentic', color='green', edgecolor='black')
    plt.hist(manipulated_scores, bins=20, alpha=0.7, label='Manipulated', color='red', edgecolor='black')
    plt.axvline(0.5, color='black', linestyle='--', label='Threshold padrão (0.5)')
    plt.xlabel('Probabilidade de ser Manipulated')
    plt.ylabel('Frequência')
    plt.title('Distribuição de Probabilidades')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.boxplot([authentic_scores, manipulated_scores], labels=['Authentic', 'Manipulated'])
    plt.ylabel('Probabilidade de ser Manipulated')
    plt.title('Boxplot por Classe')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'probability_distribution.png'), dpi=150)
    print(f"\n💾 Gráfico salvo em {output_dir}/probability_distribution.png")
    
    # Precision-Recall Curve
    precision, recall, thresholds_pr = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(recall, precision)
    
    # Encontrar threshold ótimo (F1 máximo)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds_pr[optimal_idx] if optimal_idx < len(thresholds_pr) else 0.5
    
    print(f"\n🎯 Threshold Ótimo (F1 máximo):")
    print(f"   Threshold: {optimal_threshold:.4f}")
    print(f"   Precision: {precision[optimal_idx]:.4f}")
    print(f"   Recall: {recall[optimal_idx]:.4f}")
    print(f"   F1-Score: {f1_scores[optimal_idx]:.4f}")
    print(f"   PR AUC: {pr_auc:.4f}")
    
    # Plotar PR Curve
    plt.figure(figsize=(10, 6))
    plt.plot(recall, precision, label=f'PR Curve (AUC = {pr_auc:.4f})', linewidth=2)
    plt.scatter(recall[optimal_idx], precision[optimal_idx], 
                color='red', s=100, zorder=5, 
                label=f'Ótimo (F1={f1_scores[optimal_idx]:.4f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'pr_curve_analysis.png'), dpi=150)
    print(f"💾 PR Curve salva em {output_dir}/pr_curve_analysis.png")
    
    # Testar diferentes thresholds
    print(f"\n📊 Testando Thresholds:")
    test_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    best_f1 = 0
    best_thresh = 0.5
    
    for thresh in test_thresholds:
        y_pred = (y_scores >= thresh).astype(int)
        tp = ((y_true == 1) & (y_pred == 1)).sum()
        fp = ((y_true == 0) & (y_pred == 1)).sum()
        fn = ((y_true == 1) & (y_pred == 0)).sum()
        tn = ((y_true == 0) & (y_pred == 0)).sum()
        
        precision_t = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall_t = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_t = 2 * (precision_t * recall_t) / (precision_t + recall_t + 1e-10)
        acc_t = (tp + tn) / len(y_true)
        
        if f1_t > best_f1:
            best_f1 = f1_t
            best_thresh = thresh
        
        print(f"   Threshold {thresh:.1f}: P={precision_t:.3f}, R={recall_t:.3f}, F1={f1_t:.3f}, Acc={acc_t:.3f}")
    
    print(f"\n🏆 Melhor Threshold: {best_thresh:.2f} (F1={best_f1:.4f})")
    
    # Salvar resultados
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(output_dir, 'probability_analysis.csv'), index=False)
    
    summary = {
        'optimal_threshold': float(optimal_threshold),
        'best_threshold': float(best_thresh),
        'best_f1': float(best_f1),
        'pr_auc': float(pr_auc),
        'mean_prob': float(y_scores.mean()),
        'median_prob': float(np.median(y_scores)),
        'std_prob': float(y_scores.std()),
        'authentic_mean': float(authentic_scores.mean()),
        'manipulated_mean': float(manipulated_scores.mean()),
    }
    
    import json
    with open(os.path.join(output_dir, 'threshold_recommendation.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n💾 Recomendações salvas em {output_dir}/threshold_recommendation.json")
    print(f"\n✅ Use threshold = {best_thresh:.3f} para melhor resultado!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/raw", help="Dataset directory")
    parser.add_argument("--model", default="models/fusion_model.pth", help="Path to model")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    analyze_probability_distribution(args.data, args.model, device)

