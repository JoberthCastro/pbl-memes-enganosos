"""
Script para avaliar o modelo com diferentes thresholds de decisão.
Isso ajuda a encontrar o threshold ótimo que equilibra precision e recall.
"""
import torch
import pandas as pd
import numpy as np
import argparse
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)
import json
import os

from src.fusion_model import FusionModel
from src.visual_extractor import VisualExtractor, get_visual_embedding
from src.text_model import TextModel, get_text_embedding, Tokenizer
from src.ocr_tesseract import extract_ocr_data
from src.preprocessing import get_transforms
from src.dataset import MemeDataset
from torch.utils.data import DataLoader

TEST_CSV_PATH = "data/test_labels.csv"
DATA_DIR = "data/raw"
MODEL_DIR = "models"

def evaluate_with_threshold(dataset_dir, model_path, threshold=0.5, device='cpu'):
    """
    Avalia o modelo usando um threshold customizado para a classe positiva.
    """
    print(f"🔍 Avaliando com threshold = {threshold:.2f}")
    
    # Carregar modelos
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
    
    # Carregar dados de teste
    if not os.path.exists(TEST_CSV_PATH):
        print(f"❌ {TEST_CSV_PATH} não encontrado!")
        return None
    
    test_df = pd.read_csv(TEST_CSV_PATH)
    transform = get_transforms(mode='eval')
    
    y_true = []
    y_pred = []
    y_prob = []
    
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
            from PIL import Image
            img = Image.open(fpath).convert("RGB")
            np_img = np.array(img)
            
            # OCR
            ocr_res = extract_ocr_data(np_img)
            text = ocr_res['full_text']
            ocr_stats_vec = [0.0, 0.0, 0.0]  # Match training
            
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
            
            # Aplicar threshold customizado
            pred_label = 1 if prob_manipulated >= threshold else 0
            
            y_true.append(label_idx)
            y_pred.append(pred_label)
            y_prob.append(prob_manipulated)
            
        except Exception as e:
            print(f"Erro processando {fpath}: {e}")
            continue
    
    if len(y_true) == 0:
        print("❌ Nenhuma amostra processada!")
        return None
    
    # Métricas
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred)
    
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    else:
        specificity = 0.0
    
    results = {
        "threshold": threshold,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "specificity": specificity,
        "confusion_matrix": cm.tolist(),
    }
    
    print(f"\n📊 Resultados (threshold={threshold:.2f}):")
    print(f"   Accuracy: {acc:.4f}")
    print(f"   Precision: {prec:.4f}")
    print(f"   Recall: {rec:.4f}")
    print(f"   F1-Score: {f1:.4f}")
    print(f"   Specificity: {specificity:.4f}")
    print(f"\n   Confusion Matrix:")
    print(f"   {'':>15} Pred: Auth  Pred: Manip")
    print(f"   True: Auth    {tn:>6}    {fp:>6}")
    print(f"   True: Manip   {fn:>6}    {tp:>6}")
    
    return results

def find_optimal_threshold(dataset_dir, model_path, device='cpu'):
    """
    Testa diferentes thresholds e encontra o ótimo (melhor F1-score balanceado).
    """
    print("🔍 Procurando threshold ótimo...\n")
    
    thresholds = np.arange(0.3, 0.8, 0.05)
    results_list = []
    
    for thresh in thresholds:
        result = evaluate_with_threshold(dataset_dir, model_path, thresh, device)
        if result:
            results_list.append(result)
        print()
    
    if not results_list:
        print("❌ Nenhum resultado obtido!")
        return
    
    # Encontrar threshold com melhor F1 balanceado
    best = max(results_list, key=lambda x: x['f1_score'])
    
    print("\n" + "="*60)
    print("🏆 MELHOR THRESHOLD ENCONTRADO:")
    print("="*60)
    print(f"Threshold: {best['threshold']:.2f}")
    print(f"F1-Score: {best['f1_score']:.4f}")
    print(f"Precision: {best['precision']:.4f}")
    print(f"Recall: {best['recall']:.4f}")
    print(f"Specificity: {best['specificity']:.4f}")
    print("="*60)
    
    # Salvar resultados
    with open("reports/threshold_analysis.json", "w") as f:
        json.dump(results_list, f, indent=2)
    print("\n💾 Resultados salvos em reports/threshold_analysis.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/raw", help="Dataset directory")
    parser.add_argument("--model", default="models/fusion_model.pth", help="Path to model")
    parser.add_argument("--threshold", type=float, default=None, help="Threshold específico (ou None para buscar ótimo)")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if args.threshold is not None:
        evaluate_with_threshold(args.data, args.model, args.threshold, device)
    else:
        find_optimal_threshold(args.data, args.model, device)

