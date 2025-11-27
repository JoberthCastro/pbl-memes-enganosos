import os
import io

import numpy as np
from PIL import Image

import torch
import streamlit as st

from src.visual_extractor import VisualExtractor, get_visual_embedding
from src.text_model import TextModel, Tokenizer, get_text_embedding
from src.fusion_model import FusionModel
from src.ocr_tesseract import extract_ocr_data
from src.llm_integration import LLMIntegration
from src.interpretability import GradCAM, overlay_heatmap, gather_ocr_evidence
from src.preprocessing import get_transforms


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@st.cache_resource
def load_models():
    """Carrega modelos uma única vez para uso no Streamlit."""
    # Visual
    visual = VisualExtractor(model_name="mobilenet_v2")
    visual.to(DEVICE)
    visual.eval()

    visual_ckpt = "models/visual_model.pth"
    if os.path.exists(visual_ckpt):
        visual.load_state_dict(torch.load(visual_ckpt, map_location=DEVICE))

    # Tokenizer (não treinado persistente, segue o padrão da API)
    tokenizer = Tokenizer(num_words=10000)

    # Text model
    text_model = TextModel(vocab_size=10001, embedding_dim=128, hidden_dim=128)
    text_model.to(DEVICE)
    text_model.eval()

    text_ckpt = "models/text_model.pth"
    if os.path.exists(text_ckpt):
        try:
            text_model.load_state_dict(torch.load(text_ckpt, map_location=DEVICE))
        except RuntimeError:
            # Se der mismatch de dimensão, seguimos com pesos aleatórios
            pass

    # Fusion (usa embeddings pré-calculados, como em train/evaluate)
    fusion_config = {"num_classes": 2, "text_output_dim": 256}
    fusion = FusionModel(fusion_config)
    fusion_ckpt = "models/fusion_model.pth"
    if os.path.exists(fusion_ckpt):
        fusion.load_state_dict(torch.load(fusion_ckpt, map_location=DEVICE))
    fusion.to(DEVICE)
    fusion.eval()

    # LLM (usa GEMINI_API_KEY se existir, senão mock_mode=True)
    llm = LLMIntegration(mock_mode=False)

    return {
        "visual": visual,
        "tokenizer": tokenizer,
        "text_model": text_model,
        "fusion": fusion,
        "llm": llm,
    }


def run_inference(image: Image.Image, platform: str = "unknown"):
    models = load_models()

    visual = models["visual"]
    tokenizer = models["tokenizer"]
    text_model = models["text_model"]
    fusion = models["fusion"]
    llm = models["llm"]

    np_image = np.array(image)

    # OCR (threshold mais baixo para evitar perder texto válido)
    ocr_result = extract_ocr_data(np_image, confidence_threshold=10)
    text_extracted = ocr_result["full_text"]
    ocr_stats = [
        ocr_result["stats"]["mean_conf"],
        ocr_result["stats"]["std_conf"],
        len([w for w in ocr_result["words"] if w["conf"] < 50]),
    ]
    ocr_evidence = gather_ocr_evidence(ocr_result["words"])

    # Embeddings visual e textual
    visual_emb = get_visual_embedding(image, model=visual, device=DEVICE)
    text_emb = get_text_embedding(
        text_extracted, tokenizer, text_model, device=DEVICE
    )

    t_v_emb = torch.tensor([visual_emb], dtype=torch.float32).to(DEVICE)
    t_t_emb = torch.tensor([text_emb], dtype=torch.float32).to(DEVICE)
    t_ocr = torch.tensor([ocr_stats], dtype=torch.float32).to(DEVICE)

    # Predição do Modelo Fusion (PyTorch)
    # Aplicar threshold ótimo (0.70) para melhor balanceamento
    # Threshold atualizado após retreino com mais dados
    OPTIMAL_THRESHOLD = 0.70
    with torch.no_grad():
        logits = fusion(t_v_emb, t_t_emb, t_ocr)
        probs = torch.softmax(logits, dim=1)
        prob_manipulated = probs[0, 1].item()
        fusion_label_idx = 1 if prob_manipulated >= OPTIMAL_THRESHOLD else 0
        fusion_score = float(prob_manipulated if fusion_label_idx == 1 else (1 - prob_manipulated))
    
    # LLM Análise
    llm_meta = {
        "platform": platform,
        "metrics": "N/A (Auto-detected)",
        "visual_cues": [],
    }
    llm_response = llm.analyze(text_extracted, llm_meta)
    
    # --- LÓGICA DE FUSÃO HÍBRIDA (Bidirecional) ---
    
    llm_score = llm_response.get("score", 0.0) # 0.0 a 1.0 (1.0 = Fake)
    llm_label_str = llm_response.get("label", "").lower()
    
    # Definições
    is_llm_fake = llm_score > 0.5 or llm_label_str in ["suspeito", "erro", "manipulado", "fake"]
    is_llm_safe = llm_score < 0.3 or llm_label_str == "autêntico"
    
    is_visual_fake = (fusion_label_idx == 1)
    
    final_label = "Autêntico"
    final_idx = 0
    final_score = fusion_score

    if is_visual_fake: # Visual diz FAKE
        if is_llm_safe:
            # CONFLITO: Visual diz Fake, mas LLM diz Seguro (conteúdo real).
            # Decisão: Confiar no LLM para evitar censura de notícias reais com artefatos visuais.
            final_label = "Autêntico (Validado pelo LLM)"
            final_idx = 0
            final_score = (1.0 - fusion_score + (1.0 - llm_score)) / 2 # Inverte score visual para confiança 'safe'
        else:
            # Concordância ou LLM em cima do muro
            final_label = "Enganoso/Suspeito"
            final_idx = 1
            
    else: # Visual diz AUTÊNTICO
        if is_llm_fake:
            # CONFLITO: Visual diz Seguro, mas LLM diz Fake (texto perigoso).
            # Decisão: Confiar no LLM para segurança.
            final_label = "Suspeito (Alertado pelo LLM)"
            final_idx = 1
            final_score = (fusion_score + llm_score) / 2
        else:
            # Concordância Total
            final_label = "Autêntico"
            final_idx = 0
    
    # -------------------------------------------------------

    # Grad-CAM heatmap
    heatmap_img = None
    try:
        target_layer = visual.features[-1]
        cam = GradCAM(visual, target_layer)

        transform = get_transforms(mode="eval")
        img_tensor = transform(image).unsqueeze(0).to(DEVICE)
        img_tensor.requires_grad = True

        mask = cam(img_tensor)
        heatmap_img = overlay_heatmap(np_image, mask)
    except Exception as e:
        st.warning(f"Falha ao gerar Grad-CAM: {e}")

    return {
        "label": final_label,
        "label_idx": final_idx,
        "confidence_score": final_score,
        "fusion_raw_prediction": "Autêntico" if fusion_label_idx == 0 else "Enganoso",
        "ocr_text": text_extracted,
        "ocr_evidence": ocr_evidence,
        "llm_explanation": llm_response,
        "heatmap_img": heatmap_img,
    }


def main():
    st.set_page_config(
        page_title="Classificação de Memes Enganosos",
        page_icon="🧐",
        layout="wide",
    )

    st.title("🧐 Classificação de Memes Enganosos — PBL 4 UNDB")
    st.markdown(
        "Envie um meme ou print de rede social para analisar se o conteúdo parece **autêntico** "
        "ou **enganoso/suspeito**, combinando visão computacional, OCR e LLM (Gemini)."
    )

    cols = st.columns([2, 1])

    with cols[0]:
        uploaded_file = st.file_uploader(
            "Envie uma imagem (JPG/PNG)", type=["jpg", "jpeg", "png"]
        )
        platform = st.selectbox(
            "Plataforma (opcional)", ["unknown", "twitter", "instagram", "facebook", "whatsapp"]
        )
        analyze_btn = st.button("Analisar meme", type="primary", use_container_width=True)

    if uploaded_file and analyze_btn:
        image = Image.open(io.BytesIO(uploaded_file.read())).convert("RGB")

        with st.spinner("Rodando OCR, modelos e LLM..."):
            result = run_inference(image, platform=platform)

        col_img, col_info = st.columns([1, 1])

        with col_img:
            st.subheader("Imagem original")
            st.image(image, use_column_width=True)

            if result["heatmap_img"] is not None:
                st.subheader("Regiões mais relevantes (Grad-CAM)")
                st.image(result["heatmap_img"], caption="Heatmap de atenção do modelo", use_column_width=True)

        with col_info:
            st.subheader("Resultado da classificação")
            
            # Cor dinâmica baseada no resultado
            if result["label_idx"] == 1:
                st.error(f"**{result['label']}**")
            else:
                st.success(f"**{result['label']}**")
            
            st.metric(
                "Confiança Combinada",
                f"{result['confidence_score']*100:.2f}%",
                delta="-Suspeito" if result["label_idx"] == 1 else "+Autêntico",
                delta_color="inverse"
            )
            
            with st.expander("Detalhes da Decisão"):
                st.write(f"**Modelo Visual/OCR:** {result['fusion_raw_prediction']}")
                st.write(f"**Análise Semântica (LLM):** {result['llm_explanation'].get('label', 'N/A')}")

            st.markdown("### Texto extraído via OCR")
            if result["ocr_text"]:
                st.write(result["ocr_text"])
            else:
                st.caption("Nenhum texto detectado pelo OCR.")

            if result["ocr_evidence"]["low_confidence_words"]:
                st.markdown("### Palavras com baixa confiança no OCR")
                st.json(result["ocr_evidence"])

            st.markdown("### Análise semântica (LLM / Gemini)")
            st.json(result["llm_explanation"])

    elif not uploaded_file:
        st.info("Envie uma imagem para começar a análise.")


if __name__ == "__main__":
    main()
