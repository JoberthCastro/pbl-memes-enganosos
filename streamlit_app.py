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


# =================== CARREGAMENTO DOS MODELOS ===================
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


# =================== INFERÊNCIA ===================
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

    # Predição do Modelo Fusion (PyTorch) com threshold ótimo
    OPTIMAL_THRESHOLD = 0.70
    with torch.no_grad():
        logits = fusion(t_v_emb, t_t_emb, t_ocr)
        probs = torch.softmax(logits, dim=1)
        prob_manipulated = probs[0, 1].item()
        fusion_label_idx = 1 if prob_manipulated >= OPTIMAL_THRESHOLD else 0
        fusion_score = float(
            prob_manipulated if fusion_label_idx == 1 else (1 - prob_manipulated)
        )

    # LLM Análise — passando também a predição do modelo treinado
    llm_meta = {
        "platform": platform,
        "model_prediction": "enganoso" if fusion_label_idx == 1 else "autêntico",
        "model_confidence": fusion_score,
    }
    llm_response = llm.analyze(text_extracted, llm_meta)

    # ===== Fusão Híbrida =====
    llm_score = llm_response.get("score", 0.0)  # 0.0 a 1.0 (1.0 = Fake)
    llm_label_str = llm_response.get("label", "").lower()

    is_llm_fake = llm_score > 0.5 or llm_label_str in [
        "suspeito",
        "erro",
        "manipulado",
        "fake",
    ]
    is_llm_safe = llm_score < 0.3 or llm_label_str == "autêntico"
    is_visual_fake = fusion_label_idx == 1

    final_label = "Autêntico"
    final_idx = 0
    final_score = fusion_score

    if is_visual_fake:
        if is_llm_safe:
            # Visual diz Fake, mas LLM diz seguro → confiar no LLM
            final_label = "Autêntico (Validado pelo LLM)"
            final_idx = 0
            final_score = (1 - fusion_score + (1 - llm_score)) / 2
        else:
            final_label = "Enganoso/Suspeito"
            final_idx = 1
    else:
        if is_llm_fake:
            # Visual diz autêntico, mas LLM vê problema → alerta
            final_label = "Suspeito (Alertado pelo LLM)"
            final_idx = 1
            final_score = (fusion_score + llm_score) / 2

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
    except Exception:
        # Não quebra a interface se Grad-CAM falhar
        heatmap_img = None

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


# =================== INTERFACE FUTURISTA ===================
def main():
    st.set_page_config(
        page_title="Classificação de Memes Enganosos",
        page_icon="🧬",
        layout="wide",
    )

    # ============== CSS FUTURISTA ==============
    st.markdown(
        """
    <style>
        .main {
            background: radial-gradient(circle at 20% 20%, #0b0f19, #000000 70%);
            color: #e0e0e0;
        }
        h1, h2, h3 {
            color: #a855f7 !important;  /* roxo suave */
            text-shadow: 0 0 4px #6d28d9;  /* brilho bem mais discreto */
        }
        .stButton>button {
            background: linear-gradient(90deg, #a855f7, #7c3aed);
            color: white;
            border-radius: 10px;
            box-shadow: 0 0 10px #7c3aed88;
            font-weight: bold;
            transition: 0.2s;
        }
        .stButton>button:hover {
            transform: scale(1.03);
            box-shadow: 0 0 15px #7c3aedaa;
        }
        img {
            border-radius: 10px;
            box-shadow: 0 0 18px #7c3aed55;
        }
        .stMetric {
            background: rgba(0,0,0,0.35);
            border: 1px solid #7c3aed55;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 0 18px #7c3aed33;
        }
    </style>
    """,
        unsafe_allow_html=True,
    )

    # ============== HEADER ==============
    st.markdown(
        """
    <div style="text-align:center;">
        <h1>🔮 Classificação de Memes Enganosos</h1>
        <p style="color:#9bdfff; font-size:18px;">
            Sistema Híbrido — Visão Computacional + OCR + LLM (Gemini)<br>
            PBL 4 • UNDB
        </p>
    </div>
    <hr>
    """,
        unsafe_allow_html=True,
    )

    # ============== UPLOAD ==============
    left, right = st.columns([2.2, 1])

    with left:
        st.markdown("### 📤 Envie uma imagem")
        uploaded_file = st.file_uploader(
            "", type=["jpg", "jpeg", "png"], label_visibility="collapsed"
        )
        st.markdown("### 🌐 Plataforma (opcional)")
        platform = st.selectbox(
            "",
            ["unknown", "twitter", "instagram", "facebook", "whatsapp"],
            label_visibility="collapsed",
        )
        analyze_btn = st.button("🚀 Iniciar análise", use_container_width=True)

    with right:
        st.markdown("### 🧠 Como funciona?")
        st.markdown(
            "- 🔎 OCR com Tesseract  \n"
            "- 👁️ Modelo Visual (CNN)  \n"
            "- 🤖 LLM Gemini para análise semântica  \n"
            "- 🔗 Fusão híbrida inteligente Vision + Text + OCR"
        )
        st.info("Ideal para prints, manchetes, tweets, posts e memes.")

    st.markdown("<br>", unsafe_allow_html=True)

    if not uploaded_file:
        st.info("Envie uma imagem para começar.")
        return

    if uploaded_file and analyze_btn:
        image = Image.open(io.BytesIO(uploaded_file.read())).convert("RGB")

        with st.spinner("⏳ Processando..."):
            result = run_inference(image, platform)

        img_col, info_col = st.columns([1.3, 1])

        # ----- Coluna da imagem -----
        with img_col:
            st.markdown("### 🖼️ Imagem")
            st.image(image, use_column_width=True)

            if result["heatmap_img"] is not None:
                st.markdown("### 🔥 Regiões relevantes (Grad-CAM)")
                st.image(result["heatmap_img"], use_column_width=True)

        # ----- Coluna de informações -----
        with info_col:
            st.markdown("### 🎯 Resultado")

            if result["label_idx"] == 1:
                st.error(f"### ❌ {result['label']}")
            else:
                st.success(f"### ✔ {result['label']}")

            st.metric(
                "Confiança combinada",
                f"{result['confidence_score']*100:.2f}%",
            )

            st.markdown("---")
            with st.expander("📌 Detalhes da classificação"):
                st.write(f"**Modelo Visual/OCR:** {result['fusion_raw_prediction']}")
                st.write(
                    f"**LLM (rótulo):** {result['llm_explanation'].get('label','N/A')}"
                )
                st.write(
                    f"**LLM (score):** {result['llm_explanation'].get('score',0.0)}"
                )

            # ----- Texto OCR -----
            st.markdown("### 📝 Texto OCR")
            if result["ocr_text"]:
                st.info(result["ocr_text"])
            else:
                st.caption("Nenhum texto detectado pelo OCR.")

            if result["ocr_evidence"]["low_confidence_words"]:
                st.markdown("### ⚠ Palavras suspeitas no OCR")
                st.json(result["ocr_evidence"])

            # ----- Análise LLM (apenas explicação bonita) -----
            st.markdown("### 🤖 Análise semântica (LLM / Gemini)")
            explanation = result["llm_explanation"].get("explanation", "")
            if explanation:
                st.markdown(f"> {explanation}")
            else:
                st.caption("Nenhuma explicação disponível.")


if __name__ == "__main__":
    main()
