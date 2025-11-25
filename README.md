# PBL 4 - Classificação de Memes Enganosos

Este repositório contém a implementação de um sistema multimodal para classificação de memes como "Autênticos" ou "Enganosos/Suspeitos". O projeto utiliza **Visão Computacional** (MobileNetV2), **NLP** (LSTM/Embeddings) e integração com **LLM (Gemini)** para validação semântica.

## Estrutura

- `src/`: Código fonte dos modelos e pipeline.
- `data/`: Scripts de geração de dados e armazenamento.
- `src/api/`: API FastAPI para inferência.
- `docker/`: Dockerfiles para treino e deploy.

## Como usar

### 1. Setup
```bash
make setup
```

### 2. Gerar Dataset Sintético
Gera imagens de memes fake/real para teste.
```bash
make data
```

### 3. Treinar Modelo
```bash
make train
```

### 4. Rodar API
Para rodar a API localmente:
```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8080 --reload
```

Ou usando Docker:
```bash
docker build -t meme-api -f docker/api.Dockerfile .
docker run -p 8080:8080 meme-api
```

## API Usage

### Check Health
```http
GET /health
```

### Inference
Enviar uma imagem via POST (multipart/form-data).

**Exemplo cURL:**
```bash
curl -X POST "http://localhost:8080/infer" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@teste_meme.jpg" \
     -F "platform=twitter"
```

**Resposta Exemplo:**
```json
{
  "request_id": "uuid...",
  "label": "Suspeito",
  "confidence_score": 0.98,
  "ocr_text": "URGENTE: ...",
  "ocr_evidence": [
    {"text": "fake", "conf": 45.0, "bbox": [10, 10, 50, 20]}
  ],
  "llm_explanation": {
    "label": "suspeito",
    "score": 0.9,
    "explanation": "Texto alarmista..."
  },
  "heatmap_url": "/static/heatmaps/heatmap_uuid.png"
}
```

## Arquitetura

1. **Entrada**: Imagem do meme.
2. **Visual**: MobileNetV2 extrai vetor de características (1280 dim).
3. **OCR**: Tesseract extrai texto da imagem e métricas de confiança.
4. **Textual**: Texto é convertido em embeddings e processado por Bi-LSTM (256 dim).
5. **Fusão**: Late Fusion (concatenação) dos vetores visual, textual e stats OCR -> MLP Classifier.
6. **Validação LLM**: O texto extraído é enviado ao Gemini para checagem de fatos/análise de sentimento.
7. **Interpretabilidade**: Grad-CAM gera heatmap das regiões visuais críticas.

## Requisitos
- Python 3.9+
- Tesseract OCR
