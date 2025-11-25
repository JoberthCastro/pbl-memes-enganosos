# Metodologia: Classificação de Memes Enganosos (PBL 4)

## 1. Visão Geral do Pipeline

O sistema foi projetado como uma arquitetura multimodal de fusão tardia (*late fusion*), combinando visão computacional, processamento de linguagem natural (NLP) e verificação semântica via LLM.

### Fluxo de Dados:
1. **Entrada**: Imagem (JPG/PNG).
2. **Pré-processamento**: 
   - Normalização de intensidade.
   - Denoising (remoção de artefatos de compressão).
   - Resize com preservação de *aspect ratio* (letterboxing).
3. **Extração de Características (Paralelo)**:
   - **Visual**: `MobileNetV2` (pre-trained ImageNet) extrai vetor de features (1280d).
   - **OCR**: `Tesseract` extrai texto bruto e metadados de confiança.
   - **Textual**: `Bi-LSTM` processa embeddings do texto extraído (256d).
4. **Fusão Multimodal**:
   - Concatenação dos vetores Visual + Textual + Estatísticas OCR.
   - Classificador MLP (`Dense -> ReLU -> Dropout`).
5. **Inferência & Validação**:
   - Predição de classe (Autêntico vs. Enganoso).
   - Geração de mapa de calor (*Grad-CAM*) para explicabilidade.
   - Validação semântica via LLM (Gemini) para detecção de incoerências lógicas.

---

## 2. Decisões Arquiteturais

### 2.1. Modelo Visual: MobileNetV2
- **Justificativa**: Escolhido pelo balanço ideal entre performance (acurácia no ImageNet) e eficiência computacional. Essencial para viabilizar a inferência em contêineres Docker sem GPU dedicada.
- **Adaptação**: Remoção do "top" (classificador original) e uso de *Global Average Pooling* para obter um vetor de características denso.

### 2.2. Modelo Textual: Bi-LSTM vs Transformers
- **Decisão**: Uso de `Bi-LSTM` com embeddings treináveis.
- **Motivo**: Memes possuem textos curtos e muitas vezes gramaticalmente incorretos. Transformers (BERT) seriam overkill computacional para o escopo e poderiam sofrer com o vocabulário ruidoso do OCR sem fine-tuning pesado. LSTMs bidirecionais capturam contexto suficiente com muito menos parâmetros.

### 2.3. Fusão Tardia (Late Fusion)
- **Abordagem**: Processar cada modalidade independentemente e fundir apenas no final.
- **Vantagem**: Permite que cada "braço" da rede especialize-se em seu domínio. Facilita o debug (ex: saber se o erro veio do OCR ruim ou da imagem).

### 2.4. Explainable AI (XAI)
- **Grad-CAM**: Implementado para gerar transparência visual. Permite auditar se o modelo está olhando para o texto manipulado ou para o fundo da imagem.
- **LLM como Juiz**: O uso de um LLM como "segunda opinião" adiciona uma camada de análise lógica que redes neurais puras têm dificuldade em realizar (ex: checar se uma data no texto condiz com o evento histórico).

---

## 3. Dataset e Geração Sintética

Dada a escassez de datasets públicos rotulados de "Fake News em Memes" em PT-BR, desenvolvemos um gerador sintético robusto (`data/synthetic_generator.py`).

- **Autênticos**: Templates simulando tweets e mensagens de WhatsApp com textos neutros/positivos e métricas plausíveis.
- **Manipulados**: 
  - Inserção de caixas de texto ("splicing").
  - Alteração de métricas para valores absurdos (ex: 999M likes).
  - Degradação visual (compressão JPEG agressiva).
  - Conteúdo textual alarmista/sensacionalista.

---

## 4. Métricas de Avaliação

O sistema é avaliado utilizando as seguintes métricas no conjunto de validação:

1. **Acurácia**: Taxa global de acertos.
2. **F1-Score**: Média harmônica entre precisão e recall, crucial dado o desbalanceamento potencial e o custo de falsos negativos em desinformação.
3. **Confidence do OCR**: Monitoramos a média de confiança do Tesseract. Baixa confiança correlaciona-se com manipulações visuais de texto.
4. **Latência de Inferência**: Tempo total desde o upload até a resposta JSON.

