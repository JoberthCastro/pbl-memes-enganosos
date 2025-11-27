# 🚀 Guia de Execução - Classificação de Memes Enganosos

Este guia vai te ajudar a rodar o projeto do zero até ter a aplicação funcionando.

## 📋 Pré-requisitos

### 1. Python 3.9 ou superior
```bash
python --version
```

### 2. Tesseract OCR
**Windows:**
- Baixe em: https://github.com/UB-Mannheim/tesseract/wiki
- Instale em: `C:\Program Files\Tesseract-OCR` (padrão)
- Adicione ao PATH ou configure o caminho no código

**Linux:**
```bash
sudo apt-get install tesseract-ocr
sudo apt-get install libtesseract-dev
```

**Mac:**
```bash
brew install tesseract
```

### 3. (Opcional) Google Gemini API Key
- Para usar validação semântica com LLM
- Obtenha em: https://makersuite.google.com/app/apikey
- Crie arquivo `.env` na raiz do projeto:
```
GEMINI_API_KEY=sua_chave_aqui
```

## 🔧 Instalação

### Passo 1: Instalar dependências Python
```bash
# No diretório do projeto
pip install -r requirements.txt
```

Ou usando o Makefile:
```bash
make setup
```

### Passo 2: Configurar Tesseract (Windows)
Se o Tesseract não estiver no caminho padrão, edite `src/ocr_tesseract.py` linha 17:
```python
TESSERACT_WINDOWS_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
```

## 📊 Gerar Dataset

O projeto precisa de um dataset para treinar. Você pode gerar dados sintéticos:

```bash
# Gerar 50 imagens autênticas e 50 manipuladas
python data/synthetic_generator.py --n_authentic 50 --n_manipulated 50
```

Ou usando o Makefile:
```bash
make data
```

Isso vai criar:
- `data/raw/authentic/` - Imagens autênticas
- `data/raw/manipulated/` - Imagens manipuladas
- `data/labels.csv` - Metadados das imagens

## 🎓 Treinar o Modelo

Após gerar o dataset, treine o modelo:

```bash
python src/train.py
```

Ou usando o Makefile:
```bash
make train
```

O treinamento vai:
- Separar dados em treino (80%) e teste (20%)
- Treinar por 3 épocas
- Salvar modelos em `models/`:
  - `visual_model.pth`
  - `text_model.pth`
  - `fusion_model.pth`

**Nota:** O treinamento pode demorar alguns minutos dependendo do hardware.

## 🧪 Avaliar o Modelo

Após o treinamento, avalie o modelo:

```bash
python src/evaluate.py --data data/raw --model models/fusion_model.pth
```

Isso gera relatórios em `reports/`:
- `metrics.json` - Métricas numéricas
- `confusion_matrix.png` - Matriz de confusão
- `precision_recall_curve.png` - Curva PR
- `evaluation_results.csv` - Resultados detalhados
- `report.md` - Relatório completo

## 🌐 Rodar a API

### Opção 1: Localmente (Recomendado para desenvolvimento)
```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8080 --reload
```

Ou usando o Makefile:
```bash
make run-api
```

A API estará disponível em: `http://localhost:8080`

**Endpoints:**
- `GET /health` - Verifica se a API está funcionando
- `POST /infer` - Classifica uma imagem

**Testar a API:**
```bash
curl -X POST "http://localhost:8080/infer" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@caminho/para/imagem.jpg" \
     -F "platform=twitter"
```

### Opção 2: Docker
```bash
# Build
make docker-build-api

# Run
make docker-run-api
```

## 🖥️ Rodar Interface Streamlit

Para uma interface web mais amigável:

```bash
streamlit run streamlit_app.py
```

A interface abrirá automaticamente no navegador em `http://localhost:8501`

**Funcionalidades:**
- Upload de imagem
- Visualização do resultado
- Heatmap Grad-CAM
- Análise de OCR
- Explicação do LLM

## 🐛 Solução de Problemas

### Erro: "Tesseract not found"
- Verifique se o Tesseract está instalado
- Configure o caminho em `src/ocr_tesseract.py`
- No Windows, adicione ao PATH do sistema

### Erro: "CUDA out of memory"
- O modelo roda em CPU por padrão
- Se tiver GPU, pode acelerar, mas precisa de mais memória
- Reduza o `batch_size` em `src/train.py` se necessário

### Erro: "No module named 'src'"
- Certifique-se de estar na raiz do projeto
- Execute: `export PYTHONPATH="${PYTHONPATH}:$(pwd)"` (Linux/Mac)
- Ou: `set PYTHONPATH=%PYTHONPATH%;%CD%` (Windows)

### Erro: "GEMINI_API_KEY not found"
- Isso é normal! O sistema usa modo mock se não houver chave
- Para usar LLM real, crie `.env` com a chave

### Modelos não encontrados
- Certifique-se de ter treinado o modelo primeiro (`make train`)
- Ou baixe modelos pré-treinados (se disponíveis)

## 📁 Estrutura de Arquivos Importantes

```
pbl-memes-enganosos/
├── data/
│   ├── raw/              # Imagens do dataset
│   │   ├── authentic/
│   │   └── manipulated/
│   ├── labels.csv        # Metadados
│   └── synthetic_generator.py
├── models/              # Modelos treinados (gerados)
│   ├── visual_model.pth
│   ├── text_model.pth
│   └── fusion_model.pth
├── src/
│   ├── api/main.py      # API FastAPI
│   ├── train.py         # Script de treinamento
│   ├── evaluate.py      # Script de avaliação
│   └── ...
├── streamlit_app.py     # Interface web
├── requirements.txt     # Dependências
└── Makefile            # Comandos úteis
```

## 🎯 Fluxo Completo Recomendado

1. **Instalar dependências:**
   ```bash
   make setup
   ```

2. **Gerar dataset:**
   ```bash
   make data
   ```

3. **Treinar modelo:**
   ```bash
   make train
   ```

4. **Avaliar modelo:**
   ```bash
   python src/evaluate.py
   ```

5. **Rodar interface (escolha uma):**
   - **Streamlit (mais fácil):**
     ```bash
     streamlit run streamlit_app.py
     ```
   - **API REST:**
     ```bash
     make run-api
     ```

## 💡 Dicas

- Comece com poucas imagens (20-50) para testar rapidamente
- Use o Streamlit para visualizar resultados facilmente
- A API é melhor para integração com outros sistemas
- O modo mock do LLM funciona, mas a análise real é mais precisa

## 📞 Próximos Passos

- Adicione mais dados ao dataset para melhorar a precisão
- Ajuste hiperparâmetros em `src/train.py`
- Configure a API do Gemini para análise semântica real
- Deploy em produção usando Docker

---

**Boa sorte! 🚀**

