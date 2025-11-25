FROM python:3.9-slim

WORKDIR /app

# Dependências de sistema para OpenCV e Tesseract
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-por \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Instalar dependências Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código
COPY . .

# Criar diretório para static
RUN mkdir -p src/api/static/heatmaps

# Expor porta 8080
EXPOSE 8080

# Executar
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8080"]

