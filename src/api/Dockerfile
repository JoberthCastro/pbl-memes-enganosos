FROM python:3.9-slim

WORKDIR /app

# Instalar dependencias de sistema (Tesseract)
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Expor porta API
EXPOSE 8000

# Comando para rodar API
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

