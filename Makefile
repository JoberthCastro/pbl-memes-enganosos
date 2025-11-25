.PHONY: setup data train test run-api docker-build-api docker-build-train docker-run-api clean

# Variáveis
IMAGE_API_NAME=meme-classifier-api
IMAGE_TRAIN_NAME=meme-classifier-train
PORT=8080

setup:
	pip install -r requirements.txt
	
data:
	python data/synthetic_generator.py --n_authentic 50 --n_manipulated 50

train:
	python src/train.py

test:
	pytest tests/

run-api:
	uvicorn src.api.main:app --host 0.0.0.0 --port $(PORT) --reload

docker-build-api:
	docker build -t $(IMAGE_API_NAME) -f docker/api.Dockerfile .

docker-build-train:
	docker build -t $(IMAGE_TRAIN_NAME) -f docker/Dockerfile.train .

docker-run-api:
	docker run -p $(PORT):$(PORT) $(IMAGE_API_NAME)

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
