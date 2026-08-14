.PHONY: install train-bigru train-distilbert evaluate deploy test clean

install:
	pip install -r requirements.txt

train-bigru:
	python -m src.train_bigru

train-distilbert:
	python -m src.train_distilbert

evaluate:
	python -m src.evaluate

deploy:
	python -m src.deploy

test:
	python test_api.py

mlflow:
	mlflow ui --port 5000

all: train-bigru train-distilbert evaluate

clean:
	rm -rf models/* logs/* mlflow_runs/* __pycache__ .pytest_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

docker-build:
	docker-compose build

docker-up:
	docker-compose up

docker-down:
	docker-compose down

docker-clean:
	docker-compose down -v
	rm -rf models/* logs/*