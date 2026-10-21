.PHONY: install train train-synthetic test serve drift mlflow docker

install:
	pip install -r requirements.txt

train:            ## real data at data/creditcard.csv
	python -m fraud.train

train-synthetic:
	python -m fraud.train --synthetic

test:
	pytest -q

serve:
	uvicorn fraud.serve:app --port 8000 --reload

drift:
	python -m fraud.drift --demo-shift

mlflow:
	mlflow ui --backend-store-uri sqlite:///mlflow.db

docker:
	docker build -t fraud-api .
