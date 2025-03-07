.PHONY: run_all clean

run_all: features train tune evaluate forecast 

features:
	python scripts/feature_engineering.py \
	--input_inhaler data/processed/inhaler_air_merged \
	--input_patients data/raw/iot_inhaler/patients.parquet \
	--output data/processed/train_ready/clinical_df_v1.csv

train:	
	python src/train.py \
	--data data/processed/features.csv \
	--model models/baseline.pkl

tune:
	python src/hyperparameter.py \
	--model models/baseline.pkl \
	--output models/tuned.pkl

evaluate:
	python src/model_evaluation.py \
	--model models/tuned.pkl \
	--report reports/evaluation.html

forecast:
	python src/model_forecast.py \
	--model models/tuned.pkl \
	--output data/forecasts/predictions.csv

clean:
	rm -rf data/processed/* models/* reports/*
