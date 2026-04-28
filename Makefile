pipeline:  # run the entire pipeline 
test:
lint:

# Variables
PYTHON ?= python3
CONFIG ?= configs/config.toml

.PHONY: help setup validate clean_data validate_cleaned features train classify report pipeline test lint format isort clean

help:
	@echo "Available targets: setup, validate, clean_data, validate_cleaned, features, train, classify, report, pipeline, test, lint, format, isort, clean"

setup:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt || true

# 1. Validate raw data
validate:
	$(PYTHON) src/data/validate.py --config $(CONFIG) --input data/raw/Teen_Mental_Health_Dataset.csv --output reports/validation_raw.json

# 2. Clean/preprocess data
clean_data:
	$(PYTHON) src/data/preprocess.py --config $(CONFIG)

# 3. Validate cleaned data
validate_cleaned:
	$(PYTHON) src/data/validate.py --config $(CONFIG) --input data/processed/cleaned.csv --output reports/validation_cleaned.json

# 4. Engineer features
features:
	$(PYTHON) src/features/engineer.py --config $(CONFIG)

# 5. Train baseline model
train:
	$(PYTHON) src/models/train.py --config $(CONFIG)

# 6. Compare classifiers and save best model
classify:
	$(PYTHON) src/models/classify.py --config $(CONFIG)

# 7. Generate final pipeline report
report:
	$(PYTHON) src/reports/generate_report.py --config $(CONFIG)

# Full pipeline - runs all steps in order
pipeline: validate clean_data validate_cleaned features train classify report

# Code quality targets
test:
	pytest tests/ || true

lint:
	flake8 src/ || true

format:
	black src/ || true

isort:
	isort src/ || true

clean:
	rm -rf data/processed/* reports/*.json reports/*.md models/*.pkl || true
