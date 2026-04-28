# TODO: Add common commands for students.
# Suggested targets:
# - setup: install dependencies
# - test: run tests
# - lint: run lint checks
# - clean: remove generated files

.PHONY: help setup test lint clean

setup:
	@echo "TODO: Install dependencies."

pipeline:  # run the entire pipeline 
# example
reports/validation_raw.json: data/raw/Teen_Mental_Health_Dataset.csv src/data/validate.py $(CONFIG)
	$(PYTHON) src/data/validate.py --config $(CONFIG) --input data/raw/Teen_Mental_Health_Dataset.csv --output reports/validation_raw.json

test:
	@echo "TODO: Run tests."

lint:
	@echo "TODO: Run linting/format checks."

clean:
	@echo "TODO: Remove generated files."
