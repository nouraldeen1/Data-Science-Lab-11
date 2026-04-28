"""Benchmark multiple classifiers and select the best one."""

import argparse
import json
import pickle
from pathlib import Path

import pandas as pd
import toml
from pydantic import BaseModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


class DataConfig(BaseModel):
	featured_data_path: str
	target_column: str
	test_size: float
	random_state: int


class ModelConfig(BaseModel):
	model_output_path: str


class ReportsConfig(BaseModel):
	classification_metrics_path: str


class AppConfig(BaseModel):
	data: DataConfig
	model: ModelConfig
	reports: ReportsConfig


def load_config(filepath: str) -> AppConfig:
	"""Load and validate project config."""
	raw_config = toml.load(filepath)
	try:
		return AppConfig.model_validate(raw_config)
	except AttributeError:
		return AppConfig.parse_obj(raw_config)


def evaluate_models(df: pd.DataFrame, config: AppConfig) -> tuple[dict, object]:
	"""Train and compare several classifiers."""
	X = df.drop(columns=[config.data.target_column])
	y = df[config.data.target_column]
	X_train, X_test, y_train, y_test = train_test_split(
		X,
		y,
		test_size=config.data.test_size,
		random_state=config.data.random_state,
		stratify=y,
	)

	models = {
		"logistic_regression": LogisticRegression(max_iter=300),
		"knn": KNeighborsClassifier(n_neighbors=7),
		"svm_rbf": SVC(kernel="rbf", probability=False),
		"random_forest": RandomForestClassifier(
			n_estimators=200, random_state=config.data.random_state
		),
	}

	scores = {}
	best_name = ""
	best_score = -1.0
	best_model = None

	for name, model in models.items():
		model.fit(X_train, y_train)
		preds = model.predict(X_test)
		metrics = {
			"accuracy": round(accuracy_score(y_test, preds), 4),
			"f1_score": round(f1_score(y_test, preds, zero_division=0), 4),
		}
		scores[name] = metrics
		if metrics["f1_score"] > best_score:
			best_score = metrics["f1_score"]
			best_name = name
			best_model = model

	payload = {
		"models": scores,
		"best_model": best_name,
		"best_f1_score": best_score,
		"test_size": config.data.test_size,
	}
	return payload, best_model


def save_json(payload: dict, filepath: str) -> None:
	"""Save classification metrics."""
	Path(filepath).parent.mkdir(parents=True, exist_ok=True)
	with open(filepath, "w", encoding="utf-8") as f:
		json.dump(payload, f, indent=2)
	print(f"Classification report saved -> {filepath}")


def save_model(model, filepath: str) -> None:
	"""Persist the best classifier."""
	Path(filepath).parent.mkdir(parents=True, exist_ok=True)
	with open(filepath, "wb") as f:
		pickle.dump(model, f)
	print(f"Best model saved -> {filepath}")


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--config", default="configs/config.toml")
	args = parser.parse_args()

	app_config = load_config(args.config)
	features_df = pd.read_csv(app_config.data.featured_data_path)
	metrics_payload, winner_model = evaluate_models(features_df, app_config)
	save_json(metrics_payload, app_config.reports.classification_metrics_path)
	save_model(winner_model, app_config.model.model_output_path)
