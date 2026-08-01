SHELL := /bin/bash
.DEFAULT_GOAL := help

MLFLOW_URI ?= sqlite:///mlflow_db/malnutrition.sqlite
OPTUNA_URI ?= sqlite:///mlflow_db/optuna.db
RUN_DIR    ?= $(shell cat outputs/last_run.txt 2>/dev/null)
TRAIN_OVERRIDES ?=
EVAL_OVERRIDES  ?=

help: 		 ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

.PHONY: setup curate split train train-quick evaluate mlflow-ui optuna-ui

setup:		 ## Sync the environment from uv.lock
	uv sync

curate:		 ## Build the curated dataset from raw data
	uv run orchestration/curate.py dataset=sample

split:		 ## Produce train/validation/test splits
	uv run orchestration/split.py

train:		 ## Train a model (see TRAIN_OVERRIDES)
	uv run orchestration/train.py $(TRAIN_OVERRIDES)
# example: make train TRAIN_OVERRIDES="model=logistic_regression preprocessor=classic training.n_trials=50 optuna.study_version=3"

train-quick: ## Fast smoke-test training run
	$(MAKE) train TRAIN_OVERRIDES="training=quick" 

evaluate:    ## Evaluate a trained run (RUN_DIR=<path> to pin one)
	@test -n "$(RUN_DIR)" || { echo "No run pointer. Run 'make train' first, or pass RUN_DIR=<path>."; exit 1; }
	@echo "Evaluating: $(RUN_DIR)"
	uv run orchestration/evaluate.py eval.run_dir=$(RUN_DIR) $(EVAL_OVERRIDES)

mlflow-ui:	 ## Launch the MLflow tracking UI
	uv run mlflow ui --backend-store-uri $(MLFLOW_URI) 

optuna-ui:	 ## Launch the Optuna dashboard
	uv run optuna-dashboard $(OPTUNA_URI)