.PHONY: install test experiment matrix dashboard

install:
	pip install -e '.[dev]'

test:
	pytest -q

experiment:
	python -m causalguard.pipelines.run_experiment --config configs/experiments/base.yaml --output experiments/latest

matrix:
	python -m causalguard.pipelines.run_matrix --config configs/experiments/base.yaml --seeds 1,2,3 --output experiments/matrix_summary.csv

dashboard:
	python -m streamlit run dashboard/app.py
