.PHONY: setup data train report test all clean lint

PYTHON = .venv/bin/python
PYTEST = .venv/bin/pytest

setup:
	@echo "Setting up Python virtual environment and dependencies..."
	uv venv --python 3.11 .venv
	uv pip install --python .venv -e ".[dev]"

data:
	@echo "Running data ingestion and feature engineering..."
	$(PYTHON) -c "from dappf.data.ingest import IngestionPipeline; IngestionPipeline().run_all()"
	$(PYTHON) -c "from dappf.features.build import FeatureBuilder; FeatureBuilder().build_all()"

test:
	@echo "Executing test suite (including strict gate-closure leakage test)..."
	$(PYTEST) tests/ -v

train:
	@echo "Running walk-forward model training and backtesting..."
	$(PYTHON) scripts/run_pipeline.py

report:
	@echo "Generating final executive HTML and PDF reports..."
	$(PYTHON) scripts/make_report.py

all: test train report
	@echo "End-to-end pipeline successfully executed!"

lint:
	@echo "Running linting and static checks..."
	$(PYTHON) -m ruff check src/ tests/ scripts/ || true

clean:
	rm -rf data/processed/*.parquet artifacts/models/*.pkl reports/*.html reports/*.pdf reports/figures/*.png
