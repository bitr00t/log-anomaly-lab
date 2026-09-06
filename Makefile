.PHONY: install qdrant test data detect clean

install:            ## Install dependencies + package (editable)
	pip install -r requirements.txt && pip install -e .

qdrant:             ## Start a local Qdrant instance
	docker compose up -d

test:               ## Run unit tests (no Qdrant / torch required)
	pytest -v

data:               ## Generate the labelled synthetic log dataset
	python scripts/generate_logs.py 2000 0.03 42

detect:             ## Run the full pipeline (needs Qdrant + data)
	python scripts/run_detection.py

clean:
	rm -rf results .pytest_cache
