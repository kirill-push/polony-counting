.DEFAULT_GOAL := help
.PHONY: help
help:  ## Display this help
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

##@ Formatters

.PHONY: format-black
format-black: ## run black (code formatter)
	@black .

.PHONY: format-isort
format-isort: ## run isort (imports formatter)
	@isort .

.PHONY: format
format: format-black format-isort ## run all formatters

##@ Linters

.PHONY: lint-black
lint_black: ## run black in linting mode
	@black . --check

.PHONY: lint-isort
lint-isort: ## run isort in linting mode
	@isort . --check

.PHONY: lint-flake8
lint-flake8: ## run flake8 (code linter)
	@flake8 .

.PHONY: lint-mypy
lint-mypy: ## run mypy (static-type checker)
	@mypy --config-file pyproject.toml .

.PHONY: lint-mypy-report
lint-mypy-report: # run mypy & create report
	@mypy --config-file pyproject.toml . --html-report ./mypy_html

lint: lint-black lint-isort lint-flake8 ## run all linters without mypy

##@ Testing

.PHONY: unit-tests
unit-tests: ## run unit-tests
	@pytest

.PHONY: unit-tests-cov
unit-tests-cov: ## run unit-tests with coverage
	@pytest --cov=src --cov-report term-missing --cov-report=html

.PHONY: unit-tests-cov-fail
unit-tests-cov-fail: ## run unit-tests with coverage and cov-fail level
	@pytest --cov=src --cov-report term-missing --cov-report=html --cov-fail-under=50 --junitxml=pytest.xml | tee pytest-coverage.txt

##@ Documentation
docs-build: ## build documentation locally
   @mkdocs build

docs-deploy: ## build & deploy documentation to "gh-pages" branch
   @mkdocs gh-deploy -m "docs: update documentation" -v --force

##@ Clean-up

clean-cov: ## run cleaning from reports
	@rm -rf .coverage
	@rm -rf htmlcov
	@rm -rf pytest.xml
	@rm -rf pytest-coverage.txt

clean-docs: ## remove output files from mkdocs
   @rm -rf site
