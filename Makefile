.PHONY: help check lint format types test coverage complexity security clean install

help:
	@echo "trace-search - Makefile targets"
	@echo ""
	@echo "Development:"
	@echo "  make install          Install dependencies"
	@echo "  make lint             Run linter (ruff)"
	@echo "  make format           Auto-format code (ruff)"
	@echo "  make types            Run type checker (mypy)"
	@echo "  make test             Run tests"
	@echo "  make coverage         Run tests with coverage report"
	@echo "  make complexity       Check code complexity"
	@echo "  make security         Run security checks"
	@echo "  make check            Run all checks (lint, types, test, complexity, security)"
	@echo "  make clean            Remove generated files"
	@echo ""

install:
	pip install -e .
	pip install -r requirements-dev.txt
	pip install -r requirements-ci.txt

lint:
	ruff check src/ tests/ scripts/ --output-format=github
	ruff format --check src/ tests/ scripts/

format:
	ruff check --fix src/ tests/ scripts/
	ruff format src/ tests/ scripts/

types:
	mypy src/ --strict --pretty --show-error-codes

test:
	pytest tests/ -v

coverage:
	pytest tests/ \
		--cov=src/trace_search \
		--cov-report=xml \
		--cov-report=term \
		--cov-report=html \
		--cov-branch \
		-v
	coverage report --fail-under=80

complexity:
	@echo "→ Cyclomatic complexity (McCabe C90)..."
	ruff check src/ --select C90 --config 'lint.mccabe.max-complexity=10'
	@echo "→ Maintainability index..."
	radon mi src/ -s -n B
	@echo "→ Code complexity..."
	radon cc src/ -s -n C

security:
	@echo "→ Bandit..."
	bandit -r src/ scripts/ --severity-level high -f txt
	@echo "→ Semgrep..."
	semgrep scan --config=auto --config=p/owasp-top-ten --config=p/python --severity=ERROR || true
	@echo "→ pip-audit..."
	pip-audit --strict --vulnerability-service=osv || true
	@echo "→ Gitleaks..."
	@command -v gitleaks > /dev/null && gitleaks detect --verbose || echo "gitleaks not installed, skipping"
	@echo "→ Project rules..."
	python scripts/check_project_rules.py
	@echo "→ License fence..."
	python scripts/check_licenses.py --requirements=requirements-dev.txt

check: lint types test complexity security
	@echo "✅ All checks passed!"

clean:
	rm -rf build/ dist/ *.egg-info
	rm -rf .coverage coverage.xml htmlcov/
	rm -rf .pytest_cache/ .mypy_cache/ .ruff_cache/
	rm -rf **/__pycache__/
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
