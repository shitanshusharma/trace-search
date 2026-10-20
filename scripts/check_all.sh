#!/bin/bash
# Run all CI checks locally
# Usage: ./scripts/check_all.sh

set -e

echo "🔍 Running all CI checks locally..."
echo ""

# Quality checks
echo "════════════════════════════════════════════════════════════════"
echo "QUALITY CHECKS"
echo "════════════════════════════════════════════════════════════════"
echo ""

echo "→ Lint (ruff)..."
ruff check src/ tests/ scripts/ --output-format=github
ruff format --check src/ tests/ scripts/
echo "✅ Lint passed"
echo ""

echo "→ Type checking (mypy)..."
mypy src/ --strict --pretty --show-error-codes
echo "✅ Type checking passed"
echo ""

echo "→ Tests + Coverage..."
pytest tests/ \
  --cov=src/trace_search \
  --cov-report=term \
  --cov-branch \
  -v
coverage report --fail-under=80
echo "✅ Tests passed, coverage ≥80%"
echo ""

echo "→ Complexity..."
ruff check src/ --select C90 --config 'lint.mccabe.max-complexity=10'
radon mi src/ -s -n B
radon cc src/ -s -n C
echo "✅ Complexity checks passed"
echo ""

# Security checks
echo "════════════════════════════════════════════════════════════════"
echo "SECURITY CHECKS"
echo "════════════════════════════════════════════════════════════════"
echo ""

echo "→ Bandit..."
bandit -r src/ scripts/ --severity-level high -ll -f txt
echo "✅ Bandit passed"
echo ""

echo "→ Semgrep..."
semgrep scan \
  --config=auto \
  --config=p/owasp-top-ten \
  --config=p/security-audit \
  --config=p/python \
  --severity=ERROR
echo "✅ Semgrep passed"
echo ""

echo "→ pip-audit..."
pip-audit --strict --vulnerability-service=osv
echo "✅ pip-audit passed"
echo ""

echo "→ Gitleaks..."
if command -v gitleaks > /dev/null; then
    gitleaks detect --verbose
    echo "✅ Gitleaks passed"
else
    echo "⚠️  Gitleaks not installed, skipping (install: https://github.com/gitleaks/gitleaks#installing)"
fi
echo ""

# Project rules
echo "════════════════════════════════════════════════════════════════"
echo "PROJECT RULES CHECKS"
echo "════════════════════════════════════════════════════════════════"
echo ""

echo "→ Project rules..."
python scripts/check_project_rules.py
echo "✅ Project rules passed"
echo ""

echo "→ License fence..."
python scripts/check_licenses.py --requirements=requirements-dev.txt --report=license-report.json
echo "✅ License fence passed"
echo ""

echo "════════════════════════════════════════════════════════════════"
echo "✅ ALL CHECKS PASSED!"
echo "════════════════════════════════════════════════════════════════"
