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
.venv-ci/bin/mypy src/ --strict --pretty --show-error-codes
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
.venv-ci/bin/radon mi src/ -s -n B
.venv-ci/bin/radon cc src/ -s -n C
echo "✅ Complexity checks passed"
echo ""

# Security checks
echo "════════════════════════════════════════════════════════════════"
echo "SECURITY CHECKS"
echo "════════════════════════════════════════════════════════════════"
echo ""

echo "→ Bandit..."
.venv-ci/bin/bandit -r src/ scripts/ --severity-level high -f txt
echo "✅ Bandit passed"
echo ""

echo "→ Semgrep..."
.venv-ci/bin/semgrep scan \
  --config=auto \
  --config=p/owasp-top-ten \
  --config=p/security-audit \
  --config=p/python \
  --severity=ERROR
echo "✅ Semgrep passed"
echo ""

echo "→ pip-audit..."
.venv-ci/bin/python scripts/check_pip_audit_ignore_expiry.py
# expires 2026-11-21; affects opentelemetry-api 1.37.0 in semgrep's CI-tools env only; blocked on semgrep pinning opentelemetry-api~=1.37.0 (latest 1.182.1); remove when semgrep allows >=1.45.1; approved by Shitanshu 22 Oct 2026; B-79
ci_ignore=(--ignore-vuln GHSA-xcq7-p3pf-p88h)
.venv-ci/bin/pip-audit --strict --vulnerability-service=osv "${ci_ignore[@]}"
.venv-ci/bin/pip-audit -r requirements-dev.lock --strict --vulnerability-service=osv
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

echo "→ License fence (app environment; CI tools stay in .venv-ci)..."
python scripts/check_licenses.py --requirements=requirements.txt --requirements=requirements-dev.txt --report=license-report.json
echo "✅ License fence passed"
echo ""

echo "→ CI tools banned licences (.venv-ci site-packages; no prune)..."
site="$(.venv-ci/bin/python -c 'import sysconfig; print(sysconfig.get_path("purelib"))')"
PYTHONPATH=src .venv-ci/bin/python -m scripts.check_ci_licenses --path "$site"
echo "✅ CI tools banned-licence check passed"
echo ""

echo "════════════════════════════════════════════════════════════════"
echo "✅ ALL CHECKS PASSED!"
echo "════════════════════════════════════════════════════════════════"
