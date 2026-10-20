# CI/CD Documentation - trace-search

This document lists all CI checks, what they block on, how to run them locally, and the exact job names for branch protection.

## Overview

The CI pipeline enforces quality gates and security checks on every PR and push to `main`. All checks must pass before merging.

## Quality Checks

### 1. Lint (`quality / lint`)

**What it checks:** Code style and formatting with ruff.

**Fails on:**
- Style violations (PEP 8)
- Import ordering issues
- Unused imports or variables
- Formatting violations

**Run locally:**
```bash
ruff check src/ tests/ scripts/ --output-format=github
ruff format --check src/ tests/ scripts/
```

**Auto-fix:**
```bash
ruff check --fix src/ tests/ scripts/
ruff format src/ tests/ scripts/
```

---

### 2. Type Checking (`quality / types`)

**What it checks:** Static type checking with mypy in strict mode on `src/`.

**Fails on:**
- Missing type annotations
- Type inconsistencies
- Untyped function definitions
- Any violations of strict mode rules

**Run locally:**
```bash
pip install -r requirements-ci.txt
pip install -e .
mypy src/ --strict --pretty --show-error-codes
```

**Config:** `mypy.ini`

---

### 3. Tests + Coverage (`quality / tests+coverage`)

**What it checks:**
- All unit tests pass
- Overall coverage ≥80%
- Diff coverage on new/changed code ≥80% (PRs only, soft-fail)

**Fails on:**
- Any test failure
- Overall coverage below 80%

**Run locally:**
```bash
# Run tests with coverage
pytest tests/ \
  --cov=src/trace_search \
  --cov-report=xml \
  --cov-report=term \
  --cov-report=html \
  --cov-branch \
  -v

# Check coverage floor
coverage report --fail-under=80

# Check diff coverage (requires comparison branch)
diff-cover coverage.xml \
  --compare-branch=origin/main \
  --fail-under=80
```

**Config:** `.coveragerc`, `pyproject.toml` (pytest section)

---

### 4. Complexity (`quality / complexity`)

**What it checks:**
- Cyclomatic complexity ≤10 (McCabe C90)
- Maintainability index grade B or better
- Code complexity grade C or better

**Fails on:**
- Functions with complexity >10
- Low maintainability index (below B)
- Poor code complexity scores (below C)

**Run locally:**
```bash
# Cyclomatic complexity
ruff check src/ --select C90 --config 'lint.mccabe.max-complexity=10'

# Maintainability index
radon mi src/ -s -n B

# Code complexity
radon cc src/ -s -n C
```

---

### 5. UI Checks (`quality / ui (eslint+tsc+prettier)`)

**What it checks:** (Runs only when `ui/package.json` or `web/package.json` exists)
- ESLint for TypeScript/JavaScript
- TypeScript type checking (`tsc --noEmit`)
- Prettier formatting

**Fails on:**
- Lint violations
- Type errors
- Formatting violations

**Run locally:**
```bash
cd ui/  # or web/
npm run lint
npx tsc --noEmit
npx prettier --check src/
```

**Status:** Currently skipped (no UI package yet)

---

## Security Checks

### 6. Bandit (`security / bandit`)

**What it checks:** Python security vulnerabilities and code smells.

**Fails on:**
- High or critical severity issues
- Common security anti-patterns
- Hardcoded credentials or secrets

**Run locally:**
```bash
pip install bandit[toml]==1.8.0

# Check for high/critical issues
bandit -r src/ scripts/ \
  --severity-level high \
  -ll \
  -f txt
```

**Config:** `.bandit.toml`

---

### 7. Semgrep (`security / semgrep`)

**What it checks:**
- OWASP Top 10 vulnerabilities
- Python security audit rules
- Custom trace-search rules (no face recognition, no cloud APIs, etc.)

**Fails on:**
- ERROR severity findings
- Banned imports (face recognition, cloud APIs)
- Security vulnerabilities

**Run locally:**
```bash
pip install semgrep==1.99.0

semgrep scan \
  --config=auto \
  --config=p/owasp-top-ten \
  --config=p/security-audit \
  --config=p/python \
  --severity=ERROR
```

**Config:** `.semgrep.yml` (custom rules)

---

### 8. CodeQL - Python (`security / codeql (python)`)

**What it checks:** Deep semantic security analysis for Python.

**Fails on:**
- Security vulnerabilities detected by CodeQL
- Code quality issues (security-and-quality query suite)

**Run locally:**
```bash
# Requires GitHub CLI with CodeQL extension
gh codeql database create python-db --language=python
gh codeql database analyze python-db --format=sarif-latest \
  --output=python-results.sarif
```

**Note:** CodeQL runs automatically on this public repo. Results appear in the Security tab > Code scanning alerts.

---

### 9. CodeQL - JavaScript (`security / codeql (javascript)`)

**What it checks:** Deep semantic security analysis for JavaScript/TypeScript.

**Status:** Currently skipped (no UI code yet). Runs automatically when `ui/` or `web/` TypeScript/JavaScript files exist.

---

### 10. Gitleaks (`security / gitleaks`)

**What it checks:** Secret scanning on full git history.

**Fails on:**
- Hardcoded credentials, API keys, tokens
- Private keys or certificates
- Any secret pattern match

**Run locally:**
```bash
# Using Docker
docker run -v $(pwd):/path ghcr.io/gitleaks/gitleaks:latest detect \
  --source="/path" \
  --verbose

# Or install gitleaks locally
gitleaks detect --verbose
```

**Config:** `.gitleaks.toml`

---

### 11. pip-audit (`security / pip-audit`)

**What it checks:** Known vulnerabilities in Python dependencies.

**Fails on:**
- High or critical CVEs in installed packages
- Vulnerabilities from OSV database

**Run locally:**
```bash
pip install pip-audit==2.7.3
pip install -e .
pip install -r requirements.txt
pip install -r requirements-dev.txt
pip install -r requirements-ci.txt

pip-audit --strict --vulnerability-service=osv
```

---

### 12. Dependency Review (`security / dependency-review`)

**What it checks:** (PRs only) License and vulnerability review for dependency changes.

**Fails on:**
- High or critical vulnerabilities in new dependencies
- Banned licenses: AGPL, GPL, SSPL in new dependencies

**Run locally:**
```bash
# Requires GitHub CLI
gh api repos/:owner/:repo/dependency-graph/compare/:basehead
```

**Note:** This check only runs on PRs and uses GitHub's dependency graph.

---

### 13. npm audit (`security / npm-audit`)

**What it checks:** (Runs only when UI package-lock.json exists)
- Known vulnerabilities in npm dependencies
- High or critical severity issues in production dependencies

**Run locally:**
```bash
cd ui/  # or web/
npm audit --audit-level=high --production
```

**Status:** Currently skipped (no UI package yet)

---

## Project Rules Checks

### 14. Project Rules (`rules / project-rules`)

**What it checks:**
- No banned imports (face_recognition, insightface, ultralytics, etc.)
- No banned identifiers (watchlist, enrol, face_embedding, gait, etc.)
- No media files committed outside allowed paths

**Fails on:**
- Banned imports or dependencies (PRD §2 OUT, §3)
- Banned identifiers in code (watchlist, face recognition, gait)
- Video/image/weight files outside `fixtures/`, `tests/fixtures/`, `docs/diagrams/`

**Run locally:**
```bash
python scripts/check_project_rules.py
```

**Allow-list for false positives:** `.project-rules-allow`

**Config:** `scripts/check_project_rules.py`

---

### 15. License Fence (`rules / licence-fence`)

**What it checks:**
- All dependencies in license manifest
- No AGPL/GPL/NC/unknown licenses
- FFmpeg is LGPL-only build
- Manifest schema validation

**Fails on:**
- Missing manifest entries
- Banned licenses (AGPL, GPL, SSPL, BUSL, NC)
- Invalid manifest schema
- Banned components (Ultralytics, Market-1501 datasets, etc.)

**Run locally:**
```bash
python scripts/check_licenses.py \
  --requirements=requirements-dev.txt \
  --report=license-report.json
```

**Note:** This check is part of the existing CI from PR #1. Do not duplicate it.

---

## Automated Review

### 16. PR Summary (`review / pr-summary`)

**What it does:**
- Waits for all other checks to complete
- Aggregates results into a single sticky PR comment
- Updates the check summary

**Output:**
- ✅/❌ status for each check
- Summary: X passing, Y failing, Z pending
- Automatically updates on each push

**Note:** This is an informational check and does not block merges by itself.

---

## Running All Checks Locally

### Quick check script

```bash
#!/bin/bash
set -e

echo "🔍 Running all CI checks locally..."

# Quality
echo "→ Lint..."
ruff check src/ tests/ scripts/
ruff format --check src/ tests/ scripts/

echo "→ Types..."
mypy src/ --strict

echo "→ Tests + Coverage..."
pytest tests/ --cov=src/trace_search --cov-report=term --cov-branch -v
coverage report --fail-under=80

echo "→ Complexity..."
ruff check src/ --select C90 --config 'lint.mccabe.max-complexity=10'
radon mi src/ -s -n B
radon cc src/ -s -n C

# Security
echo "→ Bandit..."
bandit -r src/ scripts/ --severity-level high -ll

echo "→ Semgrep..."
semgrep scan --config=auto --config=p/owasp-top-ten --config=p/python --severity=ERROR

echo "→ pip-audit..."
pip-audit --strict

echo "→ Gitleaks..."
gitleaks detect

# Project rules
echo "→ Project rules..."
python scripts/check_project_rules.py

echo "→ License fence..."
python scripts/check_licenses.py --requirements=requirements-dev.txt

echo "✅ All checks passed!"
```

Save as `scripts/check_all.sh` and run:
```bash
chmod +x scripts/check_all.sh
./scripts/check_all.sh
```

Or use the Makefile:
```bash
make check
```

---

## Branch Protection Settings

For the `main` branch, configure the following in GitHub Settings > Branches > Branch protection rules:

### Required Status Checks

Mark these job names as required:

1. `quality / lint`
2. `quality / types`
3. `quality / tests+coverage`
4. `quality / complexity`
5. `security / bandit`
6. `security / semgrep`
7. `security / codeql (python)`
8. `security / gitleaks`
9. `security / pip-audit`
10. `security / dependency-review` (PRs only)
11. `rules / project-rules`
12. `rules / licence-fence` (from existing CI)

**Note:** `quality / ui (eslint+tsc+prettier)`, `security / codeql (javascript)`, and `security / npm-audit` are skipped when no UI package exists. Do not mark them as required until the UI is added.

### Other Protection Rules

- ✅ Require a pull request before merging
- ✅ Require approvals: **1**
- ✅ Dismiss stale pull request approvals when new commits are pushed
- ✅ Require review from Code Owners
- ✅ Require conversation resolution before merging
- ✅ Require status checks to pass before merging
- ✅ Require branches to be up to date before merging
- ✅ Do not allow bypassing the above settings (include administrators)
- ✅ Restrict who can push to matching branches (no direct pushes)
- ✅ Block force pushes
- ✅ Do not allow deletions

---

## Repository Settings

### Code Security and Analysis

Enable in Settings > Security > Code security and analysis:

#### 1. CodeQL Analysis

- **Option 1 (Recommended):** Advanced setup via `.github/workflows/codeql.yml` (already configured)
- **Option 2:** Default setup (GitHub manages the workflow)

We use **Advanced setup** to control query suites, languages, and schedule.

#### 2. Secret Scanning

- ✅ **Secret scanning:** Already enabled (public repo)
- ✅ **Push protection:** Already enabled (blocks commits with secrets)

#### 3. Dependabot

- ✅ **Dependabot alerts:** Enable (currently off)
- ✅ **Dependabot security updates:** Enable (currently off)

Dependabot configuration is in `.github/dependabot.yml`:
- Python dependencies: weekly on Monday
- GitHub Actions: weekly on Monday
- npm dependencies: weekly on Monday (when UI exists)

---

## Troubleshooting

### Check Failures

**Lint failures:**
- Run `ruff check --fix src/ tests/ scripts/`
- Run `ruff format src/ tests/ scripts/`

**Type errors:**
- Add missing type annotations
- Use `# type: ignore[error-code]` sparingly with justification

**Test failures:**
- Fix the failing test
- If coverage drops, add tests for new code

**Security findings:**
- Review Bandit/Semgrep/CodeQL reports
- Fix the vulnerability or suppress with justification
- Never ignore high/critical security findings

**Project rules violations:**
- Remove banned imports/dependencies
- If false positive, add to `.project-rules-allow` with review comment

**License fence failures:**
- Add missing dependencies to `licenses/manifest.json`
- Replace banned dependencies (AGPL/GPL/NC)
- Run `python scripts/check_licenses.py --update` to refresh hashes

### CI Performance

All jobs run in parallel where possible:
- Quality checks: ~5-10 minutes total
- Security checks: ~10-15 minutes total
- CodeQL: ~15-20 minutes (runs in parallel)

Total CI time: ~15-20 minutes for a typical PR.

---

## LLM Code Review (Optional)

### Option 1: Cursor Bugbot (Recommended)

Enable the Cursor Bugbot GitHub App for AI-powered code review:

1. Install the app: https://cursor.com/bugbot
2. Configure in repo settings
3. Bugbot reads `.cursor/BUGBOT.md` for project-specific rules
4. Automatic reviews on every PR

**What Bugbot checks:**
- PRD hard-stops (no face recognition, no AGPL, no cloud APIs)
- Code quality and best practices
- Security vulnerabilities
- Alignment with project constraints

**Cost:** Included with Cursor subscription.

### Option 2: Self-Hosted LLM Review

Add a workflow job with an LLM API:

**Requirements:**
- Model API key (OpenAI, Anthropic, etc.)
- Store as repo secret: `LLM_API_KEY`
- Add workflow: `.github/workflows/llm-review.yml`

**Example workflow snippet:**

```yaml
name: LLM Review

on:
  pull_request:
    types: [opened, synchronize, reopened]

jobs:
  review:
    name: LLM Code Review
    runs-on: ubuntu-latest
    permissions:
      contents: read
      pull-requests: write
    
    steps:
    - uses: actions/checkout@v4
    - name: Run LLM review
      env:
        OPENAI_API_KEY: ${{ secrets.LLM_API_KEY }}
      run: |
        # Call LLM API with diff and BUGBOT.md context
        # Post review as PR comment
```

**Note:** Not currently implemented. Requires a repo secret (`LLM_API_KEY`) to be added.

---

## Quick Reference

| Check | Job Name | Blocks On | Local Command |
|-------|----------|-----------|---------------|
| Lint | `quality / lint` | Style violations, formatting | `ruff check src/ && ruff format --check src/` |
| Types | `quality / types` | Type errors, missing annotations | `mypy src/ --strict` |
| Tests | `quality / tests+coverage` | Test failures, coverage <80% | `pytest --cov=src/trace_search --cov-report=term` |
| Complexity | `quality / complexity` | High complexity, poor maintainability | `ruff check src/ --select C90` |
| Bandit | `security / bandit` | High/critical security issues | `bandit -r src/ --severity-level high` |
| Semgrep | `security / semgrep` | OWASP Top 10, banned patterns | `semgrep scan --config=p/owasp-top-ten` |
| CodeQL | `security / codeql (python)` | Security vulnerabilities | `gh codeql database analyze ...` |
| Gitleaks | `security / gitleaks` | Hardcoded secrets | `gitleaks detect` |
| pip-audit | `security / pip-audit` | High/critical CVEs | `pip-audit --strict` |
| Dep Review | `security / dependency-review` | High CVEs, banned licenses (PRs) | `gh api .../dependency-graph/compare/...` |
| Project Rules | `rules / project-rules` | Banned imports, media files | `python scripts/check_project_rules.py` |
| License Fence | `rules / licence-fence` | AGPL/GPL/NC, missing manifest | `python scripts/check_licenses.py` |

---

**Last Updated:** Tue 20 Oct 2026 (IST)  
**Owner:** Shitanshu Sharma  
**Source:** PRD Cut 01 rev 2, CI/CD implementation (T1)
