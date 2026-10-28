# CI/CD Documentation - trace-search

This document lists all CI checks, what they block on, how to run them locally, and the exact job names for branch protection.

## Overview

The CI pipeline enforces quality gates and security checks on every PR and push to `main`. All checks must pass before merging.

Two hashed locks stay apart:

- `requirements-dev.lock` is the app environment. The licence fence job installs only this file. Its package versions match `licenses/manifest.json`, including `jsonschema==4.26.0`.
- `requirements-ci.lock` is CI-only (semgrep, mypy, bandit, pip-audit, radon, diff-cover, and their dependencies). Install it with `--require-hashes --no-deps` in its own virtualenv. Do not install it into the app fence environment, and do not uninstall packages from either environment. These packages are out of scope for AC-01: they are not added to `licenses/manifest.json`. The same job then runs `scripts/check_ci_licenses.py` on that virtualenv's site-packages (`--path`). Trove classifiers and short licence headers are mapped to SPDX, then judged with the app fence's banned-licence parser. AGPL and GPL fail. Unrecognised text fails closed. It does not add an allowlist. `semgrep` (`LGPL-2.1-or-later`) and `certifi` (`MPL-2.0`) are not banned terms. `face` 26.0.1 and `peewee` 3.19.0 ship no licence metadata; the check warns for those two exact versions after the upstream LICENSE files were read (BSD-3-Clause and MIT) and fails closed on any other unknown. `requirements-ci-licenses.txt` is an informational pip-licenses dump, not a manifest.

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
python3 -m venv .venv-ci
.venv-ci/bin/python -m pip install --require-hashes --no-deps -r requirements-ci.lock
pip install -e .
.venv-ci/bin/mypy src/ --strict --pretty --show-error-codes
```

**Config:** `mypy.ini`

---

### 3. Tests + Coverage (`quality / tests+coverage`)

**What it checks:**
- All unit tests pass
- Overall coverage ≥80% of `src/trace_search` (the package, including `ingest`)
- Diff coverage on new/changed code ≥80% (PRs only, soft-fail; the step does not use `--show-missing`)

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
python3 -m venv .venv-ci
.venv-ci/bin/python -m pip install --require-hashes --no-deps -r requirements-ci.lock

.venv-ci/bin/semgrep scan \
  --error \
  --metrics=off \
  --config=.semgrep.yml \
  --config=p/owasp-top-ten \
  --config=p/security-audit \
  --config=p/python \
  --exclude tests/semgrep \
  --severity=ERROR

.venv-ci/bin/semgrep --test --metrics=off --config .semgrep.yml tests/semgrep/rules.py
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

**What it checks:** Known vulnerabilities in two places that are never mixed into the licence-fence environment.

- `requirements-ci.lock`, installed with `--no-deps` in this job. That lock pins `protobuf==5.29.6`, `mcp==1.30.0`, and `setuptools==84.2.1`. `--no-deps` is required because semgrep 1.182.0 declares `mcp==1.29.0` and pip will not resolve the fixed pin. `pip-audit -r requirements-ci.lock` hits that same resolver conflict, so this job audits the installed CI-tools environment instead.
- `requirements-dev.lock`, the app lock, via `pip-audit -r requirements-dev.lock`. Its versions match the licence manifest, including `jsonschema==4.26.0`.
- The installed CI-tools audit ignores only GHSA-xcq7-p3pf-p88h through 2026-11-21 (B-79). The dev-lock audit does not.

**Fails on:**
- High or critical CVEs in installed packages
- Vulnerabilities from OSV database

**Run locally:**
```bash
python3 -m venv .venv-ci
.venv-ci/bin/python -m pip install --require-hashes --no-deps -r requirements-ci.lock
.venv-ci/bin/pip-audit --strict --vulnerability-service=osv
.venv-ci/bin/pip-audit -r requirements-dev.lock --strict --vulnerability-service=osv
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
- No media file unless `entries[].path` and its sha256 match that file

**Fails on:**
- Banned imports or dependencies (PRD §2 OUT, §3)
- Banned identifiers in code (watchlist, face recognition, gait)
- A media file whose path is missing from `entries[]`, whose sha256 does not match, or a listed path whose file is missing

**Run locally:**
```bash
python scripts/check_project_rules.py
```

**Allow-list for false positives:** `.project-rules-allow`

**Config:** `scripts/check_project_rules.py`

---

### Action SHA pins (`rules / validate-action-shas`)

**What it checks:** Every `uses:` in `.github/workflows` is `./`, `docker://`, or `owner/repo@<40-hex> # vX.Y.Z`. Each pin is the peeled upstream tag commit. The commits API returns 200 and compare against that tag is `identical`. One action, `owner/repo` with any sub-path ignored, must use a single SHA across those workflows. `github/codeql-action/init` and `github/codeql-action/analyze` count as one action.

**Fails on:**
- A SHA with no version comment, a comment without `v`, or a mutable tag
- A pin that is not the peeled commit for the named tag
- The same owner/repo pinned to more than one SHA. The output lists every mismatched `file:line` and SHA.

**How to fix a mixed SHA:** Pick one pin for that action, usually the peeled commit for the tag you want, and set every `uses:` of that `owner/repo` to that same SHA and its `# vX.Y.Z` comment. Different actions may keep different SHAs. Re-run the command below.

**Run locally:**
```bash
GITHUB_TOKEN="$(gh auth token)" python scripts/validate_action_shas.py
```

---

### 15. License Fence (`Lint, Test, and License Check`)

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

The same job then checks CI-only tools. That step installs `requirements-ci.lock` with `--require-hashes --no-deps` into `.venv-ci-fence` and scans only that environment's site-packages:

```bash
python3 -m venv .venv-ci-fence
.venv-ci-fence/bin/python -m pip install --require-hashes --no-deps -r requirements-ci.lock
site="$(.venv-ci-fence/bin/python -c 'import sysconfig; print(sysconfig.get_path("purelib"))')"
PYTHONPATH=src .venv-ci-fence/bin/python -m scripts.check_ci_licenses --path "$site"
```

`scripts/check_ci_licenses.py` reads each installed package's `License-Expression`, every licence classifier, and a one-line `License` header. The header is checked even when classifiers exist, so `OSI Approved` plus `License: AGPL-3.0` fails. Classifier tails and shorthands are mapped to SPDX, then passed to `is_license_banned` in `scripts/check_licenses.py`. `GNU General Public License v3 (GPLv3)`, `GNU Affero General Public License v3 or later (AGPLv3+)`, and bare `GPLv3` fail. `GNU Lesser General Public License v2 or later (LGPLv2+)` stays allowed. `ISC License (ISCL)` maps to ISC. `The Unlicense (Unlicense)` and `Public Domain` map to Unlicense. A classifier that is only `OSI Approved` is not a licence and fails closed. Any other free text that does not map to SPDX fails closed. Missing or `UNKNOWN` metadata fails closed, except the exact name and version `face==26.0.1` (BSD-3-Clause, https://github.com/mahmoud/face/blob/v26.0.1/LICENSE) and `peewee==3.19.0` (MIT, https://github.com/coleifer/peewee/blob/3.19.0/LICENSE). Those two rows are warnings, and only when metadata is missing. `PYTHONPATH` is only so the checker can import the package; it is not the scan scope. No allowlist entry was added for LGPL or MPL.

**E2E demo step.** The same job, `Lint, Test, and License Check`, runs one more step: `E2E demo (synthetic, CODE_TEST)`. It is after `Run unit tests` and before `Worktree clean after pytest`. The run line is:

```bash
python scripts/run_e2e_demo.py --workdir "$RUNNER_TEMP/e2e"
```

`$RUNNER_TEMP/e2e` is outside the checkout. The script refuses a workdir inside the repository, an ancestor of the repository, the home directory, or an ancestor of home. An existing workdir must be empty. The step uses the same shell and working directory as the other `run` steps in this job.

The default schedule is seed 0. A passing run prints these lines and exits 0. `run time` and `run_id` follow them and change every run:

```
CODE_TEST: not evidence for any AC (AC-07 is DEFERRED)
F1: 3/3
F2: 3/3
F3: 3/3
F4: 3/3
segment_recall=1.0000 (>= 0.80) PASS
segment_precision=1.0000 (>= 0.90) PASS
false_merge_rate=0.0000 (<= 0.05) PASS
```

The bars are fixed in `src/trace_search/pipeline/e2e.py` (`SEGMENT_RECALL_MIN` 0.80, `SEGMENT_PRECISION_MIN` 0.90, `FALSE_MERGE_MAX` 0.05). They are not fitted to a run. The printer is `{name}={value:.4f} ({op} {limit:.2f}) PASS`. A miss prints `FAIL` and the process exits 1.

- **Segment recall** at least 0.80. `segment_recall` is matched truth segments divided by the truth-segment count (`trace_search.matching.route_metrics.segment_recall`). A predicted segment matches a truth segment when the ownership map gives the predicted track that figure, the camera ids are equal, and the gap between the two intervals is at most 10.0 seconds. The gap of 10.0 matches. The gap of 10.1 does not. Matching is one-to-one. An empty truth count raises `ValueError`. It is not reported as 0.0 or 1.0. The passing line on the default schedule is `segment_recall=1.0000 (>= 0.80) PASS`.
- **Segment precision** at least 0.90. `segment_precision` is matched predicted segments divided by the predicted-segment count. The match rule is the same one-to-one rule as recall. An empty predicted count raises `ValueError`. The passing line on the default schedule is `segment_precision=1.0000 (>= 0.90) PASS`.
- **False-merge rate** at most 0.05. `false_merge_rate` is the number of cross-camera links whose tracks belong to different figures, divided by all cross-camera links in the routes. A cross-camera link is two consecutive segments in a route with different `camera_id`s. Same-camera consecutive pairs are not in the denominator. The seed is not an extra link. No cross-camera link raises `ValueError` rather than returning 0.0. The passing line on the default schedule is `false_merge_rate=0.0000 (<= 0.05) PASS`.

These numbers are not evidence for any acceptance criterion. AC-07 is DEFERRED.

**Note:** This check is part of the existing `.github/workflows/ci.yml` from PR #1. The job name is `Lint, Test, and License Check` and includes linting, tests, the synthetic end-to-end demo, the app licence fence, and the CI-tools banned-licence check. No extra required check name is added.

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
bandit -r src/ scripts/ --severity-level high

echo "→ Semgrep..."
semgrep scan --error --metrics=off --config=.semgrep.yml --config=p/owasp-top-ten --config=p/python --severity=ERROR

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
10. `security / dependency-review`
11. `rules / project-rules`
12. `rules / validate-action-shas`
13. `Lint, Test, and License Check`

**Note:** `quality / ui (eslint+tsc+prettier)`, `security / codeql (javascript)`, and `security / npm-audit` are skipped when no UI package exists. Do not mark them as required until the UI is added.

### Other Protection Rules

Agreed setup for `main` (Option A). There is no CODEOWNERS file. Code-owner review stays off, because a sole author cannot approve their own pull request and the existing ruleset would otherwise block later PRs.

- Require a pull request before merging
- Required approvals: **0**
- Code-owner review: **off**
- Require status checks to pass before merging
- Require branches to be up to date before merging
- Require linear history
- Block force pushes
- Block deletions

Required status checks, exact job names:

1. `quality / lint`
2. `quality / types`
3. `quality / tests+coverage`
4. `quality / complexity`
5. `security / bandit`
6. `security / semgrep`
7. `security / codeql (python)`
8. `security / gitleaks`
9. `security / pip-audit`
10. `security / dependency-review`
11. `rules / project-rules`
12. `rules / validate-action-shas`
13. `Lint, Test, and License Check`

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
- Add missing app dependencies to `licenses/manifest.json`
- Replace banned dependencies (AGPL/GPL/NC)
- For `requirements-ci.lock`, fix the package or its metadata. Do not add a manifest row for a CI-only tool. Unrecognised licence text fails closed. Missing metadata fails unless that exact name and version is already in the reviewed override list in `scripts/check_ci_licenses.py`.

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
| Action pins | `rules / validate-action-shas` | Unpinned, unverified, or mixed SHAs for one action | `python scripts/validate_action_shas.py` |
| License Fence | `Lint, Test, and License Check` | AGPL/GPL/NC, missing manifest | `python scripts/check_licenses.py` |
| E2E demo | `Lint, Test, and License Check` | Segment recall below 0.80, segment precision below 0.90, or false-merge rate above 0.05 | `python scripts/run_e2e_demo.py` |

---

**Last Updated:** Wed 28 Oct 2026  
**Owner:** Shitanshu Sharma  
**Source:** PRD Cut 01 rev 2, CI/CD implementation (T1)
