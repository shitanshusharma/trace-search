# trace-search

Anonymous cross-camera route search.

## What trace-search is

trace-search is an on-premises prototype for same-day forensic search. The aim is to follow one anonymous person across cameras, then leave the result for a person to review. This repository is code only.

Words used later in this file:

- **Cut 01** is this first build. It is still in progress.
- **ReID** (re-identification) means matching one anonymous person from one camera to the next by how they look, not by a name.
- **AC-06** is the acceptance check for that match. The harness is deferred. A number from the eval script is not an AC-06 result.
- An **embedder** turns a person image into a list of numbers. **Random init** means those numbers start out random. These embedders are not trained.
- **SMOKE** is a label for a synthetic wiring run. Smoke output is not evidence for any acceptance criterion.
- **T9** is the retention and purge task. Some of its pieces are merged. The rest are not. The status list says which.

Cut 01 is still in progress. End-to-end route search does not run yet. What you can run today is below.

## What it is not

By design, trace-search does not do any of these:

- Face recognition
- Identity
- Gait
- A watchlist
- Gender, age, or ethnicity outputs

It also does not match across days, match across a clothing change, or send real-time alerts. This is investigative tooling with human review, not automated surveillance.

## Install and run it today

Use Python 3.11 or 3.12. Pillow 13 requires 3.11+. The `Lint, Test, and License Check` job in [`.github/workflows/ci.yml`](.github/workflows/ci.yml) uses Python 3.11. `venv/` is gitignored. Run the later commands from the repository root, with that virtualenv active.

From a new machine:

```bash
git clone https://github.com/shitanshusharma/trace-search.git
cd trace-search
python3.11 -m venv venv
source venv/bin/activate
```

Install the way that job does. The install downloads large wheels (OpenCV and onnxruntime).

```bash
python -m pip install --upgrade pip
pip install --require-hashes -r requirements-dev.lock
pip install --require-hashes -r requirements.txt
pip install -e .
```

Tests, same job. The full pytest run takes about 5 minutes.

```bash
pytest tests/ -v --tb=short
```

An HTML coverage report. This shorter command is not what CI runs. It writes `htmlcov/`, which is gitignored. The `quality / tests+coverage` job runs pytest with coverage (xml, term, and html), branch coverage, and `-v`, then fails under 80%.

```bash
pytest --cov=src/trace_search --cov-report=html
```

Lint, same job:

```bash
ruff check src/ tests/ scripts/
```

Synthetic smoke. Generate the clips first. They go in `smoke/`, which is gitignored, and they are not committed. With no flags the generator writes 5 clips, seed 42.

Do not pass `--update-manifest`. The generator prints a hint about that flag. The manifest already lists these clips, and this command does not edit it.

The database path has to sit outside the repo. `/tmp/smoke.sqlite` does.

```bash
python scripts/generate_smoke_fixtures.py
PYTHONPATH=src python scripts/run_smoke.py \
  --manifest licenses/manifest.json \
  --clips-dir smoke \
  --db /tmp/smoke.sqlite \
  --detector scripted-figures \
  --min-clips 5
```

Synthetic ReID eval. Both embedders are random init. The JSON sets `evidence` to false. Its note says the numbers are not an AC-06 result. Do not treat them as evidence.

```bash
python scripts/run_reid_eval.py --embedder placeholder --synthetic-seed 0 --json
python scripts/run_reid_eval.py --embedder osnet-lite --synthetic-seed 0 --json
```

`--embedder osnet-lite` is `OsnetLiteEmbedder` with random init only. The placeholder is `UntrainedPlaceholderEmbedder`. Neither one is trained.

A short terminal demo of these commands is in `docs/demo.cast`. It shows the current pipeline steps, not route search.

```bash
asciinema play docs/demo.cast
```

## Limits

- Smoke clips are synthetic only. This repo has no real footage and no consented footage.
- The embedders are random-init: an untrained placeholder and a random-init OSNet-lite. No training runs. There is no trained ReID model here.
- Scores are not evidence. Smoke output is not evidence for any acceptance criterion. ReID eval output is not an AC-06 result. The AC-06 harness is DEFERRED.
- Cut 01 is still in progress.
- The web UI and the search API are planned. Both packages are stubs.
- End-to-end route search is planned. `matching/` has route, topology, and tracklet code, and nothing serves that as a search yet.
- Retention windows and a footage-derived purge function are in `retention/`. There is no operator UI for them. The status list marks T9 partly done.

A GPU is not required for the commands above. The planned on-prem target is below.

## Technical details

### Project structure

```
trace-search/
├── src/trace_search/       # Main package
│   ├── ingest/             # Video file/RTSP ingest
│   ├── detection/          # Person detection (YOLOX-S on onnxruntime, plus scripted detectors)
│   ├── tracking/           # Single-camera tracking (in-repo ByteTrack)
│   ├── pipeline/           # Per-camera ingest, detect, track, and persist
│   ├── attributes/         # Clothing color + bag detection
│   ├── reid/               # Cross-camera ReID (untrained placeholder and random-init OSNet-lite embedders; no training)
│   ├── matching/           # Route, topology, and tracklets (library code; end-to-end search is planned)
│   ├── api/                # Search and query API (planned; the package is a stub)
│   ├── ui/                 # Web UI (planned: site map, route view, clips; the package is a stub)
│   ├── fixtures/           # Synthetic fixture generation + separation guards (T1)
│   ├── audit/              # Append-only HMAC audit log (T8, AC-13)
│   ├── retention/          # Shipped retention windows and footage-derived purge
│   └── common/             # Shared utilities
├── tests/                  # Unit and integration tests
├── scripts/                # Build and maintenance scripts
│   ├── check_licenses.py   # License fence validator
│   ├── generate_smoke_fixtures.py  # Synthetic video generator (T1)
│   ├── run_smoke.py                # M1a smoke runner (scripted-figures)
│   ├── run_reid_eval.py            # Synthetic ReID eval (placeholder or osnet-lite)
│   ├── check_separation.py         # Fixture separation checker (T1)
│   ├── check_project_rules.py      # Project rules checker
│   └── demo_audit_log.py   # AC-13 JSON proof
├── licenses/               # License manifest and schema
│   ├── manifest.json       # Machine-readable license manifest
│   └── manifest_schema.json
├── smoke/                  # Synthetic smoke test fixtures (generated, not committed)
└── .github/workflows/      # CI/CD
```

### Design targets (not built yet)

These are goals. They are not working features in this repo.

- Indian hospitals and tech parks
- Mixed Hikvision/CP Plus estates (2+ brands, 6+ cameras)
- Video files from NVR native exports (optional RTSP)
- Attribute-based search: clothing color, bags/backpacks

File and RTSP ingest code is in `ingest/`. Live RTSP is a stretch goal. Brand-specific accuracy is not claimed here. `attributes/` has clothing-color and bag code. Nothing serves that as a search yet.

### Licence fence

This project maintains **strict license hygiene** for commercial use:

- **Banned licenses:** AGPL, GPL, NC (Non-Commercial), unknown/unrecognized
- **GCC Runtime Library Exception:** The single GPL-family exception is GCC runtime libraries (`libgfortran`, `libquadmath`, `libgcc_s`, `libstdc++`) with the GCC Runtime Library Exception (`GPL-3.0 WITH GCC-exception-3.1`), which are required runtime components
- **OpenCV bundled files (21 Oct 2026, 7:45 AM IST):** OpenSSL 1.1.1w (`OpenSSL`) on `libcrypto-337dac8b.so.1.1` and `libssl-28bef1ac.so.1.1`, and Intel IPP 2021.11.0 (`LicenseRef-Intel-Simplified-Software-License`) on `cv2/cv2.abi3.so`, only inside opencv-python-headless 4.10.0.84. Counsel confirmation required before release. These are not global allowlist entries
- **Pillow bundled FreeType:** FreeType 2.14.3 (`FTL`) on `pillow.libs/libfreetype-57f805ba.so.6.20.6`, only inside Pillow 13.0.0. This is not a global allowlist entry. The FreeType credit line is in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)
- **Banned components:** Ultralytics (AGPL), boxmot (AGPL), InsightFace/ArcFace, OpenGait
- **Banned datasets:** Market-1501, MSMT17, DukeMTMC, CUHK03, CASIA-B, GREW, Gait3D, Objects365-trained weights

Every dependency, model, weight, dataset, and test clip is tracked in [`licenses/manifest.json`](licenses/manifest.json) with:

- Name, version, license, source URL, SHA-256
- For video clips: repo-relative `path`, license version, date checked (IST), perceptual hashes, usage tag

CI verifies the full installed closure (direct + transitive dependencies) against the manifest and bans GPL/AGPL/SSPL/BUSL/NC/proprietary licenses and problematic components.

Longer fence notes are in [`docs/license-fence.md`](docs/license-fence.md).

The `Lint, Test, and License Check` job runs the fence on the installed environment:

```bash
python scripts/check_licenses.py --requirements=requirements.txt --requirements=requirements-dev.txt --report=license-report.json
```

Licence fence tests live in `tests/test_license_fence.py`. Synthetic fixture tests include `tests/test_generator.py`, `tests/test_guards_comprehensive.py`, `tests/test_separation_comprehensive.py`, `tests/test_t1_integration.py`, and `tests/test_manifest_pins.py`.

The same job then checks CI tools in a separate venv built from `requirements-ci.lock`. That venv is not the app environment. The job does not uninstall packages from either environment.

The `Fixture separation` job checks that SMOKE_ONLY and CODE_TEST fixtures are not in training, eval, or study data. It sets `PYTHONPATH=src` and runs:

```bash
PYTHONPATH=src python scripts/check_separation.py --json
```

The `rules / project-rules` job runs:

```bash
python scripts/check_project_rules.py
```

### Planned on-prem machine

Not required for the commands above. The deployment target is:

- Ubuntu 20.04+ or similar Linux
- GPU: NVIDIA RTX 4070-class (12 GB) or equivalent for production
- CPU: 8+ cores
- RAM: 32 GB

### Smoke runner (M1a)

No workflow step runs `scripts/generate_smoke_fixtures.py` or `scripts/run_smoke.py`. CI calls the runner in-process from `tests/test_smoke_runner.py`.

`scripts/run_smoke.py` imports `trace_search` from `src/`, so the command above sets `PYTHONPATH=src`. It calls `run_cameras` with `derived_from='SMOKE'`. After the arguments are accepted it prints `SMOKE: not evidence for any AC`. That output is not evidence for any acceptance criterion.

`--detector` is `scripted-figures` or `yolox`. The script does not download weights.

CI runs `scripted-figures` on the 5 synthetic clips only, through pytest (`tests/test_smoke_runner.py::test_scripted_figures_on_five_synthetic_clips`). That test generates the clips in a temp directory. CI does not run `yolox` and does not download weights. `scripted-figures` is an OpenCV threshold and contour pass over the synthetic generator's flat-colour figures. It labels every blob that clears the threshold as a person, including the generator's obstructions. The cutoffs are fitted to that generator. It is a SMOKE-wiring stub, not a real detector, and it is not run or tuned on stock clips.

The scripted-figures command is in [Install and run it today](#install-and-run-it-today).

`yolox` is local only. It needs `--weights` or `TRACE_SEARCH_YOLOX_WEIGHTS`, and that file must sit outside the repo. The runner builds `YoloxDetector` with that path. The detector hashes the file and compares it to the `YOLOX-S COCO` row in the repo `licenses/manifest.json` before it builds an onnxruntime `CPUExecutionProvider` session. `--manifest` on this command is the clip manifest, not the weights hash. Weights and stock clips stay outside the repo. The local run uses the same script with `--detector yolox`, plus `--manifest`, `--clips-dir`, `--db`, `--weights`, and `--min-clips`. CI does not run that command. This README does not give a copy-paste line for it, because the paths have to be outside the repo and this checkout has no weights file.

`--db` is required. `TrackStore` refuses a path inside the repo. On a smoke run it creates a missing parent directory and the SQLite file. `--min-clips` defaults to 5 and must be at least 1.

Each clip is checked with `check_fixture_allowed` in the `SMOKE_TEST` context before the pipeline opens it. The runner then identifies the file again. An identified clip is allowed only when its tag is `SMOKE_ONLY`. Any other tag is refused, and so is a missing row. Each invocation prints a new `run_id`. The `tracks` number on each clip line is the distinct track ids from `TrackStore.read` for that camera and `run_id`. `RunStats.tracks` counts observation rows, and the runner does not print that field. A file in the clip directory whose suffix is not `.mp4` is printed as `skipped:` and is not counted. The process exits 0 only when there are 0 errors, at least `--min-clips` clips finished, and each finished clip has at least one `SMOKE` observation for this `run_id`.

`--purge --db PATH` deletes `SMOKE` rows from an existing track database, then prints the deleted count and `SMOKE rows remaining`. It accepts no `--manifest`, `--clips-dir`, `--detector`, `--weights`, or `--min-clips`. It does not create a database and it does not delete other rows.

The T-3-9 local real-weights `yolox` run on 5 Pexels/Pixabay clips (kept outside the repo, not in git) is recorded in [PR #34](https://github.com/shitanshusharma/trace-search/pull/34). SMOKE: not evidence for any AC.

### CI

Pull requests to `main`, and pushes to `main`, run these checks. The CI workflow also runs on pushes to `cursor/**`. No workflow step runs `scripts/run_reid_eval.py`. In [`.github/workflows/ci.yml`](.github/workflows/ci.yml), `Lint, Test, and License Check` does this:

1. **Lint** with ruff (`ruff check src/ tests/ scripts/`)
2. **FFmpeg LGPL check** before the tests
3. **Unit tests** with pytest (`pytest tests/ -v --tb=short`)
4. **Clean worktree** — fails if `git status --porcelain` is not empty after pytest, and fails if pytest created `audit_logs/` in the checkout
5. **License fence check** — the `check_licenses.py` command above. It fails on any AGPL/GPL/NC/unknown license or banned component
6. **CI tools banned-licence check** — separate venv from `requirements-ci.lock`
7. **AC-13 demo** — the command below

```bash
python scripts/demo_audit_log.py --json
```

**Fixture separation** is a separate job in that same workflow file. Its command is in the licence section above.

Other workflows in `.github/workflows/` cover quality (lint, format, types, coverage, complexity, UI), security (bandit, semgrep, pip-audit, dependency review, npm audit, CodeQL, gitleaks), and project rules. `quality / tests+coverage` runs the suite with coverage and fails under 80%. `quality / lint` also checks formatting. If a note in `docs/ci.md` disagrees with a workflow file, the workflow file is what CI runs.

numpy 1.26.4 and its manifest row come from #6. ImageHash 4.3.2 is recorded as BSD-2-Clause.

### Development workflow

This is a **multi-task build** (T0–T10). Each task is one pull request. Current status on this commit:

- ✅ **T0:** Scaffold + license fence
- **T1:** Synthetic smoke fixtures and separation checks are on main. numpy 1.26.4 and its manifest row come from #6.
- ⏳ **T2:** Ingest (video file + optional RTSP)
- ✅ **T3:** Detection and single-camera tracking: YOLOX-S on onnxruntime (CPU; weights stay outside the repo), scripted detectors, an in-repo ByteTrack tracker, a stdlib SQLite track store outside the repo, and the M1a runner `scripts/run_smoke.py`.
- ⏳ **T4:** Attributes (color + bag)
- ⏳ **T5:** ReID has landed: untrained placeholder and random-init OSNet-lite embedders, vector stores, and an eval harness. The AC-06 harness is DEFERRED. No training runs.
- ⏳ **T6:** Cross-camera matching + route
- ⏳ **T7:** Web UI + auth
- ✅ **T8:** Audit log (HMAC chain and signed head anchor; see below). Merged in #5.
- **Partly done — T9:** Retention and purge. T-P-1, T-P-2, and T-P-3 are merged (#48, #49, #50). T-P-4 through T-P-9 are not.
- ⏳ **T10:** Zero-face guarantee

`docs/trace-search-cut01-plan.md` is not in this repo. The cut notes that are here are [docs/cut-01-prototype.md](docs/cut-01-prototype.md).

### Audit log (AC-13)

The audit log records operator username, IST time, action, query, case ID, purpose, an optional `details` note, and result ids. It is an HMAC-SHA256 chain. Canonical JSON is `json.dumps(..., sort_keys=True, separators=(",", ":"))`. A signed head anchor at `<log_path>.head` stores the last HMAC and the entry count. `initialize()` writes a signed count-0 anchor for a new log. `append()` checks that anchor under the lock and raises when it does not match the tail. A missing, unsigned, or mismatched anchor fails `verify()`. `verify()` takes a shared lock on the same lock file and returns `valid=False` for non-UTF-8 bytes and for a blank `details` string.

The key comes only from `TRACE_AUDIT_HMAC_KEY` or from an explicit key file outside the configured audit directory. There is no default key. `verify()` never creates a key. `init_hmac_key()` is the only function that creates one. `AUDIT_LOG_DIR` and `is_audit_path()` (exported from `trace_search.audit`) are the default path and the purge-exemption check. `is_audit_path()` matches the resolved configured directory only, not every folder named `audit_logs`.

An attacker who holds the key and deletes both the log and the anchor is not detectable on this host. An attacker without the key who restores an older copy of both the log and the anchor also passes `verify()`. Logs written before `details` existed fail HMAC verification, because the canonical payload now includes `"details": null`. External anchoring is out of scope for Cut 01. The 1-year purge checkpoint is still T9 (T-P-6). It is not built yet. Audit retention is 365 days.

`tests/test_audit_log.py` collects 58 tests, all under `tmp_path`. The package note at [`src/trace_search/audit/README.md`](src/trace_search/audit/README.md) still says 55. Details and a sample proof are in that note.

### Privacy and ethics

- **No face recognition.** No face embeddings, no face crops, no identity lookup.
- **No demographic outputs.** No gender, age, ethnicity, or other sensitive attributes.
- **Human review only.** Every result is meant to be reviewed by a person before action. There is no web UI for that review yet.
- **Short retention.** The shipped windows are 7 days for footage-derived data and 365 days for the audit log (`trace_search.retention`). `purge_expired` deletes expired footage-derived rows. T9 is partly done (see the status list). The 1-year audit purge checkpoint is not built yet.
- **Consent-first rule.** Training does not run. The ReID eval in this repo uses in-memory synthetic samples only. Later training, if it is ever allowed, is consented self-recorded fixtures only. No public datasets.
- **Anonymous queries.** No names, no watchlists. The audit log requires a case ID and purpose on search, view, and export.

This tool is designed for **investigative forensic search** (for example tracing a lost patient or reviewing an incident), not real-time surveillance or automated alerts.

### Legal and compliance

- **India DPDP Act 2023:** Designed for private-site legal basis (Sec 7(a) with signage). Consent Manager registration and full duties start 13 May 2027.
- **No EU export yet.** Body-appearance tracking counts as biometric data under GDPR. EU deployment blocked pending legal review.
- **No real buyer footage in this repo.** Training does not run. Smoke and the ReID eval here are synthetic. No real buyer footage until Cut 02 (paid pilot under NDA).

See the PRD at [docs/cut-01-prototype.md](docs/cut-01-prototype.md) for the risk assessment and legal basis.

### Contributing

This is a **locked-scope prototype** for Cut 01. External contributions are not accepted yet.

Internal contributors: follow the task plan. Each task gets one PR. Run tests and license checks before pushing.

### License

MIT License. See [LICENSE](LICENSE) for full text.

All dependencies are tracked in [`licenses/manifest.json`](licenses/manifest.json). CI verifies that all installed packages (direct + transitive) have allowed licenses and are not on the banned list. See AC-01 for details.

### Contact

Owner: Shitanshu Sharma  
Cut 01 target date: Fri 4 Dec 2026 (IST)  
Kill/continue review: Tue 8 Dec 2026 (IST)
