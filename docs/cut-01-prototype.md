Decision needed: Cut 01 rev 3 is approved in code-only mode, and Q1, Q6, FX-CODE and Q10 are now decided. Three things are still open for you. (1) When to unpark the parked items: hardware, consented recording days, buyer interviews with the 11 Nov gate, the timed operator study, and counsel review. Unpark them by Fri 30 Oct to keep the kill test (AC-11) near Fri 4 Dec; otherwise Tue 8 Dec is only a "pipeline ready" review. (2) Name the independent QA grader by Mon 30 Nov 2026, or 8 Dec is NOT READY. (3) The remaining undecided defaults: Q2, Q3, Q4, Q5, Q7, Q8 and the open part of Q9.

**Status: APPROVED (code-only) — Tue 20 Oct 2026, 11:25 PM IST, by Shitanshu Sharma.**
He confirmed at 11:25 PM IST in the requesting agent's chat. His chief-of-staff agent reports he first said yes at 11:19 PM IST. Eng may start **code-only** work in `shitanshusharma/trace-search` (§0). Parked items stay parked until he unparks them in a new dated rev.

**Changelog**
- **rev 3b (Wed 21 Oct 2026, 7:45 AM IST):** Shitanshu Sharma: "allow OpenSSL and Intel IPP as narrow named exceptions for the OpenCV-bundled files only." OpenSSL 1.1.1w (`OpenSSL`) is allowed only for `opencv_python_headless.libs/libcrypto-337dac8b.so.1.1` and `opencv_python_headless.libs/libssl-28bef1ac.so.1.1` inside opencv-python-headless 4.10.0.84. Intel IPP 2021.11.0 (`LicenseRef-Intel-Simplified-Software-License`) is allowed only for `cv2/cv2.abi3.so` in that same wheel (ICV and IW). Counsel confirmation required before release. Neither licence is a global allowlist entry (AC-01, §3).
- **rev 3a (Wed 21 Oct 2026, 12:57 AM IST):** Shitanshu confirmed in the PM bot chat a narrow licence exception, first given in the loops chat. `GPL-3.0 WITH GCC-exception-3.1` is allowed for the GCC runtime libraries only (AC-01, §3). All other GPL still fails.
- **Rev 3, Wed 21 Oct 2026 (IST).** The owner confirmed four decisions at 12:43 AM IST on Wed 21 Oct; his chief-of-staff agent reports he first stated them at 11:50 PM IST on Tue 20 Oct. Rev 2 is saved as `cut-01-prototype.rev2.md`.
  - **Q1 DECIDED:** YOLOX is the detector.
  - **Q6 DECIDED:** retention is 7 days for footage-derived data and 1 year for the audit log. Updated in the IN list, AC-14 and both handoff templates.
  - **FX-CODE APPROVED:** the no-people test-fixture class, tag `CODE_TEST`.
  - **Q10 DECIDED:** QA is graded by an independent agent or person who is not a build bot (not loops, SWE or Engineering Lead), named by Mon 30 Nov 2026. If nobody is named by then, the 8 Dec review is NOT READY.
  - The Decision-needed line now lists only what is still open. Status is unchanged (APPROVED, code-only). Risks 7 and 8 are unchanged. No dates moved.
- **Rev 2, Tue 20 Oct 2026 (IST).** Owner approved in code-only mode (rev 1 saved as `cut-01-prototype.rev1.md`):
  - Status changed to APPROVED (code-only). The Decision-needed line now lists what is still open.
  - Added **§0 Code-only mode**: parked items, untrained ReID placeholder, synthetic smoke video, repo and Eng bots.
  - Milestones M1 (consent/hardware parts), M2, M3 and M7 (held-out and study parts) marked **PARKED**. M4–M6 reworded to code-only scope. M8 becomes a **pipeline-ready review**.
  - Split AC-02 into **AC-02a** (consent records, DEFERRED) and **AC-02b** (separation check, ACTIVE). AC-01 now also covers synthetic generators and code-test fixtures.
  - ACs **AC-03, AC-04, AC-05, AC-06, AC-07, AC-08, AC-11 and AC-16 are DEFERRED.** AC-09 and AC-10 reworded to run on in-repo synthetic code-test fixtures (`FX-CODE`).
  - §5 adds a code-only rule. §6 and the §7 templates updated for code-only mode. Risks renumbered (old numbering skipped 5) and Risk 8 added. Q9 partly answered. **Q10 added** (QA role).
  - **No dates moved.** Fri 4 Dec 2026 holds for the active ACs.
- **Rev 1, Tue 20 Oct 2026 (IST).** Owner request, relayed by the chief-of-staff agent:
  - Added **M1a: stock-footage smoke test**, Mon 26 Oct – Wed 4 Nov. It runs in parallel with M1 and finishes before the M2 recording day.
  - Added the **§2a separation rule** for stock clips.
  - Reworded **AC-01** (stock clip rows in the licence manifest) and **AC-02** (consent scope plus an FX-SMOKE separation check).
  - Added **Out** items: public webcam streams and unsecured-camera sites.
  - Added a stock-footage licence row to §3, **Risk 6** (stock-footage terms) and **Q9**.
  - Added one line to each locked handoff template.
  - **No existing dates moved. Target is still Fri 4 Dec 2026.**
- **Rev 0, Tue 20 Oct 2026 (IST).** First draft (`cut-01-prototype.rev0.md`).

# PRD Cut 01 rev: Anonymous same-day route reconstruction, prototype

| | |
|---|---|
| Revision | **Rev 3**, dated Wed 21 Oct 2026 (IST). Earlier revs are kept at `cut-01-prototype.rev0.md`, `.rev1.md` and `.rev2.md` |
| Owner | Shitanshu Sharma |
| Dated | Tue 20 Oct 2026 (IST) |
| Target date | **Fri 4 Dec 2026 (IST)**: proof for all **active** ACs attached. This date holds in code-only mode. **Tue 8 Dec 2026 (IST): pipeline-ready review.** It becomes the kill/continue review only if the parked items are unparked in time (§0, §5). |
| Sources | `research/forensic-search-india/validation-pack.md` (VP), `stress-tests/occluded-person-id-cctv.md` (ST), `repo-shortlist.md` (RS, used for licences only) |
| Binding | Once approved, scope, ACs and dates change only through a new dated cut signed off by the owner. |

**What this cut proves.** One claim only: on mixed-brand cheap CCTV, an operator finds one anonymous person's same-day route across cameras **≥5x faster than manual review at usable recall** (VP §5 "Kill if", 4th bullet). We are not proving that the market exists, how the product is priced, or how it differs from competitors. Tentosoft and Staqu already market multi-NVR, DPDP and no-face ReID (VP §4 "Gap summary"; ST §A2). The only open wedge is *proven speed and accuracy of appearance-based path reconstruction* on cheap mixed estates (VP §4, gap 1). This cut tests that wedge. **In code-only mode (§0) it does not test the wedge yet.** It proves only that the licence, pipeline, privacy and UI plumbing is ready for the test.

**Product in one line.** Same-day, on-prem forensic search and cross-camera route reconstruction for one anonymous person, built for hospital chains and tech-park command centres with mixed Hikvision/CP Plus estates (VP §1 scoring: hospitals 24, tech parks 23). There are no names, no watchlist and no face recognition. A human reviews every result. This is the "investigative, human review" mode in ST §A1 "What is plausible", not alerting.

---

## 0. Code-only mode (rev 2; binding until the owner unparks items in a new dated rev)

- **Repo:** `shitanshusharma/trace-search`.
- **Eng role:** filled by the bots **loops**, **SWE** and **Engineering Lead**.
- **QA role (DECIDED 21 Oct, Q10):** an independent grader, agent or person, who is **not** a build bot (not loops, SWE or Engineering Lead). Must be named by **Mon 30 Nov 2026**. If nobody is named, the 8 Dec review is NOT READY. Eng bots may not sign QA handoffs.
- **PARKED** (not cancelled; milestones are kept and marked):
  - volunteer recording and fixture days (FX-TRAIN, FX-DEV, FX-HOLD)
  - hardware procurement (6 cameras, 2 NVRs, GPU box)
  - buyer interviews, including the **Wed 11 Nov interview gate**
  - the AC-11 timed operator study
  - counsel review of the licence manifest
- **ReID weights stay an untrained placeholder.** That means a random-init OSNet (torchreid, MIT) or a stub embedder behind the same interface.
  - No public or research-only weights.
  - **No training on any footage**: not stock, not synthetic, not consented.
  - Any similarity score the placeholder produces has no accuracy meaning and may not be reported as one.
- **Smoke footage:** licence-checked Pexels/Pixabay clips (§2a) **and synthetic video** are allowed, for smoke only.
  - Each synthetic clip gets a manifest row: generator name and version/commit, generator licence (and output terms, if a third-party tool), seed or prompt, SHA-256 of the output, tag `use=SMOKE_ONLY`, and fixture ID `FX-SMOKE-SYN-nnn`.
  - Synthetic clips must not depict or imitate any real, identifiable person.
  - Like stock clips, they never reach scoring, training, FX-HOLD or the study.
- **Code-test fixtures (`FX-CODE-nnn`, tag `use=CODE_TEST`; APPROVED by the owner 21 Oct).** These exist so the active ACs can be proven without breaking the SMOKE_ONLY rule.
  - They are seeded, generated in the repo, and contain **no people and no real or stock footage**: test-pattern video, synthetic track rows and random embeddings.
  - They are used only for functional, latency and privacy ACs (AC-09, AC-10, AC-12 to AC-15).
  - They carry no accuracy meaning and are subject to the same separation check (AC-02b).
- **What code-only completion means:** the pipeline is ready. **It cannot trigger the kill/continue decision (§5).**

## 1. Milestones and unlock order (all dates IST)

| # | Date | Milestone | Unlocked by |
|---|---|---|---|
| M0 | ~~Fri 23 Oct~~ **Done Tue 20 Oct, 11:25 PM** | Approved (code-only) by the owner. Q1, Q6 and Q10 were decided on Wed 21 Oct (rev 3). Q2–Q5, Q7 and Q8 remain undecided defaults; Q9 is partly answered | — |
| M1 | Fri 30 Oct | **Active:** licence manifest v1 clean (AC-01), including stock, synthetic and code-test rows. **PARKED:** consent form, fixture plan, and on-site cameras/NVRs/GPU box | M0 |
| M1a | Mon 26 Oct – Wed 4 Nov | **Smoke test (§2a):** ingest, detection and tracking bring-up on ≤20 licensed Pexels/Pixabay clips plus synthetic video, on available dev hardware. **Counts toward no AC and no score.** Exit: end-to-end run on ≥5 clips with no crash, and every clip's manifest row present | M0, plus each clip's manifest row logged before use |
| M2 | ~~Thu 5 Nov~~ | **PARKED.** Fixtures FX-TRAIN, FX-DEV and FX-HOLD recorded, and FX-HOLD sealed to QA | Unpark + M1 parked parts |
| M3 | ~~Wed 11 Nov~~ | **PARKED.** Interview gate (VP "Next actions" #1). If the owner reports any VP §5 interview kill signal anyway, it still stops the build | Unpark |
| M4 | Fri 13 Nov | **Code-only:** single-camera pipeline (generic MP4 file ingest, YOLOX detection, OC-SORT tracking, colour/bag attributes) runs end-to-end on FX-SMOKE and FX-CODE. Brand native-export parsers are written against public format docs where available, with no AC credit (AC-03 deferred). No accuracy claims | M1a |
| M5 | Fri 20 Nov | **Code-only:** untrained ReID placeholder behind the final interface. Embedding store (Qdrant). Cross-camera linker with topology and time-window logic, unit-tested on FX-CODE. Search API. **ReID training PARKED** | M4 |
| M6 | Fri 27 Nov | Web UI (site map, route, clips), audit log, retention purge, auth. **Code freeze Wed 2 Dec**; tagged SHA | M5 |
| M7 | Thu 3 – Fri 4 Dec | **Active:** QA runs every active AC from the frozen SHA. **PARKED:** FX-HOLD accuracy runs and the AC-11 timed study | M6, independent QA grader named by Mon 30 Nov (Q10, decided) |
| M8 | Tue 8 Dec | **Pipeline-ready review** (§5): READY or NOT READY. It is a kill/continue review only if M2, M3 and the M7 study have been unparked and completed | M7 |

**Code-only order (rev 2):** licences → smoke test on stock + synthetic → single-camera pipeline → ReID placeholder and linker → UI/privacy → frozen SHA → QA on active ACs → pipeline-ready review.

**Original full order (applies once unparked):** **licences → (smoke test on stock clips, parallel, no scoring) → consent and fixtures → interview gate → pipeline → ReID → UI/privacy → frozen SHA → QA on held-out → decision.** Nothing is trained, evaluated or demoed on footage that lacks a consent record. M1a does not unlock or block any later milestone. If it slips, M4 absorbs the bring-up work and dates hold.

## 2. IN / OUT (dated 20 Oct 2026; OUT items are not revisited before 8 Dec 2026)

**IN** (proposed numbers are labelled):
- **6 cameras** (proposed): 3 Hikvision and 3 CP Plus, 2 MP class, each brand recording to its own NVR. Mixed indoor/corridor/entrance views, at least 1 low-light view, at least 1 high-mount low-res view. ~~Due M1~~ **PARKED.**
- **Video-file ingest** of each NVR's *native export*, plus per-camera clock-offset correction. Due M4. *Live RTSP is a stretch goal and is not required for any AC.*
- Person detection, single-camera tracking, cross-camera ReID with camera-topology and transit-time limits (ST §A1 "What is plausible"). Due M5. *In code-only mode, ReID is an untrained placeholder (§0).*
- **Attribute search:** upper and lower clothing colour, plus bag/backpack. Colour comes from a deterministic colour classifier on torso and leg regions, so no trained weights are needed. Bag uses the detector's COCO `backpack`/`handbag` classes. Due M4.
- Query by example crop or by attributes, inside a time window and camera set. Due M5.
- **Site-map route view:** a floor plan with camera pins, the route in time order, and a playable clip for each segment. Clip export. Due M6.
- Basic web UI with local username/password and two roles (operator, admin). Due M6.
- **Minimal audit log:** append-only and hash-chained. It records who, when, action, query, case ID/purpose and result IDs. It is kept for **1 year** (DECIDED 21 Oct, Q6) to match DPDP Rule 6 access logs (VP §1). Due M6.
- **Retention setting: 7 days for footage-derived data (DECIDED 21 Oct, Q6)**, with an automatic purge. The audit log is kept 1 year (DECIDED). Due M6.
- Consented, self-recorded fixtures only, for anything trained, scored or demoed. ~~Due M2~~ **PARKED.**
- **Smoke test (M1a):** licensed Pexels/Pixabay clips **and synthetic video (rev 2)** for detection/tracking bring-up only, under the §2a rules. Due Wed 4 Nov.
- **Code-test fixtures (`FX-CODE`, rev 2):** seeded, in-repo, no people. Used for the active functional, latency and privacy ACs. Due M4.

**OUT:**
- Gait, face recognition, face embeddings and face crops as features.
- Watchlists, enrolment and named identity: no name fields and no ID lookup.
- Cross-day or clothing-change matching. ST §A1 shows clothes-changing ReID at 42–74% rank-1.
- Gender, age, ethnicity or other sensitive attributes. Intel's attribute model, for example, outputs `is_male`; that is out.
- Height, pose and depth estimation.
- Natural-language/CLIP text search, because of LAION/DataComp weight terms (RS §9).
- 50+ cameras per node, multi-node, Kubernetes, SSO/LDAP, and the 500 ms alert target (ST §A6). No real-time alerts at all.
- Cloud processing. On-prem only, with no outbound data.
- Mobile app.
- VMS/NVR SDK integrations (Milestone, Genetec, HikCentral, Tentosoft). File ingest and optional RTSP only.
- Real buyer footage. That belongs to the next cut, under NDA.
- Public datasets for training or evaluation: Market-1501, DukeMTMC, MSMT17, CUHK03, CASIA-B, GREW, Gait3D, Objects365, ImageNet-pretrained backbones where avoidable.
- OpenGait, InsightFace/ArcFace weights, Ultralytics, boxmot.
- Pricing, packaging and OEM SDK.
- **Public webcam streams** (rev 1), for example EarthCam and YouTube live, in any form: ingest, smoke test, demo or screenshot.
- **Unsecured or exposed-camera sites** (rev 1), for example Insecam and similar aggregators of open camera feeds. They are never to be accessed or used.
- **Stock clips (rev 1)** in any training set, FX-DEV, FX-HOLD, the AC-11 study, any reported metric, any buyer demo or any marketing. Also out: bulk or scripted download from stock sites.

## 2a. Smoke-test stage on stock and synthetic footage (rev 1, synthetic added rev 2; no scoring)

**Purpose.** Bring up ingest, detection and tracking before the consented recording days. This is plumbing only. Smoke results are **not evidence** for any AC, any milestone exit (other than M1a's own) or the §5 decision.

**Rules (binding):**
1. **Sources:** Pexels, Pixabay and synthetic video (rev 2; manifest row per §0) only. For Pexels/Pixabay: At most **20 clips** (proposed), each downloaded **manually, one at a time**. No scripts, no API bulk pulls and no scraping. Both sites' terms ban automated or bulk copying, and scraping "for machine learning purposes" (see §3).
2. **Manifest row per clip before use:**
   - source page URL
   - licence name and version: "Pexels License (Pexels ToS, last updated 15 Nov 2024)", "Pixabay Content License (Pixabay ToS, last updated 18 Nov 2024)", or CC0 where the page marks it
   - date checked (IST)
   - SHA-256 of the downloaded file
   - frame perceptual hash (pHash) of 10 sampled frames
   - tag `use=SMOKE_ONLY`
   - fixture ID `FX-SMOKE-nnn` (synthetic: `FX-SMOKE-SYN-nnn`, with generator, generator licence, seed/prompt and SHA-256)
3. **Clip choice:** prefer wide, CCTV-style, crowd/corridor clips with no close-up recognisable faces and no prominent logos. Skip any clip that shows minors, medical settings or anything that could put a person "in a bad light".
4. **Use:** detection and tracking bring-up only. No labelling for metrics, no ReID or attribute training (no training of any kind in code-only mode), no tuning thresholds against these clips, and no reported numbers. Clips are never shown outside the build team and never redistributed. Unaltered redistribution is a prohibited "Standalone" use under both licences.
5. **Storage:** a separate `smoke/` store. Training, eval and study configs refuse any input tagged `SMOKE_ONLY` (a hard error in the loader).
6. **Purge:** derived data from smoke runs (detections, tracks, crops) is deleted at M1a exit. Raw clips are deleted by M8 (Tue 8 Dec).
7. **QA separation check** (AC-02b):
   - 0 matches of any FX-SMOKE (stock and synthetic) or FX-CODE SHA-256 against every file in FX-TRAIN, FX-DEV, FX-HOLD, training manifests and the AC-11 study playlist. While those sets are parked, the check runs against whatever train/eval/study inputs and configs exist in the repo, and confirms no training job exists.
   - 0 pHash near-duplicates (Hamming distance ≤ 8, proposed) against those same sets, which catches re-encoded copies.
   - The loader guard rejects a planted `SMOKE_ONLY` file.

## 3. Stack and licence status (web-checked 20 Oct 2026)

| Part | Choice | Code licence | Weights / training data | Status |
|---|---|---|---|---|
| Detector | **YOLOX-S/M, pinned commit** | Apache-2.0 [link](https://github.com/Megvii-BaseDetection/YOLOX/blob/main/LICENSE) | COCO, **trained from scratch, no ImageNet** ([paper §2](https://arxiv.org/abs/2107.08430)). COCO annotations are CC BY 4.0; images fall under Flickr terms ([COCO terms](https://cocodataset.org/#termsofuse)) | **DECIDED (Q1, 21 Oct).** Residual risk: Flickr image terms. Repo has had no pushes for about 16 months (RS §1), so we pin and own the fork |
| Detector (rejected) | RT-DETR / v2 | Apache-2.0 [link](https://github.com/lyuwenyu/RT-DETR/blob/main/LICENSE) | COCO-only checkpoints use ImageNet-pretrained `vd` backbones (`pretrained: True` in configs). Higher-accuracy rows are **COCO+Objects365**, and Objects365 is "academic purpose only" ([terms](https://www.objects365.org/download.html)) | Rejected for this cut: dirtier weight lineage than YOLOX. **This reverses the VP "Next actions" #2 and RS choice.** |
| Detector (banned) | Ultralytics YOLO | **AGPL-3.0** ([licence](https://www.ultralytics.com/license)). Since Aug 2026, tracking-of-individuals use also needs Ultralytics' review (ST §A5) | — | OUT |
| Tracker | **OC-SORT via roboflow/trackers** | Apache-2.0 [link](https://github.com/roboflow/trackers/blob/develop/LICENSE). Reference code is MIT [link](https://github.com/noahcao/OC_SORT/blob/master/LICENSE) | Motion-only, **no weights** | Clean. ByteTrack is MIT [link](https://github.com/FoundationVision/ByteTrack/blob/main/LICENSE) and an acceptable alternative. BoT-SORT is MIT [link](https://github.com/NirAharon/BoT-SORT/blob/main/LICENSE), but its ReID option pulls fast-reid weights, so avoid that option. boxmot is AGPL and OUT |
| ReID code | **torchreid (OSNet)** | MIT [link](https://github.com/KaiyangZhou/deep-person-reid/blob/master/LICENSE). fast-reid is Apache-2.0 [link](https://github.com/JDAI-CV/fast-reid/blob/master/LICENSE) | — | Clean |
| ReID weights | **Code-only (rev 2): untrained placeholder.** When unparked: train our own OSNet from random init on consented FX-TRAIN footage | — | Public model-zoo weights are trained on Market-1501, MSMT17 and Duke ([torchreid zoo](https://kaiyangzhou.github.io/deep-person-reid/MODEL_ZOO), [fast-reid zoo](https://github.com/JDAI-CV/fast-reid/blob/master/MODEL_ZOO.md)). Whether they are clean is "legally unsettled" (ST §A5) | **No commercially clean pretrained ReID weight is certain. OPEN RISK.** Candidate: Intel OMZ `person-reidentification-retail-0277`. The repo is Apache-2.0 ([link](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/intel/person-reidentification-retail-0277)), but its **training data is not disclosed** and it reports Market-1501 scores. Use only if Intel confirms the data provenance in writing |
| Attributes | Deterministic HSV colour bins plus YOLOX COCO bag classes | Ours | No extra weights | Clean. PA-100K/RAP-trained models (PP-Human) are not used: dataset terms unchecked (RS §9) |
| Search / store | Qdrant | Apache-2.0 [link](https://github.com/qdrant/qdrant/blob/master/LICENSE) | — | Clean |
| Smoke footage (rev 1) | Pexels / Pixabay stock clips, ≤20, manual download | — | [Pexels licence](https://www.pexels.com/license/), [Pexels ToS (15 Nov 2024)](https://www.pexels.com/terms-of-service/), [Pixabay licence summary](https://pixabay.com/service/license-summary/), [Pixabay ToS (18 Nov 2024)](https://pixabay.com/service/terms/). Allowed: free commercial use and modification, no attribution. Prohibited: Standalone (unaltered) sale or redistribution; commercial use of depicted trademarks/logos; immoral or illegal use "especially Content which features a recognisable person"; implying endorsement; scraping/bulk copying, including "for machine learning purposes". **Neither site warrants model or privacy releases.** Both say that for commercial use "it is likely that you will need consent or a license", and the responsibility sits with the user. Terms can be changed or licences cancelled at any time. No clause names surveillance or biometric products specifically | Allowed **for internal bring-up only** (§2a). Not cleared for training, evaluation, demos or marketing |
| Synthetic smoke video (rev 2) | Chosen by Eng; each generator is listed in the manifest | Generator's own licence | Preference: in-repo procedural generator (our code, permissive dependencies). A third-party or generative-AI tool is acceptable only if its terms grant commercial rights to outputs and it renders no real-person likeness | SMOKE_ONLY. Any unknown generator licence fails AC-01 |
| GCC runtime (rev 3a) | libgfortran and similar, bundled in numpy wheels | GPL-3.0 WITH GCC-exception-3.1 | Owner-approved named exception, Wed 21 Oct 2026 | Allowed for GCC runtime libraries only. Any other GPL fails AC-01 |
| OpenSSL (rev 3b) | libcrypto and libssl bundled in opencv-python-headless 4.10.0.84: `opencv_python_headless.libs/libcrypto-337dac8b.so.1.1`, `opencv_python_headless.libs/libssl-28bef1ac.so.1.1` | OpenSSL, version 1.1.1w | Owner-approved named exception, Wed 21 Oct 2026, 7:45 AM IST | Allowed only for those two files. Counsel confirmation required before release. Any other package, version, or file fails AC-01. Not a global allowlist entry |
| Intel IPP (rev 3b) | Intel IPP ICV and IPP IW statically linked in `cv2/cv2.abi3.so` of opencv-python-headless 4.10.0.84 | LicenseRef-Intel-Simplified-Software-License (Intel Simplified Software License), version 2021.11.0 | Owner-approved named exception, Wed 21 Oct 2026, 7:45 AM IST | Allowed only for `cv2/cv2.abi3.so` in that wheel. Counsel confirmation required before release. Any other package, version, or file fails AC-01. Not a global allowlist entry |
| Video | FFmpeg **LGPL build** (no `--enable-gpl/nonfree`) | LGPL (RS §7) | — | Clean if built correctly; checked in AC-01 |

A licence scanner (ScanCode or equivalent) runs in CI. Any AGPL, GPL, NC or unknown result fails the build (RS "Key risks" #2).

## 4. Acceptance criteria

**AC state in code-only mode (rev 2).** ACTIVE ACs must pass by Fri 4 Dec. DEFERRED ACs are **neither passed nor failed**; they may not be reported, rounded or implied as passed, and they reactivate unchanged when their parked inputs are unparked.
- **ACTIVE:** AC-01, AC-02b, AC-09, AC-10, AC-12, AC-13, AC-14, AC-15.
- **DEFERRED:** AC-02a, AC-03, AC-04, AC-05, AC-06, AC-07, AC-08, AC-11, AC-16.

All numeric thresholds are **proposed targets** except where a source is named. Accuracy and speed ACs run **only on FX-HOLD**, from the frozen SHA, and QA runs them. Each AC is binary: pass or fail.

**Fixtures.** FX-TRAIN/FX-DEV and FX-HOLD are recorded on different days with **disjoint people** (proposed: at least 25 consented participants per fixture). Clothing stays the same within a day. There is deliberate occlusion (groups, doorways, carts) and at least 1 low-light camera. FX-HOLD is at least 2 h × 6 cameras = **12 camera-hours** with at least 20 labelled query persons. Ground truth lists each person's route segments as (camera, entry time, exit time).

| ID | Criterion (measurable) | Proof |
|---|---|---|
| AC-01 **ACTIVE** | The licence manifest lists every library, model, weight, dataset, **stock clip, synthetic generator/clip (rev 2) and FX-CODE generator (rev 2)** with licence, source URL and SHA-256. Counsel review is PARKED and not required for the code-only pass. Stock clip rows also carry licence version, date checked (IST), pHash and the `use=SMOKE_ONLY` tag (§2a). There are 0 AGPL/GPL/NC/unknown entries. **Sole GPL named exception (rev 3a, owner, Wed 21 Oct 2026, 12:57 AM IST):** `GPL-3.0 WITH GCC-exception-3.1` is allowed only for the GCC runtime libraries (e.g. libgfortran bundled in numpy wheels); every other GPL variant still fails the build. **Named exception (rev 3b, Shitanshu Sharma, Wed 21 Oct 2026, 7:45 AM IST):** OpenSSL 1.1.1w (`OpenSSL`) and Intel IPP 2021.11.0 (`LicenseRef-Intel-Simplified-Software-License`) are allowed only for the named files inside opencv-python-headless 4.10.0.84 (§3). Counsel confirmation required before release. They are not global allowlist entries. No banned item (§2 OUT) appears. The scanner report is clean | SHA, API JSON |
| AC-02a **DEFERRED** | Fixture provenance: every person visible in FX-TRAIN, FX-DEV and FX-HOLD has a signed consent record (100%). Each fixture has a fixture ID and SHA-256. ≥2 brands, 6 cameras, ≥12 camera-hours in FX-HOLD | fixture id, SHA |
| AC-02b **ACTIVE** | **Separation:** 0 FX-SMOKE (stock and synthetic) or FX-CODE SHA-256 matches and 0 pHash near-duplicates in any train/eval/study input, manifest or playlist (§2a rule 7). The loader rejects planted `SMOKE_ONLY` and `CODE_TEST` files. No training job exists in code-only mode. FX-SMOKE and FX-CODE need no consent records because they are never scored | fixture id, SHA, API JSON |
| AC-03 **DEFERRED** (no licence-clean brand sample exports without the parked NVRs) | Native exports from both the Hikvision NVR and the CP Plus NVR ingest with no manual transcoding. After offset correction, per-camera clock error is ≤1 s against ground-truth sync events | API JSON, fixture id |
| AC-04 **DEFERRED** (needs the stated GPU box, which is parked) | Ingest throughput is **≥1.0x real time in aggregate** for 6 × 1080p streams on the stated box (default: 1 × RTX 4070-class 12 GB GPU, 8-core CPU, 32 GB RAM). That means 12 camera-hours are indexed in ≤2 h wall-clock | API JSON, SHA |
| AC-05 **DEFERRED** (needs FX-HOLD) | Person detection recall is ≥90% at IoU 0.5 on ≥1,000 labelled FX-HOLD boxes. Single-camera tracking IDF1 is ≥70%. The low-light/occluded subset is reported separately and must have recall ≥75% | API JSON, fixture id |
| AC-06 **DEFERRED** (needs FX-HOLD) | Cross-camera ReID with topology/time limits reaches **Rank-1 ≥90%** (the same-day pass bar from ST §(e) test 1) and mAP ≥70% | API JSON, fixture id |
| AC-07 **DEFERRED** (needs FX-HOLD) | Route reconstruction over ≥20 query persons: segment recall ≥80% ("usable recall" for this cut), segment precision ≥90%, **false-merge rate ≤5%** (share of system cross-camera links that join two different people) | API JSON, fixture id |
| AC-08 **DEFERRED** (needs FX-HOLD) | Attribute search: across ≥10 colour/bag queries, mean precision@10 ≥80% | API JSON, fixture id |
| AC-09 **ACTIVE** (code-only form) | Search latency p95 ≤2 s for an attribute or crop query over a **synthetic FX-CODE index equivalent to 12 camera-hours** (track and embedding counts stated in proof). Route build p95 ≤10 s. Measured **on available dev hardware (proposed)**, with the hardware stated. Must be re-measured on FX-HOLD when unparked | API JSON, SHA, fixture id |
| AC-10 **ACTIVE** (code-only form) | The UI shows the site map with camera pins, the ordered route, a playable clip for each segment, and clip export, for 3 named **FX-CODE** routes (test-pattern video, no people). Repeated on FX-HOLD when unparked | screenshot, fixture id |
| AC-11 **DEFERRED** (study and FX-HOLD parked) | **Kill-criterion test: tool time-to-route ≤ 1/5 of manual time AND tool route recall ≥80%** (VP §5). Protocol below | API JSON, fixture id, SHA |
| AC-12 **ACTIVE** | No face model is in the manifest. A DB/storage scan finds 0 face embeddings and 0 stored face crops. Whole-body crops are kept only for retention TTL. No name/identity field or watchlist endpoint exists in the API schema | API JSON, SHA |
| AC-13 **ACTIVE** | Audit log: 50 scripted actions (login, search, view, export, retention change, purge) produce 50/50 entries with who, when, action, case ID and result IDs. Tampering with one entry breaks hash-chain verification | API JSON |
| AC-14 **ACTIVE** | Retention (**7 days footage-derived / 1 year audit log, DECIDED 21 Oct**): (a) the shipped config is exactly 7 days and 365 days; (b) on FX-CODE, advance the clock 8 days and run purge: 0 crops, embeddings, tracks and clips older than 7 days remain, searches return nothing for that window, and all audit entries are intact; (c) advance the clock past 365 days and run purge: only audit entries older than 1 year are removed, and the hash chain still verifies for the retained entries | API JSON, screenshot |
| AC-15 **ACTIVE** | Access control: logged-out requests to any search, route, clip or export URL return 401. No public share links exist in this cut, so a logged-out share URL is not accepted as proof for any page that shows footage | API JSON |
| AC-16 **DEFERRED** (depends on held-out metrics) | QA reproduces AC-04 to AC-11 from the tagged SHA on FX-HOLD. Each metric lands within ±2 points of Eng's reported value | SHA, API JSON |

**AC-11 protocol (locked; DEFERRED with AC-11):**
- **Operators:** 4 people who are not on the build team (proposed). At least 2 should have console/NVR experience.
- **Queries:** 12 from FX-HOLD (proposed). Each query gives the operator a reference crop and a start camera and time.
- **Footage:** 12 camera-hours, both brands.
- **Design:** crossover. Each operator does 6 queries manually and 6 with the tool. No operator sees the same query twice. Order is counterbalanced.
- **Manual arm:** the brand's own NVR playback client or VLC, multi-tile and up to 8x speed allowed.
- **Measurement:** the clock runs from receiving the query to submitting a route as (camera, in, out) rows. Manual trials are capped at 60 min, and a capped trial is scored as 60 min. The cap undercounts manual time, so it is conservative for us.
- **Metrics:** ratio = median manual time / median tool time. Recall is scored with ±10 s tolerance.
- **Ingest time:** excluded from tool time only if AC-04 passes, because the system could then keep up live. If AC-04 fails, ingest time is added.
- **Raw data:** the full timing sheet is attached as JSON.

## 5. Kill / continue rule (decided Tue 8 Dec 2026)

**Code-only rule (rev 2).** Code-only completion **cannot trigger KILL or CONTINUE.** Tue 8 Dec is a **pipeline-ready review**:
- **READY:** all 8 active ACs (AC-01, AC-02b, AC-09, AC-10, AC-12, AC-13, AC-14, AC-15) pass, run by QA from the frozen SHA.
- **NOT READY:** any active AC fails or lacks proof. A fix needs a new dated rev.
- Deferred ACs are reported as DEFERRED, never as passed.
- The rules below apply only if the parked items (M2, M3, the M7 study and AC-11) are unparked and completed. The interview kill still applies at any date if the owner reports one.

- **KILL this product line** if AC-11 fails on FX-HOLD (<5x, or recall <80%). This is VP §5's 4th kill condition. Per VP §5, we then switch to H2 (DPDP governance layer), which must be validated separately.
- **KILL at once, at any date**, if VP §5 interview kills fire: fewer than 3 of about 12 interviews report monthly failures; buyers say Tentosoft, HikCentral or NVR smart search already does the job; or, by 8 Dec, no buyer will sign a paid-pilot LOI.
- **HOLD (no pilot)** if AC-11 passes but any other AC fails. Licensing (AC-01) and privacy (AC-12 to AC-15) failures block any buyer footage. A fix needs a new dated cut. FX-HOLD may not be re-run after tuning on it; a new held-out fixture is required.
- **CONTINUE to Cut 02** (a paid pilot on one LOI buyer's past incidents under NDA, VP "Next actions" #1) only if **all ACs pass** (AC-01 to AC-16, with AC-02a and AC-02b both counted) and the interview gate passed.

## 6. Proof Eng must attach

- Every AC row needs its listed proof types attached to the PR description and to the M7 report. A missing proof means the AC failed.
- **Success is not PR count, commit count, bot loop iterations or demo videos.** In code-only mode, only **active ACs passed by QA from the frozen SHA** count. When unparked, only ACs passed on FX-HOLD count.
- Deferred ACs are not partially passed. Placeholder-ReID outputs and any "looks right on synthetic" result are not evidence for AC-05 to AC-08 or AC-11.
- **No soft-pass.** "Mostly works", "passes on dev", "passes except low-light" and "within noise" all count as **fail**. There is no rounding, and thresholds are not moved after the fact.
- Numbers come from QA's run on the frozen SHA, not from Eng's notebook.
- **Smoke-test (FX-SMOKE) output is never proof.** Citing it for any AC fails that AC.

## 7. Handoff templates (locked; field names may not be changed; rev 1 and rev 2 changes were approved by the owner. Rev 2 changes cover code-only mode and deferred ACs only)

**Eng handoff, to: Eng**
```
Cut: 01 rev 3 | Mode: CODE-ONLY | Repo: shitanshusharma/trace-search | Eng bot: loops / SWE / Engineering Lead
Milestone: M_ | Date (IST): ____ | Frozen SHA: ____
ACs claimed (ACTIVE only: AC-01, AC-02b, AC-09, AC-10, AC-12–AC-15): AC-__ ... | For each: value vs threshold | proof (screenshot | logged-out share URL | API JSON | SHA | fixture id) + link
Licence manifest SHA-256: ____ | Scanner report: ____ | New deps since last handoff: ____
DEFERRED (not claimed, not failed): AC-02a, AC-03–AC-08, AC-11, AC-16 [confirm none reported as passed: yes/no]
ReID: untrained placeholder [yes/no] | No public/research weights [yes/no] | No training job on any footage [yes/no]
Fixtures touched (ids): ____ | Confirm: FX-HOLD labels NOT accessed [yes/no/n.a. – parked]
FX-SMOKE (stock/synthetic) or FX-CODE used this milestone [yes/no] | Manifest rows present for each [yes/no] | Confirm: no SMOKE_ONLY clip in any train/eval/study input [yes/no] | Smoke-derived data purged [yes/no/n.a.]
Retention config: footage-derived TTL = 7 days [yes/no] | audit log = 1 year [yes/no] (both DECIDED 21 Oct; any other value = AC-14 FAIL)
Hardware used (dev hardware, proposed): GPU/CPU/RAM ____ | Known failures / ACs not met: ____ (no soft-pass wording)
Blocked on owner: ____
```
**QA handoff, to: QA**
```
Cut: 01 rev 3 | Mode: CODE-ONLY | QA grader (independent; not loops/SWE/Engineering Lead; named by Mon 30 Nov): ____ | Run date (IST): ____ | SHA under test: ____
Fixture: FX-CODE ids ____ SHA-256 ____ | FX-HOLD: n.a. – PARKED
Per ACTIVE AC: AC-__ | measured | threshold | PASS/FAIL | proof link | reproduce cmd
DEFERRED (listed, not scored): AC-02a, AC-03, AC-04, AC-05, AC-06, AC-07, AC-08, AC-11, AC-16
AC-11 study: DEFERRED – PARKED (fill only if unparked) | operators N=__ | queries N=__ | camera-hours __ | brands __ | median manual __ min | median tool __ min | ratio __x | recall manual __% / tool __%
Retention verified: 7-day footage-derived purge [PASS/FAIL] | 1-year audit keep [PASS/FAIL] (DECIDED values, not proposals)
AC-02b separation check (FX-SMOKE stock+synthetic, FX-CODE; SHA-256 + pHash + loader guard + no training job): matches __ (must be 0) | guard rejected planted files [yes/no]
Deviations from protocol: ____ (any deviation = AC FAIL unless owner signed off before run)
Overall (active ACs): ALL PASS / FAIL (list) | Code-only result per §5: READY / NOT READY
(KILL / HOLD / CONTINUE only if parked items are unparked and completed)
```

## 8. Risks

1. **Accuracy on real estates.** ReID collapses across domains: Market→MSMT about 19.8% rank-1 (ST §A1). Our fixture is cleaner than a hospital corridor, so a pass here is necessary but not sufficient, and Cut 02 must re-test on buyer footage. Occlusion, low-res 2 MP high mounts, NVR re-compression and clock drift will all hurt. Training from scratch on a few dozen people may underfit; the fallback is more consented recording days.
2. **Licensing.** No clean public ReID weight is certain (ST §A5, RS §0). COCO images carry Flickr terms. YOLOX is stale. Counsel has not reviewed any of this (counsel review is PARKED in code-only mode). Intel OMZ provenance is unknown.
3. **Legal (India).** DPDP duties start 13 May 2027, and Consent Manager registration starts 13 Nov 2026 (ST §A4). The private-site legal basis (Sec 7(a) with signage) is untested for cross-camera profiling (ST §A4). DPDP has no special biometric category, but body-appearance tracking counts as biometric under GDPR, EDPB and ICO, which blocks easy EU export (ST §A4). Our mitigations are no names, short TTL, purpose and case ID on every query, a 1-year audit log, and human review only.
4. **Commercial.** A 5x pass does not beat Tentosoft or Staqu on its own (VP §4). The VP "Next actions" #3 competitive teardown is still not done. Price hypothesis: ₹1,500–3,000/camera/year (VP §4). A per-node hardware cost is unknown (ST §A6), and AC-04 gives the first data point.
5. **Stock and synthetic footage terms (rev 1; synthetic added rev 2).**
   - Neither Pexels nor Pixabay warrants model releases. Both reserve the privacy rights of identifiable people and put the "consent or licence" burden on the commercial user.
   - Both ban illegal or immoral use, especially of recognisable people, and both ban scraping or bulk copying, including for machine learning.
   - Running person detection and tracking on strangers who were filmed for stock use is a grey area, both for privacy (and, unverified, possibly DPDP) and for how the licences read for a surveillance-type product. No clause bans it outright.
   - Both sites can change the terms or cancel the licence at any time.
   - **Mitigation:** internal bring-up only; ≤20 manually downloaded clips; no faces close-up; purge derived data at M1a exit; never in training, evaluation, demos or marketing; terms checked on 20 Oct 2026 and re-checked on each clip's download date. Owner has allowed stock plus synthetic for smoke only (Q9). Synthetic generators carry their own output-licence and likeness risk; an unknown licence fails AC-01.
6. **Schedule.** Code-only work is on track for Fri 4 Dec, with smaller scope than rev 1 and no labelling. **The kill test is not scheduled.** If the parked items are unparked after Fri 30 Oct, the earliest kill-test date slips about day-for-day past Fri 4 Dec (proposed estimate; to be re-dated in the unparking rev). VP "Next actions" #1 (interviews within 3 weeks) lapses while interviews are parked. Diwali week (8 Nov) applies when labelling resumes.
7. **Code-only blind spot (rev 2).** Building UI, linker and search on an untrained ReID placeholder means the hardest risk (Risk 1) is untested until unparked. A READY pipeline can still fail AC-11. Sunk-cost pressure on the 8 Dec review must not be read as evidence.
8. **QA independence (rev 2).** The Eng role is filled by bots (loops, SWE, Engineering Lead). If QA is also an Eng bot, results are self-graded. Q10 must name a separate QA before M7.

## 9. Open questions for Shitanshu (rev 3: Q1, Q6 and Q10 DECIDED on Wed 21 Oct 2026; Q2–Q5, Q7 and Q8 are still **undecided defaults**; Q9 is partly answered)

1. ~~Use YOLOX instead of RT-DETR?~~ **DECIDED Wed 21 Oct 2026: YOLOX is the detector.**
2. Use any research-only ReID weights, even for internal comparison (ST §(e)1 allows this with a lawyer's view)? **Default: no. Train our own; ask Intel about OMZ-0277 provenance in parallel.**
3. Build alongside interviews, or wait for them? **Default: build in parallel, with a hard gate on 11 Nov.** *(Interviews and the gate are PARKED in code-only mode.)*
4. Fixture venue and people? **Default: our office or a rented floor, ≥25 paid consented volunteers per day, signage, no non-consenting passers-by.** *(Applies when unparked.)*
5. Hardware budget (6 cameras, 2 NVRs, 1 GPU box)? **Default: you set a cap by 23 Oct. Eng buys the cheapest 2 MP models from each brand.** *(Applies when unparked.)*
6. ~~Retention?~~ **DECIDED Wed 21 Oct 2026: 7 days for footage-derived data and 1 year for the audit log.**
7. Who are the 4 operators for AC-11? **Default: contract staff with CCTV console experience, ideally from a Silbar-type SOC (VP §2).** *(Applies when unparked.)*
8. Short counsel review of the licence manifest before AC-01 sign-off? **Default: yes, before M2.** *(PARKED; the code-only AC-01 pass does not need it.)*
9. (rev 1; **partly answered rev 2**) Smoke-test footage? **Answered:** Pexels/Pixabay stock clips (§2a) plus synthetic video are allowed, **for smoke only**. **Still open (default):** whether team-recorded consented phone clips may also be used for smoke. *Default: no; not needed in code-only mode.*
10. ~~Who fills the QA role?~~ **DECIDED Wed 21 Oct 2026:** an independent agent or person who is not a build bot (not loops, SWE or Engineering Lead), named by **Mon 30 Nov 2026**. If nobody is named by then, the 8 Dec review is NOT READY. *Naming the grader is still pending.*
