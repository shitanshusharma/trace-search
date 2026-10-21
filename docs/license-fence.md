# License fence

The fence bans GPL-family licenses. The single GPL-family named exception is GCC runtime libraries (`libgfortran`, `libquadmath`, `libgcc_s`, `libstdc++`) under `GPL-3.0 WITH GCC-exception-3.1` (or the GCC Runtime Library Exception wording). `WITH` is allowed only for that GCC exception. A banned term fails anywhere in an expression.

OpenSSL and Intel IPP are not on the global allowlist. Shitanshu Sharma, 21 Oct 2026, 7:45 AM IST: "allow OpenSSL and Intel IPP as narrow named exceptions for the OpenCV-bundled files only." Counsel confirmation required before release.

| Licence | Version | Package | Files |
|---|---|---|---|
| `OpenSSL` | 1.1.1w | opencv-python-headless 4.10.0.84 | `opencv_python_headless.libs/libcrypto-337dac8b.so.1.1`, `opencv_python_headless.libs/libssl-28bef1ac.so.1.1` |
| `LicenseRef-Intel-Simplified-Software-License` | 2021.11.0 | opencv-python-headless 4.10.0.84 | `cv2/cv2.abi3.so` (Intel IPP ICV and Intel IPP IW) |

A match also requires the manifest row name (`OpenSSL-libcrypto`, `OpenSSL-libssl`, `Intel IPP ICV`, `Intel IPP IW`) and that file's sha256. The installed-closure check hashes each named file in site-packages. The exception applies only when that on-disk sha256 matches both the manifest row and the pinned value. A missing file or a mismatch fails closed and names the file and both hashes. The same licence fails for any other package, version, file, name, or hash. `IJG` is an allowlist identifier because libjpeg-turbo ships `IJG AND BSD-3-Clause AND Zlib`. It is not one of these named exceptions.

Pillow's bundled FreeType is not on the global allowlist. Shitanshu Sharma, 21 Oct 2026, 12:01 PM IST: "Do all of it, FreeType under FTL only."

Shitanshu Sharma, 21 Oct 2026, 1:33 PM IST: "Add MIT-open-group, bzip2 and zlib rows to #19". That approval covers the new `ALLOWED_SPDX` strings `MIT-open-group` and `bzip2-1.0.6`. This PR does not add `Zlib` to `ALLOWED_SPDX`. The zlib-ng rows use the existing non-SPDX identifier `Zlib`.

| Licence | Version | Package | Files |
|---|---|---|---|
| `FTL` | 2.14.3 | Pillow 13.0.0 | `pillow.libs/libfreetype-57f805ba.so.6.20.6` |

A match also requires the manifest row name (`FreeType`) and that file's sha256. The same installed-closure hash check applies. `FTL` on any other file fails. The FreeType credit line is in [`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md).

`MIT-Modern-Variant`, `libpng-2.0`, `X11`, `BSD-3-Clause-Open-MPI`, `MIT-open-group`, and `bzip2-1.0.6` are plain allowlist identifiers. `X11` is libxcb. `MIT-open-group` is libXau. `bzip2-1.0.6` is bzip2 1.0.8 inside Pillow's libfreetype. zlib-ng uses the existing non-SPDX identifier `Zlib`. They are not named exceptions.

Shitanshu, 2026-10-21 3:49 PM IST, loops chat: "Accept all six recommendations" (D-T3-6). That approval covers the `ALLOWED_NON_SPDX` strings `Pexels License` and `Pixabay Content License`. Both strings were already in the allowlist before this approval was recorded. They are plain allowlist identifiers for stock clips. They are not named exceptions. Non-SPDX matching uppercases both sides and requires the whole string to be equal, so case is ignored (`PEXELS LICENSE` matches `Pexels License`) and a substring does not match. `ALLOWED_SPDX` membership stays case-sensitive.

| Licence | Allowlist |
|---|---|
| `Pexels License` | `ALLOWED_NON_SPDX` |
| `Pixabay Content License` | `ALLOWED_NON_SPDX` |

Shitanshu Sharma, 22 Oct 2026, 1:11 AM IST, loops chat: "Map \"3-Clause BSD License\" to BSD-3-Clause". That approval adds one `PIP_LICENSES_TO_SPDX` entry for protobuf. The exact pip-licenses string `3-Clause BSD License` is rewritten to `BSD-3-Clause` for that package only. The text is exact, including case and hyphens. A near-miss such as `3 Clause BSD Licence` is not rewritten. No other mapping was added. `ALLOWED_SPDX`, `ALLOWED_NON_SPDX`, `GCC_RUNTIME_*`, and `OPENCV_BUNDLED_FILE_EXCEPTIONS` are unchanged.

| Package | Version | Licence | Source |
|---|---|---|---|
| onnxruntime | 1.31.0 | `MIT` | Wheel `onnxruntime-1.31.0-cp311-cp311-manylinux_2_28_x86_64.whl`. METADATA `License: MIT License`. |
| protobuf | 7.36.2 | `BSD-3-Clause` | Wheel `protobuf-7.36.2-cp310-abi3-manylinux2014_x86_64.whl`. METADATA `License: 3-Clause BSD License`. |
| flatbuffers | 25.12.19 | `Apache-2.0` | Wheel `flatbuffers-25.12.19-py2.py3-none-any.whl`. METADATA `License: Apache 2.0`. The wheel has no LICENSE file. The row cites the upstream Apache-2.0 text at <https://github.com/google/flatbuffers/blob/v25.12.19/LICENSE>. No mapping was added. pip-licenses reports `Apache Software License`. |
| packaging | 26.3 | `Apache-2.0` | Existing manifest row. Also required by onnxruntime. METADATA `License-Expression: Apache-2.0 OR BSD-2-Clause`. |

YOLOX-S COCO weights are Apache-2.0. The file is the official release asset at <https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.onnx>. It is not stored in this repository and CI does not download it. COCO images are subject to Flickr terms. COCO annotations are CC BY 4.0. Counsel review of the YOLOX weights (D-5) is still pending before release.

The YOLOX reference row (B-46) is Apache-2.0 at commit `6ddff4824372906469a7fae2dc3206c7aa4bbaee`.
