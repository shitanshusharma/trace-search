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
