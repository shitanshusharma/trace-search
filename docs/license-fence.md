# License fence

The fence bans GPL-family licenses. The single GPL-family named exception is GCC runtime libraries (`libgfortran`, `libquadmath`, `libgcc_s`, `libstdc++`) under `GPL-3.0 WITH GCC-exception-3.1` (or the GCC Runtime Library Exception wording). `WITH` is allowed only for that GCC exception. A banned term fails anywhere in an expression.

OpenSSL and Intel IPP are not on the global allowlist. Shitanshu Sharma, 21 Oct 2026, 7:45 AM IST: "allow OpenSSL and Intel IPP as narrow named exceptions for the OpenCV-bundled files only." Counsel confirmation required before release.

| Licence | Version | Package | Files |
|---|---|---|---|
| `OpenSSL` | 1.1.1w | opencv-python-headless 4.10.0.84 | `opencv_python_headless.libs/libcrypto-337dac8b.so.1.1`, `opencv_python_headless.libs/libssl-28bef1ac.so.1.1` |
| `LicenseRef-Intel-Simplified-Software-License` | 2021.11.0 | opencv-python-headless 4.10.0.84 | `cv2/cv2.abi3.so` (Intel IPP ICV and Intel IPP IW) |

A match also requires the manifest row name (`OpenSSL-libcrypto`, `OpenSSL-libssl`, `Intel IPP ICV`, `Intel IPP IW`) and that file's sha256. The same licence fails for any other package, version, file, name, or hash. `IJG` is an allowlist identifier because libjpeg-turbo ships `IJG AND BSD-3-Clause AND Zlib`. It is not one of these named exceptions.
