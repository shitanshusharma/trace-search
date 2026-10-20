# Planted Test Case Outputs

Demonstrating license fence catches all blocking scenarios:

## Agpl Installed Pkg

**Status:** ✅ PASS (correctly rejected)

```
Checking license manifest: /tmp/tmp_4w0l_w3.json

=== Validating manifest schema ===
✓ Schema is valid

=== Checking manifest licenses ===
ERROR: PyMuPDF: banned license 'AGPL-3.0' (banned pattern '^AGPL' in 'AGPL-3.0')

=== Checking for banned components ===
✓ No banned components found

Summary: Checked 1 manifest entries

❌ License fence check FAILED
```

## Lgpl Gpl Dual

**Status:** ✅ PASS (correctly rejected)

```
Checking license manifest: /tmp/tmp63xysumb.json

=== Validating manifest schema ===
✓ Schema is valid

=== Checking manifest licenses ===
ERROR: bad-lib: license 'LGPLv2.1+ OR GPLv2+' not in allowed list (must be exact SPDX or approved non-SPDX)

=== Checking for banned components ===
✓ No banned components found

Summary: Checked 1 manifest entries

❌ License fence check FAILED
```

## Mit Research Only

**Status:** ✅ PASS (correctly rejected)

```
Checking license manifest: /tmp/tmp3f2h_2se.json

=== Validating manifest schema ===
✓ Schema is valid

=== Checking manifest licenses ===
ERROR: research-lib: banned license 'MIT (research use only)' (banned pattern 'Research\s+[Uu]se\s+[Oo]nly' in 'MIT (research use only')

=== Checking for banned components ===
✓ No banned components found

Summary: Checked 1 manifest entries

❌ License fence check FAILED
```

## Model Zoo Reid

**Status:** ✅ PASS (correctly rejected)

```
Checking license manifest: /tmp/tmpmsni8ofd.json

=== Validating manifest schema ===
✓ Schema is valid

=== Checking manifest licenses ===
✓ No banned licenses in manifest

=== Checking for banned components ===
ERROR: osnet_x1_0_market_256x128.pth: contains banned component 'Market-1501 dataset'

Summary: Checked 1 manifest entries

❌ License fence check FAILED
```

## Insecam Clip

**Status:** ✅ PASS (correctly rejected)

```
Checking license manifest: /tmp/tmpeg8b26i8.json

=== Validating manifest schema ===
✓ Schema is valid

=== Checking manifest licenses ===
ERROR: insecam_camera_nyc_001.mp4: banned license 'Unknown' (banned pattern '^UNKNOWN$' in 'Unknown')

=== Checking for banned components ===
ERROR: insecam_camera_nyc_001.mp4: contains banned component 'Insecam (exposed cameras)'

Summary: Checked 1 manifest entries

❌ License fence check FAILED
```

## Schema Invalid Empty Name

**Status:** ✅ PASS (correctly rejected)

```
Checking license manifest: /tmp/tmpl44uz1jh.json

=== Validating manifest schema ===
ERROR: Schema validation failed: '' should be non-empty at entries.0.name

=== Checking manifest licenses ===
✓ No banned licenses in manifest

=== Checking for banned components ===
✓ No banned components found

Summary: Checked 1 manifest entries

❌ License fence check FAILED
```

## Innocent Substring

**Status:** ✅ PASS (correctly allowed)

```
Checking license manifest: /tmp/tmpgqg8x33d.json

=== Validating manifest schema ===
✓ Schema is valid

=== Checking manifest licenses ===
✓ No banned licenses in manifest

=== Checking for banned components ===
✓ No banned components found

Summary: Checked 1 manifest entries

✅ License fence check PASSED
```


---

All planted cases demonstrate the fence working correctly.
