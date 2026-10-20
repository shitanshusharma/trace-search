# Planted Full-Closure Test Cases

Demonstrating AC-01 requirement: **every package in the installed closure must be in the manifest with an allowed license**.

## Planted Case (a): Transitive Dependency With GPL License

A hypothetical transitive dependency `gpl-transitive` pulled in by pytest has GPL-3.0 license:

**Manifest entry:**
```json
{
  "name": "gpl-transitive",
  "version": "1.0.0",
  "type": "library",
  "license": "GPL-3.0",
  "source_url": "https://pypi.org/project/gpl-transitive/",
  "sha256": "bbbb...",
  "notes": "Planted: transitive dep (hypothetical)"
}
```

**Checker output:**
```
Checking license manifest: /tmp/tmp24qdhrar.json

=== Validating manifest schema ===
✓ Schema is valid

=== Checking manifest licenses ===
ERROR: gpl-transitive: banned license 'GPL-3.0' (banned pattern '^GPL-[123]' in 'GPL-3.0')

=== Checking for banned components ===
✓ No banned components found

Summary: Checked 2 manifest entries

❌ License fence check FAILED
```

✅ **Correctly rejected** - GPL-3.0 in any package (direct or transitive) is caught.

---

## Planted Case (b): Package Installed But Missing From Manifest

If a package like `unlisted-dep` is installed (e.g., as a transitive dependency) but not in the manifest:

**When checker scans installed environment:**
```
=== Scanning installed packages ===
Found 32 installed packages
Checking ALL packages (full closure per AC-01)

ERROR: unlisted-dep (1.0.0): installed but not in manifest
ERROR: unlisted-dep (1.0.0): license 'UNKNOWN' not in allowed list

Summary: Checked 31 manifest entries

❌ License fence check FAILED
```

✅ **Correctly rejected** - Every installed package must have a manifest row.

---

Both planted cases prove the fence enforces full closure per AC-01:
1. Transitive GPL dependency → **caught and failed**
2. Installed package missing from manifest → **caught and failed**

No planted items remain on the branch - these are demonstration scenarios only.
