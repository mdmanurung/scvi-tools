# Installed-artifact dependency authority

`scripts/accept_usage_readiness_wheel` accepts only an existing local directory with this layout:

```text
authority/
├── authority.json
├── requirements.lock
└── wheelhouse/
    └── every wheel required by requirements.lock
```

`authority.json` must be strict JSON:

```json
{
  "schema_version": "cytoanvi-dependency-authority-v1",
  "python": "3.13",
  "platform": "linux-x86_64-glibc2.28",
  "requirements_file": "requirements.lock",
  "requirements_sha256": "<64 lowercase hex>",
  "wheelhouse": "wheelhouse",
  "wheel_inventory": [
    {"filename": "package-version-tag.whl", "sha256": "<64 lowercase hex>"}
  ]
}
```

The lock grammar is deliberately narrower than general pip requirements syntax. Each logical
record must contain only `name==version` followed by one or more lowercase
`--hash=sha256:<64-hex>` tokens. A record may use backslash continuation and the file may contain
blank lines or full-line comments. URLs, local paths, editable requirements, environment markers,
inline comments, indexes/find-links, constraints, nested `-r`/`-c` includes, and every other pip
directive are rejected before pip runs. The harness itself supplies the verified local wheelhouse.

Every transitive requirement must be pinned in that lock. Every locked artifact must be in the
inventory and wheelhouse, every inventory hash must match, and no unlisted wheel may be used. The
acceptance harness uses `--no-index`, `--only-binary=:all:`, `--require-hashes`, and `--no-deps` for
the candidate.

A conda environment, `pip freeze`, editable checkout, requirements file without hashes, partial
cache, or wheelhouse without an inventory is not an exact authority. Network resolution or adding
dependencies requires separate approval. When this contract is unavailable, the only valid state
is `blocked_dependency_authority`; a smoke in the source-test environment is supplemental and
cannot change installed acceptance to passed.
