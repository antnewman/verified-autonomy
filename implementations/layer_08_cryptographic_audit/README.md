# Layer 08: Cryptographic Audit Trails — Implementation

## Quick Start

```bash
make install
make run
```

## Expected Output

```
Layer 08: Cryptographic Audit Trails
==================================================

Step 1 -- Recording Entries
------------------------------
  Entry 0: APPROVED   hash=daa7fd6e94629fa8...
  Entry 1: REJECTED   hash=73f028194ac92cde...
  Entry 2: APPROVED   hash=5653f0f186e371d5...

Step 2 -- Verification
-----------------------
Chain status: VALID
  All 3 entries verified.

Step 3 -- Tampering Detection
------------------------------
  Original entry 1 decision: REJECTED
  Simulating tampering: changing decision from REJECTED to APPROVED...

Chain status after tampering: TAMPERED
  Entry 1 has been tampered with.
  First invalid sequence: 1

Step 4 -- Persistence: Export / Import
---------------------------------------
  Exported chain (995 bytes).
  Chain status after reimport: VALID
  All 2 entries verified.

Summary:
  Step                              Status
  ------------------------------  --------
  Original chain                     VALID
  After tampering                 TAMPERED
  After export/import                VALID

See: layers/08-cryptographic-audit.md
```

## What This Code Does

Implements a domain-agnostic cryptographic audit chain using nothing but the
Python standard library.  Each entry in the chain records a SHA-256 hash of the
system's input, a SHA-256 hash of its output, the decision string, optional
metadata, a UTC timestamp, and the hash of the preceding entry (the "chain
link").  The entry's own hash is a SHA-256 digest of all those fields serialised
in canonical (sort_keys=True) JSON — making it fully deterministic and
tamper-evident.  If a signing_key is provided, an HMAC-SHA256 of the canonical
form is incorporated into the hash, binding every entry to that key.

The demo walks through four steps: (1) recording three loan decisions, (2)
verifying the intact chain returns VALID, (3) simulating tampering by
reconstructing a frozen AuditEntry with a modified decision field and
replacing it in the internal list — the hash chain catches the discrepancy
immediately, (4) exporting the original chain to JSON, reimporting it, and
confirming the VALID status survives the round-trip.

No external packages are required — hashlib, hmac, json, and dataclasses are
all standard library modules available in every Python 3.11+ environment.

## API

| Symbol | Signature | Description |
|---|---|---|
| `hash_data` | `(data: str) → str` | SHA-256 of a string; returns 64-char hex |
| `AuditEntry` | frozen dataclass | Immutable record: sequence, timestamp, input_hash, output_hash, decision, metadata, previous_hash, entry_hash, signature |
| `VerificationResult` | dataclass | Outcome of verify(): status, entries_checked, first_invalid_sequence, message |
| `AuditChain.__init__` | `(signing_key=None)` | Create an empty chain; optional HMAC signing key |
| `AuditChain.record` | `(input_data, output_data, decision, metadata=None) → AuditEntry` | Append a new entry and return it |
| `AuditChain.verify` | `() → VerificationResult` | Verify the full chain; status VALID / TAMPERED / EMPTY |
| `AuditChain.verify_entry` | `(sequence: int) → bool` | Verify a single entry; raises IndexError if out of range |
| `AuditChain.get_entry` | `(sequence: int) → AuditEntry` | Retrieve an entry; raises IndexError if out of range |
| `AuditChain.export_json` | `() → str` | Serialise the chain to JSON |
| `AuditChain.from_json` | `(json_str, signing_key=None) → AuditChain` | Reconstruct a chain from exported JSON |
| `AuditChain.entries` | property → `list[AuditEntry]` | Copy of the internal entries list |
| `AuditChain.length` | property → `int` | Number of entries |

## Testing

```bash
make test
```

69 tests, 100% coverage.  Tests cover: hash_data determinism and format;
record field values (previous_hash chain, sequence numbering, ISO 8601
timestamps, 64-char hex hashes, metadata defaults); verify status for empty /
valid / tampered chains including first, middle, and last entry tampering and
broken chain links; verify_entry bounds checking; signing key behaviour
(different hashes, correct/wrong key round-trips); export/import round-trips
including tampered JSON and invalid JSON; get_entry bounds; property invariants;
and main() smoke tests.

## Dependencies

Zero runtime dependencies.  All cryptographic operations use the Python standard
library: `hashlib` (SHA-256), `hmac` (HMAC-SHA256), `json` (canonical
serialisation), and `dataclasses`.  No third-party packages required.
