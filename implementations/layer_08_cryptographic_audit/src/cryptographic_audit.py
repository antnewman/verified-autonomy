"""
Layer 08: Cryptographic Audit Trails

Proving what the system did.

Part of the Verified Autonomy field guide.
https://github.com/antnewman/verified-autonomy

This module provides a domain-agnostic implementation of cryptographic
audit trails for AI system outputs.  It creates tamper-evident hash-chain
logs that prove what the system decided and when.  Any post-hoc modification
to a recorded entry — even a single character — is detected during
verification.

The hash chain works as follows:

- Each entry contains a SHA-256 hash of the input, a SHA-256 hash of the
  output, the decision string, metadata, a timestamp, and the hash of the
  preceding entry (the "chain link").
- The entry_hash is a SHA-256 digest of all the above fields, serialised
  in a canonical (sort_keys=True) JSON form.  This makes the hash fully
  deterministic and tamper-evident.
- If a signing_key is provided, an HMAC-SHA256 signature is computed over
  the canonical form and incorporated into the entry_hash, binding the
  entry to that key.  Reimporting the chain with the wrong key breaks
  every entry_hash and is detected immediately.
- Verification walks the chain, recomputes every entry_hash from scratch,
  and checks that each entry's previous_hash matches the preceding entry's
  entry_hash.

No external dependencies are required.  All cryptographic operations use
the Python standard library (hashlib, hmac, json).

See layers/08-cryptographic-audit.md for the full explanation, production
examples, and limitations.
"""

from __future__ import annotations

import hashlib
import hmac as _hmac
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def hash_data(data: str) -> str:
    """Compute a SHA-256 hash of a string.

    Returns a 64-character lowercase hex digest — the canonical form used
    throughout this module for input_hash, output_hash, and entry_hash.

    Args:
        data: Arbitrary string to hash.

    Returns:
        64-character lowercase hex string.
    """
    return hashlib.sha256(data.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AuditEntry:
    """A single immutable entry in a cryptographic audit chain.

    Attributes:
        sequence:      Zero-based position in the chain.
        timestamp:     ISO 8601 UTC timestamp at the moment of recording.
        input_hash:    SHA-256 of the serialised input data.
        output_hash:   SHA-256 of the serialised output data.
        decision:      Human-readable decision string recorded for this entry.
        metadata:      Arbitrary key-value metadata attached at record time.
        previous_hash: entry_hash of the preceding entry, or empty string
                       for the first entry.
        entry_hash:    SHA-256 (or HMAC-enhanced SHA-256) digest of all
                       other fields combined.  Any modification to the
                       entry invalidates this hash.
        signature:     HMAC-SHA256 hex string if the chain was created with
                       a signing_key; empty string otherwise.
    """

    sequence: int
    timestamp: str
    input_hash: str
    output_hash: str
    decision: str
    metadata: dict[str, Any]
    previous_hash: str
    entry_hash: str
    signature: str


@dataclass
class VerificationResult:
    """The outcome of verifying an AuditChain.

    Attributes:
        status:                 "VALID", "TAMPERED", or "EMPTY".
        entries_checked:        Number of entries successfully verified
                                before a fault was found (or total length
                                for VALID chains).
        first_invalid_sequence: Sequence number of the first entry that
                                failed verification, or None for VALID/EMPTY.
        message:                Human-readable summary.
    """

    status: str
    entries_checked: int
    first_invalid_sequence: int | None
    message: str


# ---------------------------------------------------------------------------
# Core chain
# ---------------------------------------------------------------------------


class AuditChain:
    """A tamper-evident, append-only cryptographic audit chain.

    Records AI system decisions as a linked chain of SHA-256 hashes.
    Any modification to a recorded entry — including the decision, inputs,
    outputs, metadata, or timestamps — is detected when verify() is called.

    Optionally bind the chain to a signing key via the signing_key parameter.
    The key is incorporated into every entry_hash using HMAC-SHA256, so the
    chain cannot be verified (or forged) without the correct key.

    Usage::

        chain = AuditChain()
        chain.record(input_data={"text": "..."}, output_data={"label": "SAFE"},
                     decision="APPROVED")
        result = chain.verify()  # VerificationResult(status="VALID", ...)

    Args:
        signing_key: Optional secret key string.  When provided, each
                     entry_hash includes an HMAC-SHA256 of the canonical
                     entry data, binding the chain to that key.
    """

    def __init__(self, signing_key: str | None = None) -> None:
        self._entries: list[AuditEntry] = []
        self._signing_key = signing_key

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _canonical(
        self,
        sequence: int,
        timestamp: str,
        input_hash: str,
        output_hash: str,
        decision: str,
        metadata: dict[str, Any],
        previous_hash: str,
    ) -> str:
        """Serialise entry fields to a deterministic canonical JSON string."""
        return json.dumps(
            {
                "sequence": sequence,
                "timestamp": timestamp,
                "input_hash": input_hash,
                "output_hash": output_hash,
                "decision": decision,
                "metadata": metadata,
                "previous_hash": previous_hash,
            },
            sort_keys=True,
        )

    def _compute_entry_hash(
        self,
        sequence: int,
        timestamp: str,
        input_hash: str,
        output_hash: str,
        decision: str,
        metadata: dict[str, Any],
        previous_hash: str,
    ) -> tuple[str, str]:
        """Compute (entry_hash, signature) for the given field values.

        If a signing_key is present, the HMAC of the canonical data is
        appended before the final SHA-256, so the entry_hash is different
        from an unsigned chain and cannot be reproduced without the key.

        Returns:
            Tuple of (entry_hash_hex, signature_hex).
            signature_hex is empty string when no signing_key is set.
        """
        canonical = self._canonical(
            sequence, timestamp, input_hash, output_hash, decision, metadata, previous_hash
        )
        if self._signing_key:
            signature = _hmac.new(
                self._signing_key.encode(),
                canonical.encode(),
                hashlib.sha256,
            ).hexdigest()
            entry_hash = hash_data(canonical + "|" + signature)
        else:
            signature = ""
            entry_hash = hash_data(canonical)
        return entry_hash, signature

    def _verify_entry_fields(self, entry: AuditEntry) -> bool:
        """Return True iff the entry's stored entry_hash matches a fresh computation."""
        expected_hash, _ = self._compute_entry_hash(
            entry.sequence,
            entry.timestamp,
            entry.input_hash,
            entry.output_hash,
            entry.decision,
            entry.metadata,
            entry.previous_hash,
        )
        return entry.entry_hash == expected_hash

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record(
        self,
        input_data: Any,
        output_data: Any,
        decision: str,
        metadata: dict[str, Any] | None = None,
    ) -> AuditEntry:
        """Append a new entry to the audit chain and return it.

        The input and output data are serialised to canonical JSON before
        hashing, so they can be any JSON-serialisable value (string, dict,
        list, etc.).

        Args:
            input_data:  The system input at the time of the decision.
            output_data: The system output at the time of the decision.
            decision:    A human-readable decision string (e.g. "APPROVED").
            metadata:    Optional key-value metadata.  Defaults to {}.

        Returns:
            The newly created AuditEntry.
        """
        if metadata is None:
            metadata = {}

        sequence = len(self._entries)
        timestamp = datetime.now(timezone.utc).isoformat()
        input_hash = hash_data(json.dumps(input_data, sort_keys=True))
        output_hash = hash_data(json.dumps(output_data, sort_keys=True))
        previous_hash = self._entries[-1].entry_hash if self._entries else ""

        entry_hash, signature = self._compute_entry_hash(
            sequence, timestamp, input_hash, output_hash, decision, metadata, previous_hash
        )

        entry = AuditEntry(
            sequence=sequence,
            timestamp=timestamp,
            input_hash=input_hash,
            output_hash=output_hash,
            decision=decision,
            metadata=metadata,
            previous_hash=previous_hash,
            entry_hash=entry_hash,
            signature=signature,
        )
        self._entries.append(entry)
        return entry

    def verify(self) -> VerificationResult:
        """Verify the integrity of the entire chain.

        Recomputes every entry_hash from scratch and checks that each
        entry's previous_hash matches the preceding entry's entry_hash.

        Returns:
            VerificationResult with status "VALID", "TAMPERED", or "EMPTY".
        """
        if not self._entries:
            return VerificationResult(
                status="EMPTY",
                entries_checked=0,
                first_invalid_sequence=None,
                message="Chain is empty.",
            )

        for i, entry in enumerate(self._entries):
            # Check entry integrity
            if not self._verify_entry_fields(entry):
                return VerificationResult(
                    status="TAMPERED",
                    entries_checked=i,
                    first_invalid_sequence=entry.sequence,
                    message=f"Entry {entry.sequence} has been tampered with.",
                )
            # Check chain link
            expected_prev = self._entries[i - 1].entry_hash if i > 0 else ""
            if entry.previous_hash != expected_prev:
                return VerificationResult(
                    status="TAMPERED",
                    entries_checked=i,
                    first_invalid_sequence=entry.sequence,
                    message=f"Entry {entry.sequence} has a broken chain link.",
                )

        return VerificationResult(
            status="VALID",
            entries_checked=len(self._entries),
            first_invalid_sequence=None,
            message=f"All {len(self._entries)} entries verified.",
        )

    def verify_entry(self, sequence: int) -> bool:
        """Verify a single entry by its sequence number.

        Args:
            sequence: Zero-based index of the entry to verify.

        Returns:
            True if the entry's entry_hash is intact; False if tampered.

        Raises:
            IndexError: If sequence is negative or >= chain length.
        """
        if sequence < 0 or sequence >= len(self._entries):
            raise IndexError(
                f"Sequence {sequence} is out of range (chain length: {len(self._entries)})."
            )
        return self._verify_entry_fields(self._entries[sequence])

    def get_entry(self, sequence: int) -> AuditEntry:
        """Return the entry at the given sequence number.

        Args:
            sequence: Zero-based index.

        Returns:
            The AuditEntry at that position.

        Raises:
            IndexError: If sequence is negative or >= chain length.
        """
        if sequence < 0 or sequence >= len(self._entries):
            raise IndexError(
                f"Sequence {sequence} is out of range (chain length: {len(self._entries)})."
            )
        return self._entries[sequence]

    @property
    def entries(self) -> list[AuditEntry]:
        """Return a copy of the entries list."""
        return list(self._entries)

    @property
    def length(self) -> int:
        """Number of entries in the chain."""
        return len(self._entries)

    def export_json(self) -> str:
        """Serialise the entire chain to a JSON string.

        The exported JSON contains every field needed to reconstruct
        the chain, including entry_hash and signature.  Use from_json()
        to reimport and verify.

        Returns:
            Indented JSON string.
        """
        payload = {
            "entries": [
                {
                    "sequence": e.sequence,
                    "timestamp": e.timestamp,
                    "input_hash": e.input_hash,
                    "output_hash": e.output_hash,
                    "decision": e.decision,
                    "metadata": e.metadata,
                    "previous_hash": e.previous_hash,
                    "entry_hash": e.entry_hash,
                    "signature": e.signature,
                }
                for e in self._entries
            ]
        }
        return json.dumps(payload, indent=2)

    @classmethod
    def from_json(
        cls,
        json_str: str,
        signing_key: str | None = None,
    ) -> AuditChain:
        """Reconstruct an AuditChain from a previously exported JSON string.

        The reconstructed chain's signing_key determines how verify() will
        validate entry hashes.  Providing the wrong key (or no key when one
        was used originally) causes every entry_hash to mismatch, and
        verify() returns TAMPERED.

        Args:
            json_str:    JSON string produced by export_json().
            signing_key: Optional key to use for verification.  Must match
                         the key used when the chain was originally created.

        Returns:
            A new AuditChain containing the imported entries.

        Raises:
            ValueError: If json_str is not valid JSON.
        """
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON: {exc}") from exc

        chain = cls(signing_key=signing_key)
        for entry_data in data["entries"]:
            entry = AuditEntry(
                sequence=entry_data["sequence"],
                timestamp=entry_data["timestamp"],
                input_hash=entry_data["input_hash"],
                output_hash=entry_data["output_hash"],
                decision=entry_data["decision"],
                metadata=entry_data["metadata"],
                previous_hash=entry_data["previous_hash"],
                entry_hash=entry_data["entry_hash"],
                signature=entry_data.get("signature", ""),
            )
            chain._entries.append(entry)
        return chain


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------


def main() -> None:  # pragma: no cover
    """Walk through the full four-step cryptographic audit narrative.

    Step 1: Record entries — build a three-entry chain of AI decisions.
    Step 2: Verify       — confirm all entries are intact (VALID).
    Step 3: Tamper       — modify a decision in-place; verify catches it.
    Step 4: Persistence  — export, reimport, verify the original chain.
    """
    print("Layer 08: Cryptographic Audit Trails")
    print("=" * 50)
    print()

    # -----------------------------------------------------------------------
    # Step 1 -- Recording Entries
    # -----------------------------------------------------------------------
    print("Step 1 -- Recording Entries")
    print("-" * 30)

    chain = AuditChain()
    chain.record(
        input_data={"text": "Loan application #1042, income: £45k, term: 5yr"},
        output_data={"risk_score": 0.12, "band": "LOW"},
        decision="APPROVED",
        metadata={"model_version": "v2.3.1", "threshold": 0.35},
    )
    e1 = chain.record(
        input_data={"text": "Loan application #1043, income: £18k, term: 10yr"},
        output_data={"risk_score": 0.71, "band": "HIGH"},
        decision="REJECTED",
        metadata={"model_version": "v2.3.1", "threshold": 0.35},
    )
    chain.record(
        input_data={"text": "Loan application #1044, income: £32k, term: 3yr"},
        output_data={"risk_score": 0.28, "band": "LOW"},
        decision="APPROVED",
        metadata={"model_version": "v2.3.1", "threshold": 0.35},
    )

    for entry in chain.entries:
        print(
            f"  Entry {entry.sequence}: {entry.decision:<10} "
            f"hash={entry.entry_hash[:16]}..."
        )
    print()

    # -----------------------------------------------------------------------
    # Step 2 -- Verification
    # -----------------------------------------------------------------------
    print("Step 2 -- Verification")
    print("-" * 23)

    result = chain.verify()
    print(f"Chain status: {result.status}")
    print(f"  {result.message}")
    print()

    # -----------------------------------------------------------------------
    # Step 3 -- Tampering Detection
    # -----------------------------------------------------------------------
    print("Step 3 -- Tampering Detection")
    print("-" * 30)
    print(f"  Original entry 1 decision: {e1.decision}")
    print("  Simulating tampering: changing decision from REJECTED to APPROVED...")
    print()

    # AuditEntry is frozen — reconstruct with the modified field and replace
    # the entry in the internal list.  The stored entry_hash was computed
    # from "REJECTED"; changing the decision makes it impossible to reproduce,
    # so verify() catches the modification.
    tampered_entry = AuditEntry(
        sequence=e1.sequence,
        timestamp=e1.timestamp,
        input_hash=e1.input_hash,
        output_hash=e1.output_hash,
        decision="APPROVED",          # <-- tampered
        metadata=e1.metadata,
        previous_hash=e1.previous_hash,
        entry_hash=e1.entry_hash,     # original hash — no longer valid
        signature=e1.signature,
    )
    chain._entries[1] = tampered_entry  # type: ignore[assignment]

    tampered_result = chain.verify()
    print(f"Chain status after tampering: {tampered_result.status}")
    print(f"  {tampered_result.message}")
    print(f"  First invalid sequence: {tampered_result.first_invalid_sequence}")
    print()

    # -----------------------------------------------------------------------
    # Step 4 -- Persistence: Export / Import
    # -----------------------------------------------------------------------
    print("Step 4 -- Persistence: Export / Import")
    print("-" * 39)

    # Use a fresh, untampered chain for the round-trip demonstration.
    clean_chain = AuditChain()
    clean_chain.record(
        input_data={"text": "Loan application #1042, income: £45k, term: 5yr"},
        output_data={"risk_score": 0.12, "band": "LOW"},
        decision="APPROVED",
    )
    clean_chain.record(
        input_data={"text": "Loan application #1043, income: £18k, term: 10yr"},
        output_data={"risk_score": 0.71, "band": "HIGH"},
        decision="REJECTED",
    )

    exported = clean_chain.export_json()
    print(f"  Exported chain ({len(exported)} bytes).")

    reimported = AuditChain.from_json(exported)
    reimport_result = reimported.verify()
    print(f"  Chain status after reimport: {reimport_result.status}")
    print(f"  {reimport_result.message}")
    print()

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("Summary:")
    print(f"  {'Step':<30}  {'Status':>8}")
    print(f"  {'-'*30}  {'-'*8}")
    print(f"  {'Original chain':<30}  {result.status:>8}")
    print(f"  {'After tampering':<30}  {tampered_result.status:>8}")
    print(f"  {'After export/import':<30}  {reimport_result.status:>8}")
    print()
    print("See: layers/08-cryptographic-audit.md")


if __name__ == "__main__":  # pragma: no cover
    main()
