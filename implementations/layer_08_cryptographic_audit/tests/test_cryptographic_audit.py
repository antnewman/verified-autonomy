"""
Tests for Layer 08: Cryptographic Audit Trails

Test naming convention: test_[subject]_[scenario]_[expected_result]

Requirements:
- 90% minimum coverage
- Edge cases at both boundaries
- Corner cases at intersections of multiple boundaries
- Failure mode tests (correct exceptions, messages, fallbacks)
- No mocking of the system under test
"""

from __future__ import annotations

import json
import re

import pytest

from cryptographic_audit import (
    AuditChain,
    AuditEntry,
    VerificationResult,
    hash_data,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_chain(n: int = 3, signing_key: str | None = None) -> AuditChain:
    """Build a valid chain with n entries."""
    chain = AuditChain(signing_key=signing_key)
    for i in range(n):
        chain.record(
            input_data={"index": i},
            output_data={"result": i * 2},
            decision="APPROVED" if i % 2 == 0 else "REJECTED",
            metadata={"step": i},
        )
    return chain


def _tamper(chain: AuditChain, sequence: int, decision: str = "TAMPERED") -> None:
    """Replace entry at sequence with a copy that has a modified decision.

    AuditEntry is frozen — reconstruct it with the new field and write it
    directly into the internal list.
    """
    original = chain._entries[sequence]  # type: ignore[attr-defined]
    modified = AuditEntry(
        sequence=original.sequence,
        timestamp=original.timestamp,
        input_hash=original.input_hash,
        output_hash=original.output_hash,
        decision=decision,              # <-- the tampered field
        metadata=original.metadata,
        previous_hash=original.previous_hash,
        entry_hash=original.entry_hash, # original hash — now invalid
        signature=original.signature,
    )
    chain._entries[sequence] = modified  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# hash_data
# ---------------------------------------------------------------------------


class TestHashData:
    def test_returns_64_char_hex_string(self) -> None:
        result = hash_data("hello")
        assert len(result) == 64
        assert re.fullmatch(r"[0-9a-f]{64}", result)

    def test_deterministic_same_input_same_output(self) -> None:
        assert hash_data("hello") == hash_data("hello")

    def test_different_inputs_produce_different_hashes(self) -> None:
        assert hash_data("hello") != hash_data("world")

    def test_empty_string_produces_valid_hash(self) -> None:
        result = hash_data("")
        assert len(result) == 64
        assert re.fullmatch(r"[0-9a-f]{64}", result)

    def test_whitespace_sensitivity(self) -> None:
        assert hash_data("hello") != hash_data("hello ")


# ---------------------------------------------------------------------------
# AuditChain.record
# ---------------------------------------------------------------------------


class TestRecord:
    def test_first_entry_has_empty_previous_hash(self) -> None:
        chain = AuditChain()
        entry = chain.record({"x": 1}, {"y": 2}, "APPROVED")
        assert entry.previous_hash == ""

    def test_second_entry_previous_hash_equals_first_entry_hash(self) -> None:
        chain = AuditChain()
        first = chain.record({"x": 1}, {"y": 2}, "APPROVED")
        second = chain.record({"x": 3}, {"y": 4}, "REJECTED")
        assert second.previous_hash == first.entry_hash

    def test_sequence_numbers_are_sequential(self) -> None:
        chain = _make_chain(4)
        for i, entry in enumerate(chain.entries):
            assert entry.sequence == i

    def test_timestamps_are_valid_iso_8601_utc(self) -> None:
        chain = _make_chain(2)
        for entry in chain.entries:
            # ISO 8601 with UTC offset should parse; check the format
            # datetime.fromisoformat handles +00:00 suffix on Python 3.11+
            from datetime import datetime
            dt = datetime.fromisoformat(entry.timestamp)
            assert dt.tzinfo is not None

    def test_input_hash_is_sha256_hex(self) -> None:
        chain = AuditChain()
        entry = chain.record("some input", "some output", "APPROVED")
        assert len(entry.input_hash) == 64
        assert re.fullmatch(r"[0-9a-f]{64}", entry.input_hash)

    def test_output_hash_is_sha256_hex(self) -> None:
        chain = AuditChain()
        entry = chain.record("some input", "some output", "APPROVED")
        assert len(entry.output_hash) == 64
        assert re.fullmatch(r"[0-9a-f]{64}", entry.output_hash)

    def test_entry_hash_is_sha256_hex(self) -> None:
        chain = AuditChain()
        entry = chain.record({"a": 1}, {"b": 2}, "APPROVED")
        assert len(entry.entry_hash) == 64
        assert re.fullmatch(r"[0-9a-f]{64}", entry.entry_hash)

    def test_entry_hash_is_deterministic(self) -> None:
        """Same chain state produces the same entry_hash for identical fields."""
        chain_a = AuditChain()
        # We can't control timestamp, but we can verify that two entries
        # with the same content on the same chain produce deterministic hashes
        # by reconstructing the hash from the stored fields.
        entry = chain_a.record({"x": 1}, {"y": 2}, "APPROVED")
        # Re-derive the hash from the stored fields
        recomputed = hash_data(
            json.dumps(
                {
                    "sequence": entry.sequence,
                    "timestamp": entry.timestamp,
                    "input_hash": entry.input_hash,
                    "output_hash": entry.output_hash,
                    "decision": entry.decision,
                    "metadata": entry.metadata,
                    "previous_hash": entry.previous_hash,
                },
                sort_keys=True,
            )
        )
        assert entry.entry_hash == recomputed

    def test_metadata_defaults_to_empty_dict_when_none(self) -> None:
        chain = AuditChain()
        entry = chain.record({"x": 1}, {"y": 2}, "APPROVED", metadata=None)
        assert entry.metadata == {}

    def test_metadata_defaults_to_empty_dict_when_omitted(self) -> None:
        chain = AuditChain()
        entry = chain.record({"x": 1}, {"y": 2}, "APPROVED")
        assert entry.metadata == {}

    def test_different_inputs_produce_different_entry_hashes(self) -> None:
        chain = AuditChain()
        e0 = chain.record({"x": 1}, {"y": 2}, "APPROVED")
        e1 = chain.record({"x": 99}, {"y": 2}, "APPROVED")
        assert e0.entry_hash != e1.entry_hash

    def test_different_outputs_produce_different_entry_hashes(self) -> None:
        chain = AuditChain()
        e0 = chain.record({"x": 1}, {"y": 2}, "APPROVED")
        e1 = chain.record({"x": 1}, {"y": 999}, "APPROVED")
        assert e0.entry_hash != e1.entry_hash

    def test_metadata_is_stored_correctly(self) -> None:
        chain = AuditChain()
        meta = {"version": "1.0", "threshold": 0.5}
        entry = chain.record("input", "output", "APPROVED", metadata=meta)
        assert entry.metadata == meta

    def test_returns_audit_entry_instance(self) -> None:
        chain = AuditChain()
        entry = chain.record("input", "output", "APPROVED")
        assert isinstance(entry, AuditEntry)


# ---------------------------------------------------------------------------
# AuditChain.verify
# ---------------------------------------------------------------------------


class TestVerify:
    def test_empty_chain_returns_empty_status(self) -> None:
        chain = AuditChain()
        result = chain.verify()
        assert result.status == "EMPTY"

    def test_empty_chain_entries_checked_is_zero(self) -> None:
        result = AuditChain().verify()
        assert result.entries_checked == 0

    def test_empty_chain_first_invalid_sequence_is_none(self) -> None:
        result = AuditChain().verify()
        assert result.first_invalid_sequence is None

    def test_single_valid_entry_returns_valid(self) -> None:
        chain = _make_chain(1)
        assert chain.verify().status == "VALID"

    def test_multi_entry_valid_chain_returns_valid(self) -> None:
        chain = _make_chain(5)
        assert chain.verify().status == "VALID"

    def test_valid_chain_entries_checked_equals_chain_length(self) -> None:
        chain = _make_chain(4)
        result = chain.verify()
        assert result.entries_checked == chain.length

    def test_valid_chain_first_invalid_sequence_is_none(self) -> None:
        chain = _make_chain(3)
        assert chain.verify().first_invalid_sequence is None

    def test_tampered_entry_returns_tampered_status(self) -> None:
        chain = _make_chain(3)
        _tamper(chain, 1)
        assert chain.verify().status == "TAMPERED"

    def test_tampered_first_entry_detected(self) -> None:
        chain = _make_chain(3)
        _tamper(chain, 0)
        result = chain.verify()
        assert result.status == "TAMPERED"
        assert result.first_invalid_sequence == 0

    def test_tampered_last_entry_detected(self) -> None:
        chain = _make_chain(3)
        _tamper(chain, 2)
        result = chain.verify()
        assert result.status == "TAMPERED"
        assert result.first_invalid_sequence == 2

    def test_tampered_middle_entry_detected(self) -> None:
        chain = _make_chain(5)
        _tamper(chain, 2)
        result = chain.verify()
        assert result.status == "TAMPERED"
        assert result.first_invalid_sequence == 2

    def test_broken_chain_link_detected(self) -> None:
        """Modifying previous_hash on a stored entry breaks the chain link."""
        chain = _make_chain(3)
        original = chain._entries[2]  # type: ignore[attr-defined]
        broken = AuditEntry(
            sequence=original.sequence,
            timestamp=original.timestamp,
            input_hash=original.input_hash,
            output_hash=original.output_hash,
            decision=original.decision,
            metadata=original.metadata,
            previous_hash="0" * 64,   # deliberately wrong
            entry_hash=original.entry_hash,
            signature=original.signature,
        )
        chain._entries[2] = broken  # type: ignore[attr-defined]
        result = chain.verify()
        # Either the entry_hash mismatch or the chain link is detected
        assert result.status == "TAMPERED"

    def test_tampered_result_first_invalid_sequence_is_correct(self) -> None:
        chain = _make_chain(5)
        _tamper(chain, 3)
        result = chain.verify()
        assert result.first_invalid_sequence == 3

    def test_broken_chain_link_with_valid_entry_hash_detected(self) -> None:
        """An entry whose entry_hash is self-consistent but previous_hash is wrong."""
        chain = _make_chain(2)
        original = chain._entries[1]  # type: ignore[attr-defined]
        fake_prev = "a" * 64
        # Recompute entry_hash using the fake previous_hash so entry-level
        # verification passes, but the chain-link check should catch it.
        new_hash, new_sig = chain._compute_entry_hash(  # type: ignore[attr-defined]
            original.sequence,
            original.timestamp,
            original.input_hash,
            original.output_hash,
            original.decision,
            original.metadata,
            fake_prev,
        )
        broken = AuditEntry(
            sequence=original.sequence,
            timestamp=original.timestamp,
            input_hash=original.input_hash,
            output_hash=original.output_hash,
            decision=original.decision,
            metadata=original.metadata,
            previous_hash=fake_prev,
            entry_hash=new_hash,
            signature=new_sig,
        )
        chain._entries[1] = broken  # type: ignore[attr-defined]
        result = chain.verify()
        assert result.status == "TAMPERED"
        assert result.first_invalid_sequence == 1

    def test_tampered_result_has_message(self) -> None:
        chain = _make_chain(2)
        _tamper(chain, 0)
        result = chain.verify()
        assert isinstance(result.message, str)
        assert len(result.message) > 0

    def test_returns_verification_result_instance(self) -> None:
        result = _make_chain(2).verify()
        assert isinstance(result, VerificationResult)


# ---------------------------------------------------------------------------
# AuditChain.verify_entry
# ---------------------------------------------------------------------------


class TestVerifyEntry:
    def test_valid_entry_returns_true(self) -> None:
        chain = _make_chain(3)
        assert chain.verify_entry(0) is True
        assert chain.verify_entry(1) is True
        assert chain.verify_entry(2) is True

    def test_tampered_entry_returns_false(self) -> None:
        chain = _make_chain(3)
        _tamper(chain, 1)
        assert chain.verify_entry(1) is False

    def test_untampered_entries_still_valid_after_neighbour_tampered(self) -> None:
        chain = _make_chain(3)
        _tamper(chain, 1)
        # Entry 0 was not modified — its hash should still be valid
        assert chain.verify_entry(0) is True

    def test_negative_sequence_raises_index_error(self) -> None:
        chain = _make_chain(2)
        with pytest.raises(IndexError):
            chain.verify_entry(-1)

    def test_out_of_range_sequence_raises_index_error(self) -> None:
        chain = _make_chain(2)
        with pytest.raises(IndexError):
            chain.verify_entry(2)

    def test_zero_on_single_entry_chain_returns_true(self) -> None:
        chain = _make_chain(1)
        assert chain.verify_entry(0) is True


# ---------------------------------------------------------------------------
# AuditChain with signing_key
# ---------------------------------------------------------------------------


class TestSigningKey:
    def test_signed_entries_have_different_hashes_than_unsigned(self) -> None:
        signed = _make_chain(1, signing_key="secret")
        unsigned = _make_chain(1)
        # entry_hash must differ because HMAC is incorporated
        assert signed.get_entry(0).entry_hash != unsigned.get_entry(0).entry_hash

    def test_signed_chain_verifies_with_correct_key(self) -> None:
        chain = _make_chain(3, signing_key="my-secret-key")
        result = chain.verify()
        assert result.status == "VALID"

    def test_signed_entry_has_non_empty_signature(self) -> None:
        chain = _make_chain(1, signing_key="secret")
        assert chain.get_entry(0).signature != ""

    def test_unsigned_entry_has_empty_signature(self) -> None:
        chain = _make_chain(1)
        assert chain.get_entry(0).signature == ""

    def test_round_trip_with_correct_key_is_valid(self) -> None:
        chain = _make_chain(3, signing_key="correct-key")
        exported = chain.export_json()
        reimported = AuditChain.from_json(exported, signing_key="correct-key")
        assert reimported.verify().status == "VALID"

    def test_round_trip_with_wrong_key_is_tampered(self) -> None:
        chain = _make_chain(3, signing_key="correct-key")
        exported = chain.export_json()
        reimported = AuditChain.from_json(exported, signing_key="wrong-key")
        assert reimported.verify().status == "TAMPERED"

    def test_round_trip_no_key_on_signed_chain_is_tampered(self) -> None:
        """Reimporting a signed chain without the key should fail verification."""
        chain = _make_chain(2, signing_key="secret")
        exported = chain.export_json()
        reimported = AuditChain.from_json(exported, signing_key=None)
        assert reimported.verify().status == "TAMPERED"


# ---------------------------------------------------------------------------
# AuditChain.export_json / from_json
# ---------------------------------------------------------------------------


class TestExportImport:
    def test_round_trip_produces_identical_entries(self) -> None:
        chain = _make_chain(3)
        exported = chain.export_json()
        reimported = AuditChain.from_json(exported)
        assert reimported.length == chain.length
        for original, restored in zip(chain.entries, reimported.entries):
            assert original.sequence == restored.sequence
            assert original.decision == restored.decision
            assert original.entry_hash == restored.entry_hash
            assert original.previous_hash == restored.previous_hash
            assert original.input_hash == restored.input_hash
            assert original.output_hash == restored.output_hash

    def test_verification_passes_after_round_trip(self) -> None:
        chain = _make_chain(4)
        reimported = AuditChain.from_json(chain.export_json())
        assert reimported.verify().status == "VALID"

    def test_tampered_json_detected_after_reimport(self) -> None:
        chain = _make_chain(2)
        exported = chain.export_json()
        # Manually alter a decision in the JSON payload
        data = json.loads(exported)
        data["entries"][0]["decision"] = "MANIPULATED"
        tampered_json = json.dumps(data)
        reimported = AuditChain.from_json(tampered_json)
        assert reimported.verify().status == "TAMPERED"

    def test_invalid_json_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Invalid JSON"):
            AuditChain.from_json("not valid json {{{")

    def test_export_produces_valid_json(self) -> None:
        chain = _make_chain(2)
        exported = chain.export_json()
        parsed = json.loads(exported)  # must not raise
        assert "entries" in parsed
        assert len(parsed["entries"]) == 2

    def test_empty_chain_exports_and_reimports(self) -> None:
        chain = AuditChain()
        exported = chain.export_json()
        reimported = AuditChain.from_json(exported)
        assert reimported.length == 0
        assert reimported.verify().status == "EMPTY"

    def test_metadata_preserved_through_round_trip(self) -> None:
        chain = AuditChain()
        meta = {"version": "2.0", "threshold": 0.42}
        chain.record("input", "output", "APPROVED", metadata=meta)
        reimported = AuditChain.from_json(chain.export_json())
        assert reimported.get_entry(0).metadata == meta


# ---------------------------------------------------------------------------
# AuditChain.get_entry
# ---------------------------------------------------------------------------


class TestGetEntry:
    def test_valid_sequence_returns_correct_entry(self) -> None:
        chain = _make_chain(3)
        entry = chain.get_entry(1)
        assert entry.sequence == 1

    def test_negative_sequence_raises_index_error(self) -> None:
        chain = _make_chain(2)
        with pytest.raises(IndexError):
            chain.get_entry(-1)

    def test_out_of_range_sequence_raises_index_error(self) -> None:
        chain = _make_chain(2)
        with pytest.raises(IndexError):
            chain.get_entry(2)

    def test_get_entry_returns_audit_entry_instance(self) -> None:
        chain = _make_chain(1)
        assert isinstance(chain.get_entry(0), AuditEntry)

    def test_get_entry_zero_on_single_entry_chain(self) -> None:
        chain = AuditChain()
        chain.record("input", "output", "APPROVED")
        entry = chain.get_entry(0)
        assert entry.decision == "APPROVED"


# ---------------------------------------------------------------------------
# Properties: entries and length
# ---------------------------------------------------------------------------


class TestProperties:
    def test_entries_returns_list_of_audit_entry(self) -> None:
        chain = _make_chain(3)
        entries = chain.entries
        assert isinstance(entries, list)
        assert all(isinstance(e, AuditEntry) for e in entries)

    def test_entries_length_matches_number_of_records(self) -> None:
        chain = _make_chain(4)
        assert len(chain.entries) == 4

    def test_length_matches_number_of_records(self) -> None:
        chain = _make_chain(5)
        assert chain.length == 5

    def test_length_zero_for_empty_chain(self) -> None:
        assert AuditChain().length == 0

    def test_entries_is_a_copy(self) -> None:
        """Mutating the returned list must not affect the internal chain."""
        chain = _make_chain(2)
        entries = chain.entries
        entries.clear()
        assert chain.length == 2


# ---------------------------------------------------------------------------
# main() — integration smoke tests
# ---------------------------------------------------------------------------


class TestMain:
    def test_main_runs_without_error(self, capsys: pytest.CaptureFixture[str]) -> None:
        from cryptographic_audit import main
        main()  # must not raise

    def test_main_output_contains_valid(self, capsys: pytest.CaptureFixture[str]) -> None:
        from cryptographic_audit import main
        main()
        assert "VALID" in capsys.readouterr().out

    def test_main_output_contains_tampered(self, capsys: pytest.CaptureFixture[str]) -> None:
        from cryptographic_audit import main
        main()
        assert "TAMPERED" in capsys.readouterr().out

    def test_main_output_contains_cryptographic_audit(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from cryptographic_audit import main
        main()
        assert "Cryptographic Audit" in capsys.readouterr().out
