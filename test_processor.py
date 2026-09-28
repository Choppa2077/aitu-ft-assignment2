"""Unit tests for payment_processor.py  (run: python -m pytest test_processor.py -q)"""
from pathlib import Path

import pytest

from payment_processor import (DATABASE, NETWORK, NONE, TIMEOUT, TRANSACTIONS, PaymentProcessor,
                               Transaction)


def run(mode, txs=TRANSACTIONS, tmp=Path("logs_test")):
    return PaymentProcessor(mode, txs, tmp / mode, backoff_base=0.0).run()


def test_baseline_stops_at_first_failure(tmp_path):
    m = run("baseline", tmp=tmp_path)["metrics"]
    assert m["transactions_attempted"] == 2
    assert m["successful_transactions"] == 1
    assert m["not_attempted"] == 13
    assert m["lost_amount_kzt"] == 425_000


def test_exceptions_mode_continues_after_every_failure(tmp_path):
    r = run("exceptions", tmp=tmp_path)
    assert r["metrics"]["transactions_attempted"] == 15
    assert r["metrics"]["successful_transactions"] == 8
    assert r["exceptions"] == {"T002": "NetworkError", "T004": "GatewayTimeoutError",
                               "T006": "DatabaseError", "T008": "NetworkError",
                               "T010": "GatewayTimeoutError", "T012": "DatabaseError",
                               "T015": "NetworkError"}


def test_retry_policy_matches_rules_table(tmp_path):
    """Network: 2 attempts, Timeout: 3 attempts, Database: 3 attempts then failure."""
    r = run("retry", tmp=tmp_path)
    rows = (tmp_path / "retry" / "retry_attempts.csv").read_text().splitlines()[1:]
    attempts = {}
    for row in rows:
        _, _, tx, attempt, *_ = row.split(",")
        attempts[tx] = max(attempts.get(tx, 0), int(attempt))
    for tx in TRANSACTIONS:
        expected = {NONE: 1, NETWORK: 2, TIMEOUT: 3, DATABASE: 3}[tx.failure]
        assert attempts[tx.id] == expected, tx
    assert r["metrics"]["retries"] == 11
    assert r["status"]["T006"] == "FAILED" and r["status"]["T012"] == "FAILED"


def test_gateway_is_idempotent_no_double_charge(tmp_path):
    m = run("checkpoint", tmp=tmp_path)["metrics"]
    assert m["gateway_debit_calls"] == 23          # 15 + 11 retries - 3 network retries that never reached it
    assert m["gateway_unique_debits"] == 15        # every transaction debited at most once


def test_checkpoints_every_5_successes_plus_final_flush(tmp_path):
    cps = run("checkpoint", tmp=tmp_path)["checkpoints"]
    assert [(c["label"], c["successful_transactions"], c["total_amount"]) for c in cps] == [
        ("CP1", 5, 103_000), ("CP2", 10, 214_000), ("CP3", 13, 287_000)]
    assert (tmp_path / "checkpoint" / "checkpoints" / "CP2.json").exists()


def test_rollback_discards_pending_and_reverses_debit(tmp_path):
    r = run("checkpoint", tmp=tmp_path)
    m = r["metrics"]
    assert m["rollbacks"] == 2
    assert m["gateway_reversals"] == 2
    assert m["unreconciled_pending_kzt"] == 0          # fails if the checkpoint is not a deep copy
    assert m["ledger_total_kzt"] == 287_000
    assert r["status"]["T006"] == "ROLLED_BACK" and r["status"]["T012"] == "ROLLED_BACK"


def test_rollback_redoes_journal_when_failure_is_mid_group(tmp_path):
    """DB failure before the first checkpoint: restore CP0 (empty) and redo the 2 journaled commits."""
    txs = [Transaction("T101", 1000, NONE), Transaction("T102", 2000, NETWORK),
           Transaction("T103", 3000, DATABASE), Transaction("T104", 4000, NONE)]
    r = run("checkpoint", txs, tmp=tmp_path)
    assert r["status"] == {"T101": "SUCCESS", "T102": "SUCCESS", "T103": "ROLLED_BACK", "T104": "SUCCESS"}
    assert r["metrics"]["ledger_total_kzt"] == 7000
    log = (tmp_path / "checkpoint" / "checkpoint.log").read_text()
    assert "redo 2 journaled commit(s) ['T101', 'T102']" in log


def test_validation_error_is_never_retried(tmp_path):
    txs = [Transaction("T201", -5, NONE), Transaction("T202", 500, NONE)]
    r = run("checkpoint", txs, tmp=tmp_path)
    assert r["status"] == {"T201": "FAILED", "T202": "SUCCESS"}
    assert r["metrics"]["retries"] == 0
