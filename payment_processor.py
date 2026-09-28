"""
Assignment 2 - Software Fault-Tolerant Design
AITU Student Scholarship and Tuition Payment Processing

One file, four execution modes that build on each other:

    baseline    - Part A: no fault tolerance, the program stops at the first failure
    exceptions  - Part B: try/except + failure classification + structured log + continue
    retry       - Part C: exact retry policy from the assignment (max 3 attempts)
    checkpoint  - Part D: checkpoint every 5 successes, rollback on database failure

Every transaction goes through the pipeline  Validate -> Process -> Record.
    Validate : business checks (id format, positive amount)
    Process  : call the payment gateway (debit)     <- Network / Timeout faults live here
    Record   : write the result to the ledger DB    <- Database faults live here

Run:
    python payment_processor.py --mode all          # runs the four modes, writes logs/
    python payment_processor.py --mode checkpoint   # one mode
    python -m pytest test_processor.py -q           # unit tests
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional


def now_iso() -> str:
    """Local wall-clock time with UTC offset, millisecond precision."""
    return datetime.now().astimezone().isoformat(timespec="milliseconds")

# --------------------------------------------------------------------------- #
# 1. Input data (exactly as given in the assignment)
# --------------------------------------------------------------------------- #

NONE, NETWORK, TIMEOUT, DATABASE = "None", "Network", "Timeout", "Database"


@dataclass(frozen=True)
class Transaction:
    id: str
    amount: int          # KZT
    failure: str         # None | Network | Timeout | Database


TRANSACTIONS: list[Transaction] = [
    Transaction("T001", 12000, NONE),
    Transaction("T002", 25000, NETWORK),
    Transaction("T003", 8000, NONE),
    Transaction("T004", 45000, TIMEOUT),
    Transaction("T005", 13000, NONE),
    Transaction("T006", 70000, DATABASE),
    Transaction("T007", 9000, NONE),
    Transaction("T008", 31000, NETWORK),
    Transaction("T009", 15000, NONE),
    Transaction("T010", 50000, TIMEOUT),
    Transaction("T011", 6000, NONE),
    Transaction("T012", 80000, DATABASE),
    Transaction("T013", 11000, NONE),
    Transaction("T014", 22000, NONE),
    Transaction("T015", 40000, NETWORK),
]

# Fault rules table:  failure type -> outcome of attempt 1, 2, 3  (True = the attempt fails)
FAULT_RULES: dict[str, tuple[bool, ...]] = {
    NONE:     (False, False, False),
    NETWORK:  (True, False, False),      # initial fail, retry 1 success
    TIMEOUT:  (True, True, False),       # initial fail, retry 1 fail, retry 2 success
    DATABASE: (True, True, True),        # all three fail -> rollback
}
MAX_ATTEMPTS = 3          # initial attempt + 2 retries, exactly as in the rules table
CHECKPOINT_EVERY = 5      # Part D: checkpoint after every 5 successful transactions
BACKOFF_BASE_S = 0.05     # exponential back-off between retries (0.05, 0.10 s)

# --------------------------------------------------------------------------- #
# 2. Exception hierarchy - "failure classification"
# --------------------------------------------------------------------------- #


class TransactionError(Exception):
    """Base class. `kind` is the failure class used in logs and tables."""
    kind = "Unknown"
    retryable = False


class TransientError(TransactionError):
    """Errors that are expected to disappear if we simply try again."""
    retryable = True


class NetworkError(TransientError):
    kind = NETWORK          # connection refused / reset before the request was sent


class GatewayTimeoutError(TransientError):
    kind = TIMEOUT          # request may have reached the gateway, response was lost


class DatabaseError(TransactionError):
    kind = DATABASE         # ledger write failed; retried, then rolled back
    retryable = True        # the rules table retries it twice before rollback


class ValidationError(TransactionError):
    kind = "Validation"     # business rule broken - never retried


# --------------------------------------------------------------------------- #
# 3. Simulated infrastructure: fault injector, payment gateway, ledger DB
# --------------------------------------------------------------------------- #


class FaultInjector:
    """Deterministic fault injection driven by FAULT_RULES and the attempt number."""

    def should_fail(self, tx: Transaction, attempt: int) -> bool:
        rules = FAULT_RULES[tx.failure]
        return attempt <= len(rules) and rules[attempt - 1]


class PaymentGateway:
    """External bank/payment API. Debits are idempotent by key (= transaction id)."""

    def __init__(self) -> None:
        self.receipts: dict[str, dict] = {}     # idempotency key -> receipt
        self.debit_calls = 0
        self.reversals: list[str] = []

    def debit(self, key: str, amount: int) -> tuple[dict, bool]:
        """Return (receipt, replayed). A second call with the same key never
        charges the student twice - it returns the stored receipt instead."""
        self.debit_calls += 1
        if key in self.receipts:
            return self.receipts[key], True
        receipt = {"key": key, "amount": amount, "ref": f"GW-{key}-{len(self.receipts) + 1:03d}"}
        self.receipts[key] = receipt
        return receipt, False

    def reverse(self, key: str) -> Optional[dict]:
        """Compensating action used by rollback: undo a debit that was never recorded."""
        receipt = self.receipts.pop(key, None)
        if receipt:
            self.reversals.append(key)
        return receipt


@dataclass
class LedgerState:
    """The in-memory 'database' + everything a checkpoint must capture."""
    ledger: dict[str, int] = field(default_factory=dict)   # committed: tx id -> amount
    pending: dict[str, int] = field(default_factory=dict)  # debited at gateway, not yet recorded
    successes: int = 0

    @property
    def total_amount(self) -> int:
        return sum(self.ledger.values())

    def record(self, tx: Transaction) -> bool:
        """Idempotent write: returns False if the transaction was already recorded."""
        if tx.id in self.ledger:
            return False
        self.ledger[tx.id] = tx.amount
        self.pending.pop(tx.id, None)
        self.successes += 1
        return True


# --------------------------------------------------------------------------- #
# 4. Structured logging (text log + JSON-lines log + CSV attempt log)
# --------------------------------------------------------------------------- #


class RunLogger:
    def __init__(self, mode: str, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        self.mode = mode
        self.step = 0
        self.text_path = out_dir / f"{mode}.log"
        self.jsonl_path = out_dir / f"{mode}.jsonl"
        self.attempts_path = out_dir / f"{mode}_attempts.csv"
        for p in (self.text_path, self.jsonl_path, self.attempts_path):
            p.unlink(missing_ok=True)

        self.log = logging.getLogger(f"processor.{mode}")
        self.log.handlers.clear()
        self.log.setLevel(logging.INFO)
        self.log.propagate = False
        fmt = logging.Formatter("%(asctime)s.%(msecs)03d | %(levelname)-7s | %(message)s", "%H:%M:%S")
        fh = logging.FileHandler(self.text_path, encoding="utf-8")
        fh.setFormatter(fmt)
        self.log.addHandler(fh)
        sh = logging.StreamHandler()
        sh.setFormatter(logging.Formatter("  %(message)s"))
        self.log.addHandler(sh)

        self._attempts = open(self.attempts_path, "w", newline="", encoding="utf-8")
        self._csv = csv.writer(self._attempts)
        self._csv.writerow(["step", "timestamp", "transaction", "attempt", "failure", "action", "result"])

    def event(self, level: str, msg: str, **data) -> None:
        self.step += 1
        getattr(self.log, level)(msg)
        rec = {"step": self.step, "ts": now_iso(),
               "mode": self.mode, "msg": msg, **data}
        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def attempt(self, tx: Transaction, attempt: int, action: str, result: str) -> None:
        self.step += 1
        self._csv.writerow([self.step, now_iso()[11:23], tx.id, attempt, tx.failure, action, result])
        self._attempts.flush()

    def close(self) -> None:
        self._attempts.close()
        for h in self.log.handlers:
            h.close()


# --------------------------------------------------------------------------- #
# 5. The processor
# --------------------------------------------------------------------------- #

MODES = ("baseline", "exceptions", "retry", "checkpoint")


class PaymentProcessor:
    def __init__(self, mode: str, transactions: list[Transaction], out_dir: Path,
                 backoff_base: float = BACKOFF_BASE_S) -> None:
        assert mode in MODES, mode
        self.mode = mode
        self.txs = transactions
        self.out_dir = out_dir
        self.backoff_base = backoff_base
        self.injector = FaultInjector()
        self.gateway = PaymentGateway()
        self.state = LedgerState()
        self.logger = RunLogger(mode, out_dir)

        # Part D state
        self.checkpoints: list[dict] = []
        self.last_checkpoint: Optional[LedgerState] = None
        self.journal: list[Transaction] = []        # commits since the last checkpoint (redo log)

        # metrics
        self.status: dict[str, str] = {}            # tx id -> SUCCESS | FAILED | ROLLED_BACK | NOT_ATTEMPTED
        self.exceptions: dict[str, str] = {}        # tx id -> exception class name
        self.recovery: dict[str, str] = {}          # tx id -> recovery action text
        self.retries = 0
        self.rollbacks = 0
        self.attempted = 0

    # ---- pipeline steps ---------------------------------------------------- #

    def validate(self, tx: Transaction) -> None:
        if not (tx.id.startswith("T") and tx.id[1:].isdigit()):
            raise ValidationError(f"{tx.id}: malformed transaction id")
        if tx.amount <= 0:
            raise ValidationError(f"{tx.id}: amount must be positive, got {tx.amount}")

    def process(self, tx: Transaction, attempt: int) -> dict:
        fail = self.injector.should_fail(tx, attempt)
        if tx.failure == NETWORK and fail:
            # connection refused: the request never left our side, nothing was charged
            raise NetworkError(f"{tx.id}: connection refused by payment gateway")
        receipt, replayed = self.gateway.debit(key=tx.id, amount=tx.amount)
        self.state.pending[tx.id] = tx.amount
        if replayed:
            self.logger.event("info", f"{tx.id} attempt {attempt}: gateway replayed stored receipt "
                              f"{receipt['ref']} (idempotency key {tx.id}) - no double charge",
                              tx=tx.id, attempt=attempt, idempotent_replay=True)
        if tx.failure == TIMEOUT and fail:
            # the gateway applied the debit but the response was lost on the way back
            raise GatewayTimeoutError(f"{tx.id}: gateway response timed out after debit")
        return receipt

    def record(self, tx: Transaction, attempt: int) -> None:
        if tx.failure == DATABASE and self.injector.should_fail(tx, attempt):
            raise DatabaseError(f"{tx.id}: ledger write failed (database unavailable)")
        if not self.state.record(tx):
            self.logger.event("info", f"{tx.id}: already recorded - idempotent write skipped", tx=tx.id)

    def run_pipeline(self, tx: Transaction, attempt: int) -> None:
        self.validate(tx)
        self.process(tx, attempt)
        self.record(tx, attempt)

    # ---- Part A: baseline --------------------------------------------------- #

    def _run_baseline(self) -> None:
        for tx in self.txs:
            self.attempted += 1
            self.run_pipeline(tx, attempt=1)         # no handler: first exception ends the program
            self.status[tx.id] = "SUCCESS"
            self.logger.attempt(tx, 1, "initial attempt", "success")
            self.logger.event("info", f"{tx.id} SUCCESS amount={tx.amount}", tx=tx.id, status="SUCCESS")

    # ---- Part B: exception handling ----------------------------------------- #

    def _run_exceptions(self) -> None:
        for tx in self.txs:
            self.attempted += 1
            try:
                self.run_pipeline(tx, attempt=1)
                self.status[tx.id] = "SUCCESS"
                self.logger.attempt(tx, 1, "initial attempt", "success")
                self.logger.event("info", f"{tx.id} SUCCESS amount={tx.amount}", tx=tx.id, status="SUCCESS")
            except TransactionError as exc:           # classify, log, continue with next transaction
                self.status[tx.id] = "FAILED"
                self.exceptions[tx.id] = type(exc).__name__
                self.recovery[tx.id] = "caught, classified, logged, skipped; continue with next"
                self.logger.attempt(tx, 1, "initial attempt", f"fail:{exc.kind}")
                self.logger.event("error", f"{tx.id} FAILED [{exc.kind}] {type(exc).__name__}: {exc} "
                                  f"-> skipped, continuing", tx=tx.id, status="FAILED",
                                  failure=exc.kind, exception=type(exc).__name__)

    # ---- Part C: retry ------------------------------------------------------ #

    def _attempt_with_retry(self, tx: Transaction) -> Optional[TransactionError]:
        """Run the pipeline up to MAX_ATTEMPTS times. Returns None on success,
        or the last exception if every attempt failed."""
        last: Optional[TransactionError] = None
        for attempt in range(1, MAX_ATTEMPTS + 1):
            action = "initial attempt" if attempt == 1 else f"retry {attempt - 1}"
            try:
                self.run_pipeline(tx, attempt)
                self.logger.attempt(tx, attempt, action, "success")
                self.logger.event("info", f"{tx.id} attempt {attempt} ({action}): success",
                                  tx=tx.id, attempt=attempt, result="success")
                return None
            except TransactionError as exc:
                last = exc
                self.exceptions[tx.id] = type(exc).__name__
                self.logger.attempt(tx, attempt, action, f"fail:{exc.kind}")
                if not exc.retryable or attempt == MAX_ATTEMPTS:
                    self.logger.event("error", f"{tx.id} attempt {attempt} ({action}): {exc.kind} - "
                                      f"{'not retryable' if not exc.retryable else 'retries exhausted'}",
                                      tx=tx.id, attempt=attempt, failure=exc.kind, result="fail")
                    return last
                delay = self.backoff_base * 2 ** (attempt - 1)
                self.retries += 1
                self.logger.event("warning", f"{tx.id} attempt {attempt} ({action}): {exc.kind} - "
                                  f"{exc}; retrying in {delay:.2f}s", tx=tx.id, attempt=attempt,
                                  failure=exc.kind, result="fail", backoff_s=delay)
                time.sleep(delay)
        return last

    def _run_retry(self) -> None:
        for tx in self.txs:
            self.attempted += 1
            exc = self._attempt_with_retry(tx)
            if exc is None:
                self.status[tx.id] = "SUCCESS"
                self.recovery[tx.id] = self._recovery_text(tx)
                self.logger.event("info", f"{tx.id} SUCCESS amount={tx.amount}", tx=tx.id, status="SUCCESS")
            else:
                self.status[tx.id] = "FAILED"
                self.recovery[tx.id] = f"{MAX_ATTEMPTS} attempts failed; rollback required (Part D)"
                self.logger.event("error", f"{tx.id} FAILED after {MAX_ATTEMPTS} attempts [{exc.kind}] "
                                  f"- pending debit {tx.amount} KZT left unreconciled",
                                  tx=tx.id, status="FAILED", failure=exc.kind)

    # ---- Part D: checkpoint + rollback -------------------------------------- #

    def _create_checkpoint(self, label: str, position: int, reason: str) -> None:
        snapshot = copy.deepcopy(self.state)
        self.last_checkpoint = snapshot
        self.journal.clear()
        cp = {"label": label, "position": position, "reason": reason,
              "successful_transactions": snapshot.successes, "total_amount": snapshot.total_amount,
              "ledger": snapshot.ledger, "pending": snapshot.pending,
              "gateway_receipts": sorted(self.gateway.receipts),
              "saved_at": now_iso()}
        path = self.out_dir / "checkpoints" / f"{label}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(cp, indent=2), encoding="utf-8")
        cp["file"] = path.as_posix()
        self.checkpoints.append(cp)
        self.logger.event("info", f"CHECKPOINT {label}: {snapshot.successes} successful tx, "
                          f"{snapshot.total_amount} KZT, ledger={sorted(snapshot.ledger)} -> {cp['file']}",
                          checkpoint=label, successes=snapshot.successes, total_amount=snapshot.total_amount)

    def _rollback(self, tx: Transaction, exc: TransactionError) -> None:
        self.rollbacks += 1
        label =self.checkpoints[-1]["label"] if self.checkpoints else "CP0 (initial empty state)"
        discarded = dict(self.state.pending)

        # 1. compensating reversal at the gateway for the money that was debited but never recorded
        for key in discarded:
            receipt = self.gateway.reverse(key)
            if receipt:
                self.logger.event("warning", f"ROLLBACK {tx.id}: reversed gateway debit {receipt['ref']} "
                                  f"({receipt['amount']} KZT) - compensating action", tx=tx.id, reversed=key)
        # 2. restore the last valid checkpoint (or the initial empty state)
        self.state = copy.deepcopy(self.last_checkpoint) if self.last_checkpoint else LedgerState()
        # 3. redo the commits made after that checkpoint (idempotent, from the journal)
        for done in self.journal:
            self.state.record(done)
        self.logger.event("warning", f"ROLLBACK {tx.id} [{exc.kind}]: restored {label}; discarded in-flight "
                          f"{discarded}; redo {len(self.journal)} journaled commit(s) "
                          f"{[t.id for t in self.journal]}; state now {self.state.successes} tx / "
                          f"{self.state.total_amount} KZT", tx=tx.id, rollback_to=label,
                          discarded=discarded, redo=[t.id for t in self.journal])

    def _run_checkpoint(self) -> None:
        for i, tx in enumerate(self.txs):
            self.attempted += 1
            exc = self._attempt_with_retry(tx)
            if exc is None:
                self.status[tx.id] = "SUCCESS"
                self.recovery[tx.id] = self._recovery_text(tx)
                self.journal.append(tx)
                self.logger.event("info", f"{tx.id} SUCCESS amount={tx.amount} "
                                  f"(ledger {self.state.successes} tx / {self.state.total_amount} KZT)",
                                  tx=tx.id, status="SUCCESS")
                if self.state.successes % CHECKPOINT_EVERY == 0:
                    self._create_checkpoint(f"CP{len(self.checkpoints) + 1}", i + 1,
                                            f"every {CHECKPOINT_EVERY} successes")
            elif isinstance(exc, DatabaseError):
                self._rollback(tx, exc)
                self.status[tx.id] = "ROLLED_BACK"
                self.recovery[tx.id] = (f"{MAX_ATTEMPTS} attempts failed -> rollback to "
                                        f"{self.checkpoints[-1]['label'] if self.checkpoints else 'CP0'}, "
                                        f"debit reversed, continue")
                self.logger.event("error", f"{tx.id} ROLLED_BACK amount={tx.amount} - continuing with next",
                                  tx=tx.id, status="ROLLED_BACK")
            else:
                self.status[tx.id] = "FAILED"
                self.recovery[tx.id] = f"{MAX_ATTEMPTS} attempts failed, skipped"
                self.logger.event("error", f"{tx.id} FAILED [{exc.kind}] - skipped", tx=tx.id, status="FAILED")
        # end-of-batch checkpoint so the last partial group (< 5 successes) is durable too
        if self.journal:
            self._create_checkpoint(f"CP{len(self.checkpoints) + 1}", len(self.txs), "end of batch (flush)")

    # ---- driver -------------------------------------------------------------- #

    def _recovery_text(self, tx: Transaction) -> str:
        return {NONE: "none needed", NETWORK: "retry 1 succeeded",
                TIMEOUT: "retry 2 succeeded (gateway replayed receipt, no double charge)",
                DATABASE: "rollback"}[tx.failure]

    def run(self) -> dict:
        self.logger.event("info", f"=== mode={self.mode} start: {len(self.txs)} transactions, "
                          f"total {sum(t.amount for t in self.txs)} KZT ===")
        try:
            getattr(self, f"_run_{self.mode}")()
        except TransactionError as exc:
            # only the baseline gets here: no handler inside the loop, the program stops
            failed = exc.args[0].split(":")[0]
            self.status[failed] = "FAILED"
            self.exceptions[failed] = type(exc).__name__
            self.logger.attempt(next(t for t in self.txs if t.id == failed), 1, "initial attempt", f"fail:{exc.kind}")
            self.logger.event("critical", f"{failed} FAILED [{exc.kind}] {type(exc).__name__}: {exc} "
                              f"-> program stopped, remaining transactions never attempted",
                              tx=failed, status="FAILED", failure=exc.kind)
        for tx in self.txs:
            self.status.setdefault(tx.id, "NOT_ATTEMPTED")
        summary = self.summary()
        self.logger.event("info", f"=== mode={self.mode} end: {json.dumps(summary['metrics'])} ===")
        self.logger.close()
        (self.out_dir / f"{self.mode}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary

    def summary(self) -> dict:
        total = sum(t.amount for t in self.txs)
        ok = [t for t in self.txs if self.status[t.id] == "SUCCESS"]
        lost = [t for t in self.txs if self.status[t.id] != "SUCCESS"]
        processed_amount = sum(t.amount for t in ok)
        metrics = {
            "transactions_attempted": self.attempted,
            "successful_transactions": len(ok),
            "failed_transactions": sum(1 for t in self.txs if self.status[t.id] in ("FAILED", "ROLLED_BACK")),
            "not_attempted": sum(1 for t in self.txs if self.status[t.id] == "NOT_ATTEMPTED"),
            "transactions_lost": len(lost),
            "processed_amount_kzt": processed_amount,
            "lost_amount_kzt": total - processed_amount,
            "retries": self.retries,
            "rollbacks": self.rollbacks,
            "gateway_debit_calls": self.gateway.debit_calls,
            "gateway_unique_debits": len(self.gateway.receipts) + len(self.gateway.reversals),
            "gateway_reversals": len(self.gateway.reversals),
            "unreconciled_pending_kzt": sum(self.state.pending.values()),
            "ledger_total_kzt": self.state.total_amount,
            "completion_rate": round(len(ok) / len(self.txs), 4),
        }
        return {"mode": self.mode, "metrics": metrics, "status": self.status, "exceptions": self.exceptions,
                "recovery": self.recovery, "checkpoints": self.checkpoints}


# --------------------------------------------------------------------------- #
# 6. CLI
# --------------------------------------------------------------------------- #


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=MODES + ("all",), default="all")
    ap.add_argument("--out", default="logs", help="output directory for logs (default: logs/)")
    args = ap.parse_args()
    out = Path(args.out)
    modes = MODES if args.mode == "all" else (args.mode,)
    results = {}
    for mode in modes:
        print(f"\n### {mode}")
        results[mode] = PaymentProcessor(mode, TRANSACTIONS, out).run()

    if len(results) > 1:
        keys = ["successful_transactions", "failed_transactions", "not_attempted", "lost_amount_kzt",
                "retries", "rollbacks", "unreconciled_pending_kzt", "completion_rate"]
        print("\n### comparison")
        print(f"{'metric':28}" + "".join(f"{m:>12}" for m in results))
        for k in keys:
            print(f"{k:28}" + "".join(f"{results[m]['metrics'][k]:>12}" for m in results))
        (out / "comparison.json").write_text(
            json.dumps({m: r["metrics"] for m, r in results.items()}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
