# Assignment 2 — Software Fault-Tolerant Design (AITU, Fault Tolerance 2026–2027)

    python payment_processor.py --mode all        # runs baseline, exceptions, retry, checkpoint -> logs/
    python payment_processor.py --mode checkpoint # a single mode
    python -m pytest test_processor.py -v         # 8 unit tests

Outputs (all run-generated) in `logs/`:
- `<mode>.log` – human-readable log, `<mode>.jsonl` – structured JSON-lines log
- `<mode>_attempts.csv` – attempt log (Part C), `<mode>_summary.json` – metrics
- `checkpoints/CP1..CP3.json` – checkpoint snapshots (Part D), `comparison.json` – Part E
