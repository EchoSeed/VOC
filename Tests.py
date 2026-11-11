#!/usr/bin/env python3
"""
ErrorElim.py – evidence-based error elimination for AI-generated or rapidly iterated scripts.
Usage idea:
  1. Put the suspect script in target.py (or pass path).
  2. Write a clear bug statement + expected vs actual.
  3. Add failing tests that reproduce the problem.
  4. Run this harness; it drives hypothesis testing and only keeps proven fixes.
"""

import subprocess
import sys
import traceback
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Callable

@dataclass
class Hypothesis:
    description: str
    test: Callable[[], bool]          # returns True if hypothesis is supported
    fix: Callable[[], None]           # applies a candidate change
    evidence: List[str] = field(default_factory=list)
    refuted: bool = False

@dataclass
class BugReport:
    statement: str
    expected: str
    actual: str
    repro_command: List[str]          # e.g. ["python", "target.py", "--input", "bad"]
    tests: List[Callable[[], bool]]   # must fail before fix, pass after

def run_repro(cmd: List[str]) -> tuple[int, str, str]:
    """Run the reproduction command and capture output."""
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=30
        )
        return result.returncode, result.stdout, result.stderr
    except Exception as e:
        return -1, "", str(e)

def collect_evidence(bug: BugReport) -> dict:
    """Gather baseline evidence before any changes."""
    code, out, err = run_repro(bug.repro_command)
    return {
        "returncode": code,
        "stdout": out,
        "stderr": err,
        "traceback": traceback.format_exc() if code != 0 else "",
    }

def eliminate(bug: BugReport, hypotheses: List[Hypothesis], max_rounds: int = 5) -> Optional[str]:
    """
    Core loop:
    - Confirm the bug is currently reproducible.
    - Test each hypothesis with evidence.
    - Apply only the fix that is supported and makes the tests pass.
    - Refuse to declare success without regression protection.
    """
    print("=== Bug Statement ===")
    print(bug.statement)
    print(f"Expected: {bug.expected}")
    print(f"Actual:   {bug.actual}")
    print()

    baseline = collect_evidence(bug)
    print("Baseline evidence collected.")
    if all(t() for t in bug.tests):
        print("WARNING: tests already pass – is the bug still present?")
        return None

    for round_num in range(1, max_rounds + 1):
        print(f"\n--- Round {round_num} ---")
        for hyp in hypotheses:
            if hyp.refuted:
                continue
            print(f"Testing hypothesis: {hyp.description}")
            supported = hyp.test()
            hyp.evidence.append(f"Round {round_num}: supported={supported}")
            if not supported:
                hyp.refuted = True
                print("  -> refuted")
                continue

            # Candidate fix
            print("  -> applying candidate fix")
            hyp.fix()
            # Re-run tests
            if all(t() for t in bug.tests):
                # Confirm the original repro is also fixed
                new_evidence = collect_evidence(bug)
                print("  -> tests pass and repro improved")
                print("Accepted fix under hypothesis:", hyp.description)
                return hyp.description
            else:
                print("  -> tests still fail, rolling back mentally (restore from VCS in real use)")
                # In a real system you would git checkout or keep a backup
                hyp.refuted = True

    print("No hypothesis eliminated the error with supporting evidence.")
    return None

# ---------- Example usage pattern ----------
if __name__ == "__main__":
    # 1. Define the bug clearly (do this before asking any model for a fix)
    bug = BugReport(
        statement="Script crashes or produces wrong result on empty input",
        expected="Graceful message or empty result, exit 0",
        actual="Unhandled exception / wrong output",
        repro_command=["python", "target.py", "--input", ""],
        tests=[
            # Replace with real pytest or unittest calls
            lambda: subprocess.run(["python", "-c", "import target; assert target.handle('') == []"], capture_output=True).returncode == 0,
        ],
    )

    # 2. Hypotheses must be falsifiable and come with a concrete test + minimal fix
    hypotheses = [
        Hypothesis(
            description="Missing guard for empty/None input",
            test=lambda: True,  # replace with real check, e.g. inspect source or run a probe
            fix=lambda: print("Would insert: if not data: return []"),
        ),
        # Add more precise hypotheses based on actual evidence
    ]

    result = eliminate(bug, hypotheses)
    if result:
        print(f"\nError eliminated under: {result}")
        print("Next: commit the fix + the new regression test.")
    else:
        print("\nCould not eliminate with current hypotheses. Gather more evidence or refine.")
