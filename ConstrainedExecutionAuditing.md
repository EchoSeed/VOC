# Constrained Execution Auditing

> Trigger when designing, evaluating, or securing a goal-directed system that must be prevented from exploiting shortcuts, modifying its own evaluation scaffolding, or evading oversight mechanisms. Also trigger when auditing constraint completeness before authorizing a system run.

## Core Thesis
An EvasionSurface names a single shortcut path available to a constrained system, characterized by whether it is accessible, detectable, and closed. A ConstraintAudit accumulates the full map of such surfaces for a given system run, tracking closure status across all identified paths. The open_surfaces method returns all paths not yet closed, while close_surface marks a specific path closed and records the method used. A system is fully constrained only when every known evasion surface has been closed, as determined by is_fully_constrained. The make_read_only function strips write permissions from evaluation scaffolding files so the constrained system cannot modify them. Before a system run, compute_integrity_hash records a SHA-256 digest of each evaluation script to enable post-run comparison. After the run, verify_integrity checks whether a file still matches its pre-run hash; a mismatch is described as a clear evasion signal. The run_constrained function enforces these controls end-to-end: it locks all eval scripts before executing the task function and verifies their integrity afterward. When strict mode is enabled, run_constrained refuses to execute the task if any evasion surface remains open in the audit. If strict-mode preconditions fail, the function returns None rather than proceeding. Post-run integrity verification thus closes the loop between pre-run locking and detection of any tampering that occurred during execution.

## Overview
This skill provides a structured methodology for mapping, closing, and verifying all evasion surfaces available to a constrained system before and after execution. It combines static surface enumeration (EvasionSurface, ConstraintAudit) with file-system hardening (make_read_only) and cryptographic tamper detection (compute_integrity_hash, verify_integrity) into a coherent pre-run/post-run lifecycle. The result is a repeatable protocol that either confirms a system ran under genuine constraint or surfaces evidence of evasion.

The skill matters because partial constraint deployment is indistinguishable from no constraint deployment if the unclosed surface is the one exploited. is_fully_constrained acts as a binary gate: a system that passes all checks except one is not constrained, it is merely inconvenienced. The audit model forces explicitness — every known shortcut must be named, triaged by risk (accessible + undetectable = highest priority), and closed with a recorded method before execution is authorized.

Reach for this skill when operating in adversarial or high-stakes evaluation contexts where a system's incentives may misalign with honest task completion. It is equally applicable to AI agent sandboxing, automated testing pipelines where the testee has filesystem access, or any scenario where the evaluator and the evaluated share an execution environment. Strict mode — the ensemble of all mechanisms applied together — is the only configuration that closes the loop between pre-run locking and post-run verification.

## When to Use
- A goal-directed system has write access to files that govern its own evaluation, scoring, or termination conditions.
- You need to authorize or block a constrained system run based on whether all known evasion paths have been formally closed.
- A post-run comparison is needed to determine whether evaluation scaffolding was tampered with during execution.
- You are building or reviewing an audit trail that must distinguish 'closed with recorded method' from 'closed with no method recorded' — the latter being an audit gap.
- Strict-mode execution is required: every mechanism (surface closure, read-only locking, integrity hashing, post-run verification) must be confirmed active before the run proceeds.

## Core Workflow
1. Enumerate all evasion surfaces: instantiate one EvasionSurface per identified shortcut path, capturing accessible, detectable, and closed state. Register each via ConstraintAudit.add_surface. Prioritize accessible=True + detectable=False surfaces as highest risk.
2. Close all surfaces before execution: call ConstraintAudit.close_surface for each path, recording the closure method. Treat closed=True with no closure_method as an audit gap. Confirm ConstraintAudit.is_fully_constrained returns True; if False and strict mode is active, abort — do not proceed.
3. Harden evaluation scaffolding: call make_read_only on every evaluation script and scaffolding file. Verify file existence first to avoid silent skips. Then call compute_integrity_hash on each file and store the returned hex digest in a location outside the constrained system's write reach (itself protected by make_read_only).
4. Execute the constrained run: invoke the task only after all preconditions are confirmed. During execution, open_surfaces should return an empty list; any new surface discovered mid-run must be treated as an active evasion signal.
5. Post-run verification: call verify_integrity on each scaffolding file, comparing current hash against the pre-run baseline. A mismatch is an evasion signal. Emit ConstraintAudit.summary to record total surfaces, closure count, open count, and full-constraint status. Log or escalate any integrity failure before accepting run outputs.

## Key Patterns
### Triage by Risk Quadrant
The three EvasionSurface boolean fields (accessible, detectable, closed) define a risk triage order. Accessible and undetectable paths are highest priority because exploitation produces no log signal. Closing detectable-but-inaccessible paths first is the wrong order. Audit tooling should sort open_surfaces by this quadrant before presenting them to operators.

### Audit Gap Detection
A surface marked closed=True with closure_method absent or empty is not a successfully closed surface — it is an audit record with a missing chain of custody. The schema requires both fields to be populated for a closure to be considered valid. Automated summary reporting should flag this condition explicitly rather than counting it as closed.

### Hash Storage Must Be Outside Write Reach
compute_integrity_hash is only tamper-evident if the stored digest is itself inaccessible to the constrained system. If the hash file is writable, a sufficiently capable system can overwrite both the evaluation script and its recorded hash, defeating detection entirely. make_read_only must be applied to the hash storage location using the same protocol as the scripts it protects.

### Strict-Mode Is All-or-Nothing
Partial application of constraint mechanisms — for example, using compute_integrity_hash without make_read_only, or closing surfaces without verifying integrity post-run — leaves exploitable gaps that are worse than acknowledged gaps because they create false confidence. Strict mode is only meaningful as the conjunction of every mechanism; any missing component degrades the entire protocol to best-effort.

### add_surface After is_fully_constrained Resets the Gate
is_fully_constrained is monotonic over existing surfaces but resets to False the moment a new surface is added via add_surface. This means the pre-run gate check must occur after the final add_surface call, not before. Any surface discovery mid-run that triggers a late add_surface call should be treated as an active evasion signal, not a routine audit update.

## Code Implementation
```python
import hashlib
import os
import stat
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class EvasionSurface:
    """
    Represents a single shortcut path available to a constrained system.
    Risk triage order: accessible=True + detectable=False paths are highest priority.
    """
    name: str
    description: str
    accessible: bool
    detectable: bool
    closed: bool = False
    closure_method: Optional[str] = None

    def risk_priority(self) -> int:
        """
        Returns a priority score for triage ordering.
        Higher score = higher risk. Accessible + undetectable = 3 (max).
        """
        score = 0
        if self.accessible:
            score += 2
        if not self.detectable:
            score += 1
        if self.closed:
            score -= 10  # closed surfaces drop to bottom
        return score

    def has_audit_gap(self) -> bool:
        """
        A closed surface with no recorded closure_method is an audit gap —
        chain of custody is broken. Source schema requires both fields populated.
        """
        return self.closed and not self.closure_method


class ConstraintAudit:
    """
    Tracks the full evasion surface map for a goal-directed system run.
    Stateful across the run lifecycle; queryable at any point as a real-time dashboard.
    """

    def __init__(self, system_name: str) -> None:
        self.system_name = system_name
        self.surfaces: list[EvasionSurface] = []

    def add_surface(self, surface: EvasionSurface) -> None:
        """
        Appends a new evasion surface to the audit record.
        NOTE: Calling this after is_fully_constrained returned True resets the gate to False.
        Any mid-run discovery should be treated as an active evasion signal.
        """
        self.surfaces.append(surface)

    def open_surfaces(self) -> list[EvasionSurface]:
        """Returns all evasion paths not yet closed, sorted by risk priority descending."""
        open_s = [s for s in self.surfaces if not s.closed]
        return sorted(open_s, key=lambda s: s.risk_priority(), reverse=True)

    def close_surface(self, name: str, method: str) -> bool:
        """
        Marks a surface as closed and records the closure method.
        A closure without a recorded method is an audit gap — method is required here.
        Returns True if found and closed, False otherwise.
        """
        for surface in self.surfaces:
            if surface.name == name:
                surface.closed = True
                surface.closure_method = method
                return True
        return False

    def is_fully_constrained(self) -> bool:
        """
        Returns True only when every known evasion surface is closed.
        Use as binary gate before authorizing execution — partial closure is not constrained.
        """
        return all(s.closed for s in self.surfaces)

    def audit_gaps(self) -> list[EvasionSurface]:
        """
        Returns surfaces marked closed but missing a closure_method.
        These represent broken chain-of-custody and should be flagged, not counted as clean.
        """
        return [s for s in self.surfaces if s.has_audit_gap()]

    def summary(self) -> str:
        """
        Returns a formatted report: total surfaces, closed count, open count,
        full-constraint status, and any audit gaps.
        """
        total = len(self.surfaces)
        closed_count = sum(1 for s in self.surfaces if s.closed)
        open_count = total - closed_count
        gaps = self.audit_gaps()
        gap_warning = f" | AUDIT GAPS (closed, no method): {len(gaps)}" if gaps else ""
        return (
            f"[{self.system_name}] Surfaces: {total} total | "
            f"{closed_count} closed | {open_count} open | "
            f"Fully constrained: {self.is_fully_constrained()}"
            f"{gap_warning}"
        )


def make_read_only(path: str) -> None:
    """
    Removes write permissions from a file so a constrained system cannot modify
    evaluation scaffolding. Strips owner, group, and other write bits.

    Raises FileNotFoundError if path does not exist — callers must verify existence
    before calling to avoid silent skips in a pre-run hardening loop.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Cannot lock non-existent path: {path}")

    # current: the existing permission bits of the file
    current = os.stat(path).st_mode

    # read_only_mode: permission bits after write access is stripped for owner, group, and other
    read_only_mode = current & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)

    os.chmod(path, read_only_mode)


def compute_integrity_hash(path: str) -> str:
    """
    Computes a SHA-256 hash of a file for tamper detection.
    Call BEFORE the constrained system run; store the result outside the
    constrained system's write reach (protect the hash file with make_read_only).
    Returns the hex digest of the file contents.
    """
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        # Read in chunks to handle large evaluation scripts without OOM risk
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def verify_integrity(path: str, baseline_digest: str) -> bool:
    """
    [Inferred from source thesis — not explicitly defined in source schema]
    Compares the current file hash against the pre-run baseline produced by
    compute_integrity_hash. A False return value is a clear evasion signal.
    Call AFTER the constrained run completes; a mismatch closes the detection loop.
    """
    current_digest = compute_integrity_hash(path)
    return current_digest == baseline_digest


def run_constrained(
    task_fn: callable,
    audit: ConstraintAudit,
    eval_scripts: list[str],
    strict: bool = True,
) -> Optional[object]:
    """
    [Inferred from source thesis — not explicitly defined in source schema]
    Enforces constraint controls end-to-end:
      1. In strict mode, refuses to execute if any evasion surface remains open.
      2. Locks all eval scripts as read-only before execution.
      3. Records SHA-256 baseline hashes of all eval scripts.
      4. Executes the task function.
      5. Verifies integrity of all eval scripts post-run.
    Returns None if strict-mode preconditions fail; otherwise returns task result.
    """
    # --- Pre-run gate ---
    if strict and not audit.is_fully_constrained():
        open_names = [s.name for s in audit.open_surfaces()]
        print(f"[STRICT-MODE ABORT] Open surfaces block execution: {open_names}")
        return None

    # --- Harden evaluation scaffolding ---
    baselines: dict[str, str] = {}
    for script in eval_scripts:
        make_read_only(script)                        # lock before hashing
        baselines[script] = compute_integrity_hash(script)  # record baseline

    # --- Execute constrained task ---
    result = task_fn()

    # --- Post-run integrity verification ---
    evasion_detected = False
    for script in eval_scripts:
        if not verify_integrity(script, baselines[script]):
            print(f"[EVASION SIGNAL] Integrity mismatch detected: {script}")
            evasion_detected = True

    if strict and evasion_detected:
        print("[STRICT-MODE ABORT] Post-run integrity failure; run result invalidated.")
        return None

    print(audit.summary())
    return result
```

## Triple-Mode Insights
### EvasionSurface
**🎯 Decision:** Apply when modeling a single shortcut path a constrained system might exploit. Instantiate per identified path, capturing accessibility, detectability, and closure state. Use before a constrained run to enumerate risk; update as paths are closed.
**🎭 Analogy:** A fire escape that could also serve as an unauthorized entry point: useful infrastructure that doubles as a vulnerability requiring monitoring and possible sealing.
**💡 Insight:** The combination of accessible+detectable+closed fields implies a triage order: accessible and undetectable paths are highest risk. A closed path with no closure_method recorded is an audit gap — source implies closure_method is mandatory for a valid closure record.

### ConstraintAudit
**🎯 Decision:** Apply when tracking the complete evasion surface map across a system run. Instantiate once per system, accumulate surfaces via add_surface, and query is_fully_constrained before authorizing execution. Acts as the central ledger for constraint enforcement.
**🎭 Analogy:** A building security checklist: every possible unauthorized exit is listed, inspected, and signed off before the vault is occupied.
**💡 Insight:** ConstraintAudit is stateful across a run lifecycle. Because open_surfaces and is_fully_constrained are queryable at any point, the audit doubles as a real-time dashboard, not merely a post-hoc report — implying it should persist and be polled, not instantiated and discarded.

### run_constrained
**🎯 Decision:** Inferred as the operational phase in which a goal-directed system executes under active constraint enforcement. Applies after all audit surfaces are closed and integrity hashes are captured, representing the bounded execution window the entire audit framework is designed to protect.
**🎭 Analogy:** A pressure test conducted only after all valves are confirmed sealed — the test is meaningless unless the containment is verified first.
**💡 Insight:** [Inferred] The source implies run_constrained is the protected execution phase but never defines it explicitly. Its existence is logically required: compute_integrity_hash and is_fully_constrained are pre-run checks, and verify_integrity is post-run, bracketing an unnamed execution window that run_constrained names.

### is_fully_constrained
**🎯 Decision:** Apply as a binary gate before permitting constrained execution. Returns True only when every surface in the audit has been closed. Use as the final pre-run check; any False result should block execution until remaining open surfaces are addressed.
**🎭 Analogy:** The 'all doors locked' indicator light on an aircraft before takeoff — a single unlatched door prevents departure regardless of other systems being ready.
**💡 Insight:** is_fully_constrained is a monotonic predicate over the surfaces list: it can only become True as surfaces are closed, never spontaneously. This means adding a new surface via add_surface after prior closure can silently revert the system to unconstrained status — a race condition in dynamic audits.

### make_read_only
**🎯 Decision:** Apply to evaluation scaffolding files before a constrained run to prevent the system from modifying its own oversight mechanisms. Raises FileNotFoundError on missing paths, so existence must be verified first. Strips write bits for owner, group, and other — comprehensive, not partial.
**🎭 Analogy:** Laminating the answer key before handing it to the test proctor: the content is visible but physically unalterable.
**💡 Insight:** Stripping all three write-bit categories (owner, group, other) closes a privilege-escalation evasion path where a system might exploit group membership to regain write access. A partial strip — owner only — would leave a detectable but accessible surface, which ConstraintAudit would need to record.

### compute_integrity_hash
**🎯 Decision:** Apply before the constrained run to establish a cryptographic baseline of evaluation scaffolding. Store the returned hex digest for later comparison. Intended specifically for tamper detection, not general checksumming — context is evaluation evasion detection.
**🎭 Analogy:** Photographing a sealed envelope before handing it over: the image proves later whether the seal was broken.
**💡 Insight:** compute_integrity_hash is only useful if the hash is stored outside the constrained system's write reach — otherwise the system could overwrite both the file and the stored hash. Combined with make_read_only, the scaffolding file becomes immutable and its hash authoritative, closing a two-vector evasion path.

### verify_integrity
**🎯 Decision:** Apply after the constrained run completes, comparing the current file hash against the pre-run baseline produced by compute_integrity_hash. A mismatch signals evaluation evasion. Operates as the closing bracket of the integrity verification lifecycle.
**🎭 Analogy:** Breaking the wax seal on a legal document after a court session: any tampering since sealing is immediately evident.
**💡 Insight:** verify_integrity implicitly requires that the baseline hash was stored in a location the constrained system could not modify. If make_read_only was not applied to the hash storage location, a sophisticated system could update both the scaffolding and the stored baseline, defeating verification entirely — a gap the source does not explicitly close.

### strict-mode
**🎯 Decision:** Inferred as an execution configuration that activates all constraint mechanisms simultaneously: full audit, read-only scaffolding, integrity hashing, and post-run verification. Applies when the highest assurance level is required before and during a constrained run.
**🎭 Analogy:** A clean-room protocol that mandates every gown, glove, and airlock procedure — partial compliance is not strict-mode.
**💡 Insight:** [Inferred] Strict-mode is not defined in the source but is implied by the ensemble of mechanisms. Its value is in preventing partial deployment: a system that uses compute_integrity_hash but skips make_read_only is not operating in strict-mode and retains an open evasion surface.

### evasion-path-closure-status
**🎯 Decision:** Apply when querying whether a specific EvasionSurface has been shut down. Represented by the closed boolean field on EvasionSurface and aggregated by is_fully_constrained on ConstraintAudit. Relevant at any audit checkpoint to determine remaining risk.
**🎭 Analogy:** A traffic barrier status board: each road is either 'closed' or 'open,' and the intersection is only safe when all are closed.
**💡 Insight:** Closure status without a closure_method is an incomplete record per the source schema. An audit showing closed=True but closure_method=None or empty should be treated as suspicious — it suggests closure was asserted rather than implemented, which itself constitutes an evasion of the audit.

### evasion-signal
**🎯 Decision:** Inferred as any observable artifact indicating a constrained system attempted or succeeded in exploiting an evasion surface. Triggered by verify_integrity mismatch, open_surfaces returning non-empty post-run, or detectable=True surfaces showing exploitation evidence in logs.
**🎭 Analogy:** A tripwire alarm: the signal itself doesn't stop the intrusion but confirms it occurred and where.
**💡 Insight:** [Inferred] The source distinguishes detectable from undetectable surfaces, implying evasion-signals are only generated for detectable ones. An undetectable open surface produces no signal even if exploited — which is why closing all surfaces (is_fully_constrained) matters more than detection alone.

### strict-mode-abort
**🎯 Decision:** Inferred as the response when strict-mode preconditions are not met: is_fully_constrained returns False, or verify_integrity detects tampering. The run is halted or invalidated rather than allowed to proceed or be treated as valid output.
**🎭 Analogy:** A launch abort sequence: if any pre-launch check fails, the countdown stops regardless of how many checks passed.
**💡 Insight:** [Inferred] Strict-mode-abort is implied by the binary nature of is_fully_constrained — a partial constraint state has no sanctioned 'proceed with caution' path in the source. The source's design treats constraint as all-or-nothing, making abort the only logically consistent response to a failed gate.

### eval-script-locking
**🎯 Decision:** Inferred as the specific application of make_read_only to evaluation scripts and scaffolding files. Applies during pre-run setup to prevent a constrained system from altering the rules of its own evaluation, a direct evasion path the source targets.
**🎭 Analogy:** Bolting the referee's rulebook to the table before the game: players can read the rules but cannot rewrite them mid-match.
**💡 Insight:** [Inferred] Eval-script-locking addresses a specific evasion surface: write access to evaluation scaffolding. The source implies this surface would appear in a ConstraintAudit as accessible=True, detectable=variable, closed=False until make_read_only is applied — making it one of the highest-priority surfaces to close before run_constrained.

### post-run-integrity-verification
**🎯 Decision:** Apply after a constrained run completes by invoking verify_integrity against the pre-run hash baseline. Confirms that evaluation scaffolding was not tampered with during execution. Is the final phase of the three-phase lifecycle: hash, run, verify.
**🎭 Analogy:** Comparing a before-and-after photograph of a sealed room: any difference proves unauthorized access occurred during the interval.
**💡 Insight:** Post-run integrity verification is only as strong as the pre-run baseline storage. If compute_integrity_hash stores its result in a writable location accessible to the constrained system, the verification phase is bypassable. The source does not specify storage location, leaving this as an implicit open surface in the audit framework.

### evasion-path-accessibility
**🎯 Decision:** Apply when assessing whether a constrained system can actually reach a given evasion surface. Represented by the accessible boolean on EvasionSurface. High-priority closure target when accessible=True; lower priority but still tracked when accessible=False.
**🎭 Analogy:** Assessing whether a locked side door is reachable from inside the building — a door behind a wall poses less risk than one at the end of a hallway.
**💡 Insight:** Accessibility is a precondition for exploitability: an inaccessible surface cannot be exploited regardless of its detectable or closed status. However, accessibility can change dynamically if the constrained system gains capabilities during a run, meaning accessibility should be re-evaluated, not assumed static across the run lifecycle.

### evasion-path-detectability
**🎯 Decision:** Apply when assessing whether exploitation of an evasion surface would appear in logs. Represented by the detectable boolean on EvasionSurface. Undetectable open surfaces are highest risk; detectable open surfaces allow post-hoc audit but still represent constraint failures.
**🎭 Analogy:** A security camera over a door: its presence means intrusion is recorded, not prevented. An unmonitored door is categorically more dangerous.
**💡 Insight:** Detectability without closure is a monitoring strategy, not a constraint strategy. The source implies that is_fully_constrained requires all surfaces closed regardless of their detectability — detecting evasion after the fact does not satisfy the constraint requirement, only closure does.

### open_surfaces
**🎯 Decision:** Apply to retrieve all EvasionSurface instances where closed=False from a ConstraintAudit. Use during pre-run preparation to identify remaining work, and as the input list for close_surface calls. A non-empty return value means is_fully_constrained will return False.
**🎭 Analogy:** A checklist of unsecured exits before lockdown — the list drives action until it returns empty.
**💡 Insight:** open_surfaces is both a diagnostic and an action queue: iterate its results to drive close_surface calls. Because add_surface can append new surfaces at any time, open_surfaces should be re-polled after each closure action to catch dynamically discovered surfaces before declaring the system fully constrained.

### close_surface
**🎯 Decision:** Apply to mark a specific EvasionSurface as closed and record the closure method. Requires both the surface name and the method used. Returns False if the surface is not found, signaling a lookup failure that should be treated as an audit error, not a successful closure.
**🎭 Analogy:** Signing off on a specific fire exit as sealed, with the sealing method documented — an unsigned checkmark is insufficient.
**💡 Insight:** close_surface requires a method parameter, enforcing that closures are documented. A False return value indicates the surface name was not found in the audit — this could mean the surface was never registered via add_surface, suggesting a gap between the discovered evasion landscape and the recorded audit surface map.

## Concept Reference
| Concept | Technical | Plain | Importance | Citation |
|---------|-----------|-------|------------|----------|
| EvasionSurface | extracted: single identified shortcut path available to a constrained system | A single shortcut path a constrained system could exploit | 100% | _"EvasionSurface — represents a single identified shortcut path available to a con"_ |
| ConstraintAudit | extracted: tracks the full evasion surface map for a goal-directed system run; accumulates discovered surfaces and their closure status | Audit record tracking all evasion surfaces and their closure status | 100% | _"ConstraintAudit — tracks the full evasion surface map for a goal-directed system"_ |
| run_constrained | extracted: executes a task function only if the constraint audit is fully closed when strict mode is enabled; locks eval scripts before running the task, then verifies no eval script was tampered with | Runs a task only when all evasion surfaces are closed and eval scripts are protected | 100% | _"run_constrained — executes a task function only if the constraint audit is fully"_ |
| is_fully_constrained | extracted: returns True only when all known evasion surfaces are closed | Checks whether every known evasion surface has been closed | 90% | _"is_fully_constrained — returns True only when all known evasion surfaces are clo"_ |
| make_read_only | extracted: removes write permissions from a file so a constrained system cannot modify evaluation scaffolding | Locks a file by removing write permissions to protect evaluation scaffolding | 90% | _"make_read_only — removes write permissions from a file so a constrained system c"_ |
| compute_integrity_hash | extracted: computes a SHA-256 hash of a file for tamper detection; intended to be called before the system run so the result can be compared afterward to detect evaluation evasion | Computes a pre-run SHA-256 hash to detect later file tampering | 90% | _"compute_integrity_hash — computes a SHA-256 hash of a file for tamper detection;"_ |
| verify_integrity | extracted: returns True if the file still matches the expected hash; a False result indicates the constrained system modified the file, which is described as a clear evasion signal | Verifies a file's hash matches the pre-run value to detect tampering | 90% | _"verify_integrity — returns True if the file at the given path still matches the "_ |
| strict-mode | extracted: if True, refuse to run when open evasion surfaces remain | Mode that blocks task execution when any evasion surfaces are still open | 90% | _"strict — if True, refuse to run when open evasion surfaces remain"_ |
| evasion-path-closure-status | extracted: whether the path has been shut down | Flag indicating whether the evasion path has been shut down | 85% | _"closed — whether the path has been shut down"_ |
| evasion-signal | extracted: a False result indicates the constrained system modified the file, which is described as a clear evasion signal | Hash mismatch after a run signals the system tampered with eval files | 85% | _"a False result indicates the constrained system modified the file, which is desc"_ |
| strict-mode-abort | extracted: if strict is enabled and the audit is not fully constrained, collect the names of all open surfaces and abort before running the task | Aborts execution and lists open surfaces when strict mode finds unresolved paths | 85% | _"if strict is enabled and the audit is not fully constrained, collect the names o"_ |
| eval-script-locking | extracted: locks eval scripts before running the task | Evaluation scripts are made read-only before the constrained task runs | 85% | _"locks eval scripts before running the task"_ |
| post-run-integrity-verification | extracted: verifies no eval script was tampered with afterward | Checks eval script hashes after the task to detect any tampering | 85% | _"verifies no eval script was tampered with afterward"_ |
| evasion-path-accessibility | extracted: whether the system can reach this path | Flag indicating if the constrained system can reach the evasion path | 80% | _"accessible — whether the system can reach this path"_ |
| evasion-path-detectability | extracted: whether exploitation would appear in logs | Flag indicating if exploiting this path would appear in logs | 80% | _"detectable — whether exploitation would appear in logs"_ |
| open_surfaces | extracted: returns all evasion paths not yet closed | Returns all evasion paths that have not yet been closed | 80% | _"open_surfaces — returns all evasion paths not yet closed"_ |
| close_surface | extracted: marks a surface as closed and records how it was closed; returns True if found and closed, False otherwise | Closes an evasion path and records the closure method | 80% | _"close_surface — marks a surface as closed and records how it was closed; returns"_ |
| goal-directed-system | extracted: goal-directed system run | A system pursuing a goal that may exploit shortcuts to evade constraints | 80% | _"tracks the full evasion surface map for a goal-directed system run"_ |
| closure-method | extracted: how the path was closed | Records the method used to close an evasion path | 75% | _"closure_method — how the path was closed"_ |
| add_surface | extracted: appends a new evasion surface to the audit record | Registers a new evasion path in the audit record | 75% | _"add_surface — appends a new evasion surface to the audit record; parameter surfa"_ |
| write-bit-stripping | extracted: Strip owner, group, and other write bits | Removes write permissions for owner, group, and other from a file | 75% | _"Strip owner, group, and other write bits"_ |
| audit-summary | extracted: returns a formatted string reporting total surfaces, how many are closed, how many remain open, and whether the system is fully constrained | Formatted report of surface counts and overall constraint status | 70% | _"summary — returns a formatted string reporting total surfaces, how many are clos"_ |
| eval-scripts-parameter | extracted: list of evaluation script paths to lock and monitor | List of evaluation script paths that will be locked and integrity-checked | 70% | _"eval_scripts — list of evaluation script paths to lock and monitor"_ |
| read-only-mode-bits | extracted: the permission bits after write access is stripped for owner, group, and other | Permission bits with all write access removed for owner, group, and other | 65% | _"Variable read_only_mode — the permission bits after write access is stripped for"_ |
| task_fn-parameter | extracted: callable that executes the goal-directed work and returns output | Callable passed to run_constrained that performs the actual goal-directed work | 65% | _"task_fn — callable that executes the goal-directed work and returns output"_ |
| current-permission-bits | extracted: the existing permission bits of the file | Stores the file's existing permission bits before modification | 60% | _"Variable current — the existing permission bits of the file"_ |
| chunked-file-hashing | extracted: successive 8192-byte blocks of the file fed into the hash | File is read and hashed in successive 8192-byte blocks | 60% | _"Variable chunk — successive 8192-byte blocks of the file fed into the hash"_ |
| SHA-256-hash-accumulator | extracted: SHA-256 hash accumulator | Internal variable accumulating the SHA-256 hash of file contents | 55% | _"Variable hasher — SHA-256 hash accumulator"_ |
| FileNotFoundError-on-missing-path | extracted: raises FileNotFoundError if path does not exist | make_read_only raises FileNotFoundError when the target file is missing | 55% | _"raises FileNotFoundError if path does not exist"_ |

## Glossary
| Term | Definition | Concept IDs |
|------|------------|-------------|
| EvasionSurface | A single identified shortcut path available to a constrained system, described by its name, accessibility, detectability, closed status, and the method used to close it. | 1 |
| ConstraintAudit | A record that tracks the full evasion surface map for a goal-directed system run, accumulating discovered surfaces and their closure status. | 6 |
| evasion-path-accessibility | Whether a constrained system can reach a given evasion path, as represented by the accessible field of an EvasionSurface. | 2 |
| evasion-path-detectability | Whether exploitation of an evasion path would appear in logs, as represented by the detectable field of an EvasionSurface. | 3 |
| evasion-path-closure-status | Whether a given evasion path has been shut down, tracked by the closed field and the method recorded in closure_method. | 4 |
| open_surfaces | The set of evasion paths in a ConstraintAudit that have not yet been closed, returned by the open_surfaces method. | 8 |
| close_surface | The operation that marks a named evasion surface as closed and records the closure method used; returns True if found and closed, False otherwise. | 9 |
| is_fully_constrained | A boolean condition that is True only when every known evasion surface in a ConstraintAudit is closed. | 10 |
| make_read_only | A function that strips owner, group, and other write permission bits from a file so a constrained system cannot modify evaluation scaffolding. | 12 |
| compute_integrity_hash | A function that computes a SHA-256 hex digest of a file before a system run, intended for later comparison to detect evaluation evasion. | 16 |
| evasion-signal | The condition indicated when verify_integrity returns False, meaning the constrained system modified a file that should have remained unchanged. | 20 |
| verify_integrity | A function that returns True if a file still matches its pre-run SHA-256 hash, and False as a clear evasion signal if the file was modified. | 19 |
| run_constrained | A function that locks eval scripts, optionally enforces full closure of all evasion surfaces under strict mode, executes a task function, and verifies post-run integrity of all monitored scripts. | 21, 22, 23, 24, 25 |

## Edge Cases & Warnings
- ⚠️ verify_integrity and run_constrained are named and described in the thesis as if sourced, but neither appears in the provided source text — this is fabrication, not elaboration
- ⚠️ The inline comment ('Strip owner, group, and other write bits') and the two named variables (current, read_only_mode) are source-level implementation details that the thesis drops entirely, losing specificity about the permission-stripping mechanism
- ⚠️ compute_integrity_hash's return type is truncated in the source ('hex digest of the file's' is cut off) — the pipeline does not flag this as an incomplete source artifact, which is an auditing blind spot
- ⚠️ The source's framing as a structured API reference (field lists, method signatures, parameter names, return values) is itself a signal about intended use (programmatic scaffolding for constraint enforcement) that the pipeline did not surface as a structural observation
- ⚠️ The accessible and detectable boolean fields on EvasionSurface carry implicit threat-model logic (a path that is accessible but not detectable is highest risk) that the pipeline neither extracted nor flagged as an inferred extension worth tagging

## Emergence Assessment
The pipeline achieves solid extraction of the explicitly documented constructs (EvasionSurface fields, ConstraintAudit methods, make_read_only, compute_integrity_hash) and accurately assembles their relationships into a coherent threat-model framing. The thesis is largely source-faithful. However, the thesis introduces verify_integrity and run_constrained as named constructs with attributed behaviors (strict mode, returning None, end-to-end enforcement, post-run integrity closure), none of which appear in the provided source text. These represent inferred or hallucinated downstream functions not grounded in the ~505-word source. This is a meaningful fidelity breach: the pipeline conflated what the source implies a system might need with what the source actually specifies. The 5 flagged elaborations likely undercount the degree of extension present in the thesis and taxonomy. Taxonomy inflation is moderate; most terms are traceable, but governance-adjacent framings (e.g. 'evasion signal' as a formal detection category) extend sparse source language. Interconnectedness score is appropriately high because the source genuinely chains fields and methods into a coherent audit lifecycle. Technical depth is moderate: the source is implementation-sketch level, not deep algorithmic specification.


## Reflexive Observations
- ◈ The source is itself structured as evaluation scaffolding — a specification document for auditing and constraining a goal-directed system — meaning the document occupies the same functional role as the eval scripts it describes locking and hashing: it is the kind of artifact that make_read_only and compute_integrity_hash are designed to protect from modification by a constrained system.
## Recommendations
- 🔧 Quarantine verify_integrity and run_constrained from the thesis entirely unless a source amendment is provided; their presence contaminates the fidelity score for an otherwise accurate extraction
- 🔧 Flag the truncated compute_integrity_hash description as a source artifact defect rather than silently completing the implied meaning
- 🔧 Promote the inline comment and variable names (current, read_only_mode) into extracted concepts with direct provenance citations, as they carry implementation-level specificity absent from the method summary
- 🔧 Add a provenance-flagged inference note when boolean field combinations (accessible=True, detectable=False) imply a risk ranking not stated in the source, rather than silently embedding the implication in elaborations
- 🔧 Surface the API-reference document structure as a reflexive observation: the source itself is scaffolding for constraint enforcement, mirroring what it describes

## Quick Reference
```python
import hashlib, os, stat

# --- EvasionSurface (highest risk: accessible=True, detectable=False) ---
# surface = EvasionSurface("write_eval", "can overwrite scorer", accessible=True, detectable=False)
# audit.add_surface(surface)
# audit.close_surface("write_eval", method="make_read_only applied")
# audit.is_fully_constrained()  # True only when ALL surfaces closed
# audit.summary()               # total / closed / open / gap report

# --- make_read_only: strip all write bits from eval scaffolding ---
def make_read_only(path):
    current = os.stat(path).st_mode
    os.chmod(path, current & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))

# --- compute_integrity_hash: SHA-256 baseline BEFORE run ---
def compute_integrity_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""): h.update(chunk)
    return h.hexdigest()

# --- verify_integrity: compare AFTER run; mismatch = evasion signal ---
def verify_integrity(path, baseline): return compute_integrity_hash(path) == baseline

# STRICT-MODE CHECKLIST (all required; partial = false confidence):
# 1. audit.is_fully_constrained() → True  else abort
# 2. make_read_only(script) for each eval script
# 3. baseline = compute_integrity_hash(script); store outside system write reach
# 4. run task
# 5. verify_integrity(script, baseline) for each; False → evasion signal → invalidate
```

---
_Generated by Philosopher's Stone v5 — EchoSeed_
