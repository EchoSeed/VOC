# Runtime Receipt Generation

> Trigger this skill when you need to create verifiable proof of program execution, track computational operations for audit purposes, generate transaction logs in untrusted environments, or provide cryptographic evidence of what a program actually did at runtime. Essential for blockchain systems, zero-knowledge proofs, distributed computing verification, billing/metering systems, and any scenario requiring trustless accountability.

## Overview

Runtime receipt generation creates immutable computational artifacts that prove what operations were performed during program execution. These receipts combine transaction logs, resource consumption metrics, execution metadata, and cryptographic attestation to transform ephemeral runtime behavior into permanent, auditable evidence. This enables third-party verification of computational claims without requiring trust or re-execution, making it foundational for distributed systems, blockchain protocols, and accountable computing environments.

## When to Use

- Building blockchain or distributed ledger systems requiring transaction verification
- Implementing zero-knowledge computation with verifiable outputs
- Creating audit trails for compliance, security, or regulatory requirements
- Designing billing/metering systems that charge based on actual resource consumption
- Developing trustless computation where parties need cryptographic proof of execution
- Implementing rollback-prevention systems requiring transaction finality
- Building debugging infrastructure that captures complete execution context
- Creating accountability frameworks where actions must be attributable to actors

## Core Workflow

1. **Capture Execution Context** - Record initial state, input parameters, environment variables, and runtime configuration before execution begins
2. **Monitor State Transitions** - Log each deterministic state change as execution progresses, capturing before/after snapshots and transition rules
3. **Track Resource Consumption** - Measure CPU cycles, memory usage, storage operations, and network bandwidth throughout execution
4. **Generate Cryptographic Proof** - Create mathematical evidence of correct execution using hash chains, Merkle trees, or zero-knowledge proofs
5. **Construct Receipt Artifact** - Assemble metadata, logs, proofs, and attestations into immutable, verifiable receipt structure
6. **Sign and Finalize** - Apply cryptographic signatures to bind receipt to specific execution and prevent tampering
7. **Store and Distribute** - Persist receipt to immutable storage and make available for independent verification

## Key Patterns

### Append-Only Transaction Log

Build an immutable event log where each entry cryptographically references the previous entry, creating an unforgeable chain of execution events.

```python
from typing import List, Dict, Any
from hashlib import sha256
from datetime import datetime
import json

class TransactionLog:
    """Append-only log with cryptographic chaining."""
    
    def __init__(self):
        self.entries: List[Dict[str, Any]] = []
        self.genesis_hash = sha256(b"genesis").hexdigest()
    
    def append(self, operation: str, data: Dict[str, Any]) -> str:
        """Add entry to log and return its hash."""
        # Get previous hash or use genesis
        prev_hash = self.entries[-1]["hash"] if self.entries else self.genesis_hash
        
        # Create new entry
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "operation": operation,
            "data": data,
            "prev_hash": prev_hash,
            "sequence": len(self.entries)
        }
        
        # Compute cryptographic hash of entry
        entry_bytes = json.dumps(entry, sort_keys=True).encode()
        entry["hash"] = sha256(entry_bytes).hexdigest()
        
        self.entries.append(entry)
        return entry["hash"]
    
    def verify_chain(self) -> bool:
        """Verify integrity of entire log chain."""
        for i, entry in enumerate(self.entries):
            # Check sequence numbers
            if entry["sequence"] != i:
                return False
            
            # Check hash chain
            expected_prev = self.entries[i-1]["hash"] if i > 0 else self.genesis_hash
            if entry["prev_hash"] != expected_prev:
                return False
            
            # Recompute and verify hash
            entry_copy = {k: v for k, v in entry.items() if k != "hash"}
            entry_bytes = json.dumps(entry_copy, sort_keys=True).encode()
            computed_hash = sha256(entry_bytes).hexdigest()
            if computed_hash != entry["hash"]:
                return False
        
        return True

# Usage example
log = TransactionLog()
log.append("start_execution", {"input": "data.csv", "mode": "batch"})
log.append("state_transition", {"from": "init", "to": "processing"})
log.append("resource_consumed", {"cpu_ms": 1247, "memory_mb": 512})
log.append("complete_execution", {"output": "results.json", "status": "success"})

assert log.verify_chain()  # Cryptographically verify integrity
```

### Deterministic Receipt Generation

Create receipts that guarantee identical inputs produce identical receipts, enabling reproducible verification.

```python
from typing import Any, Dict
from hashlib import sha256
import json

class DeterministicReceipt:
    """Generate deterministic, verifiable execution receipts."""
    
    def __init__(self, execution_id: str):
        self.execution_id = execution_id
        self.state_transitions: List[Dict[str, Any]] = []
        self.resources: Dict[str, float] = {"cpu_ms": 0, "memory_mb": 0, "io_ops": 0}
        self.metadata: Dict[str, Any] = {}
    
    def record_transition(self, from_state: str, to_state: str, 
                         input_hash: str, output_hash: str) -> None:
        """Record deterministic state transition."""
        self.state_transitions.append({
            "from": from_state,
            "to": to_state,
            "input_hash": input_hash,
            "output_hash": output_hash,
            "step": len(self.state_transitions)
        })
    
    def track_resource(self, resource_type: str, amount: float) -> None:
        """Accumulate resource consumption."""
        if resource_type in self.resources:
            self.resources[resource_type] += amount
    
    def generate(self, output_data: Any, signature_key: str = None) -> Dict[str, Any]:
        """Generate final receipt with cryptographic attestation."""
        # Hash output deterministically
        output_bytes = json.dumps(output_data, sort_keys=True).encode()
        output_hash = sha256(output_bytes).hexdigest()
        
        # Construct receipt
        receipt = {
            "execution_id": self.execution_id,
            "version": "1.0",
            "state_transitions": self.state_transitions,
            "resources": self.resources,
            "output_hash": output_hash,
            "metadata": self.metadata
        }
        
        # Compute receipt hash (deterministic)
        receipt_bytes = json.dumps(receipt, sort_keys=True).encode()
        receipt["receipt_hash"] = sha256(receipt_bytes).hexdigest()
        
        # Optional: Add signature for attestation
        if signature_key:
            receipt["signature"] = self._sign(receipt["receipt_hash"], signature_key)
        
        return receipt
    
    def _sign(self, data: str, key: str) -> str:
        """Mock signature - replace with real cryptographic signing."""
        combined = f"{data}:{key}"
        return sha256(combined.encode()).hexdigest()
    
    @staticmethod
    def verify(receipt: Dict[str, Any], expected_output: Any) -> bool:
        """Verify receipt matches expected output."""
        # Recompute output hash
        output_bytes = json.dumps(expected_output, sort_keys=True).encode()
        expected_hash = sha256(output_bytes).hexdigest()
        
        # Check receipt integrity
        receipt_copy = {k: v for k, v in receipt.items() 
                       if k not in ["receipt_hash", "signature"]}
        receipt_bytes = json.dumps(receipt_copy, sort_keys=True).encode()
        computed_hash = sha256(receipt_bytes).hexdigest()
        
        return (receipt["output_hash"] == expected_hash and
                receipt["receipt_hash"] == computed_hash)

# Usage
receipt_gen = DeterministicReceipt("exec_20240218_001")
receipt_gen.record_transition("idle", "running", 
                              sha256(b"input").hexdigest(),
                              sha256(b"intermediate").hexdigest())
receipt_gen.track_resource("cpu_ms", 523.4)
receipt_gen.track_resource("memory_mb", 128.7)

output = {"result": "success", "records_processed": 1000}
receipt = receipt_gen.generate(output, signature_key="secret_key")

# Anyone can verify without trust
assert DeterministicReceipt.verify(receipt, output)
```

### Resource Consumption Tracking

Precisely measure and record computational resource usage for billing, optimization, or verification.

```python
import time
import psutil
from contextlib import contextmanager
from typing import Dict, Callable, Any

class ResourceTracker:
    """Track and record resource consumption during execution."""
    
    def __init__(self):
        self.measurements: Dict[str, float] = {
            "cpu_time_ms": 0.0,
            "wall_time_ms": 0.0,
            "memory_peak_mb": 0.0,
            "io_read_mb": 0.0,
            "io_write_mb": 0.0
        }
        self._process = psutil.Process()
    
    @contextmanager
    def measure(self):
        """Context manager to measure resources during execution."""
        # Capture start state
        start_time = time.perf_counter()
        start_cpu = self._process.cpu_times()
        start_mem = self._process.memory_info()
        start_io = self._process.io_counters() if hasattr(self._process, 'io_counters') else None
        
        try:
            yield self
        finally:
            # Capture end state
            end_time = time.perf_counter()
            end_cpu = self._process.cpu_times()
            end_mem = self._process.memory_info()
            end_io = self._process.io_counters() if hasattr(self._process, 'io_counters') else None
            
            # Calculate deltas
            self.measurements["wall_time_ms"] = (end_time - start_time) * 1000
            self.measurements["cpu_time_ms"] = (
                (end_cpu.user - start_cpu.user + end_cpu.system - start_cpu.system) * 1000
            )
            self.measurements["memory_peak_mb"] = end_mem.rss / (1024 * 1024)
            
            if start_io and end_io:
                self.measurements["io_read_mb"] = (end_io.read_bytes - start_io.read_bytes) / (1024 * 1024)
                self.measurements["io_write_mb"] = (end_io.write_bytes - start_io.write_bytes) / (1024 * 1024)
    
    def get_receipt_data(self) -> Dict[str, Any]:
        """Format measurements for inclusion in receipt."""
        return {
            "resources": self.measurements,
            "units": {
                "cpu_time_ms": "milliseconds",
                "wall_time_ms": "milliseconds",
                "memory_peak_mb": "megabytes",
                "io_read_mb": "megabytes",
                "io_write_mb": "megabytes"
            },
            "timestamp": time.time()
        }

# Usage in receipt generation
tracker = ResourceTracker()

with tracker.measure():
    # Execute the actual computation
    result = sum(i**2 for i in range(1000000))
    time.sleep(0.1)  # Simulate some work

# Include resource data in receipt
receipt_data = tracker.get_receipt_data()
print(f"CPU time: {receipt_data['resources']['cpu_time_ms']:.2f}ms")
print(f"Memory peak: {receipt_data['resources']['memory_peak_mb']:.2f}MB")
```

### Event Sourcing Pattern

Capture all state changes as immutable events, enabling complete replay and reconstruction of system state.

```python
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import json

@dataclass
class Event:
    """Immutable event capturing a state change."""
    event_type: str
    aggregate_id: str
    sequence: int
    timestamp: str
    data: Dict[str, Any]
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class EventStore:
    """Store and replay events for state reconstruction."""
    
    def __init__(self):
        self.events: List[Event] = []
        self._sequences: Dict[str, int] = {}  # Track sequence per aggregate
    
    def append_event(self, event_type: str, aggregate_id: str, 
                    data: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None) -> Event:
        """Append new event to store."""
        # Get next sequence number for this aggregate
        seq = self._sequences.get(aggregate_id, 0)
        self._sequences[aggregate_id] = seq + 1
        
        event = Event(
            event_type=event_type,
            aggregate_id=aggregate_id,
            sequence=seq,
            timestamp=datetime.utcnow().isoformat(),
            data=data,
            metadata=metadata or {}
        )
        
        self.events.append(event)
        return event
    
    def get_events(self, aggregate_id: str, from_seq: int = 0) -> List[Event]:
        """Retrieve events for aggregate starting from sequence."""
        return [e for e in self.events 
                if e.aggregate_id == aggregate_id and e.sequence >= from_seq]
    
    def replay_state(self, aggregate_id: str, handlers: Dict[str, Callable]) -> Any:
        """Reconstruct state by replaying events through handlers."""
        state = {}
        events = self.get_events(aggregate_id)
        
        for event in events:
            if event.event_type in handlers:
                state = handlers[event.event_type](state, event.data)
        
        return state
    
    def generate_receipt(self) -> Dict[str, Any]:
        """Generate receipt from complete event history."""
        return {
            "total_events": len(self.events),
            "aggregates": len(self._sequences),
            "events": [e.to_dict() for e in self.events],
            "receipt_hash": self._compute_hash()
        }
    
    def _compute_hash(self) -> str:
        """Compute hash of entire event stream."""
        from hashlib import sha256
        events_json = json.dumps([e.to_dict() for e in self.events], sort_keys=True)
        return sha256(events_json.encode()).hexdigest()

# Usage example
store = EventStore()

# Record execution events
store.append_event("ExecutionStarted", "run_123", 
                  {"input_file": "data.csv", "config": {"mode": "batch"}})
store.append_event("StateChanged", "run_123",
                  {"from": "idle", "to": "processing"})
store.append_event("ResourceAllocated", "run_123",
                  {"cpu_cores": 4, "memory_gb": 16})
store.append_event("ProcessingCompleted", "run_123",
                  {"records": 10000, "duration_ms": 5432})

# Define handlers for state reconstruction
handlers = {
    "ExecutionStarted": lambda s, d: {**s, "status": "started", "input": d["input_file"]},
    "StateChanged": lambda s, d: {**s, "state": d["to"]},
    "ResourceAllocated": lambda s, d: {**s, "resources": d},
    "ProcessingCompleted": lambda s, d: {**s, "status": "completed", "metrics": d}
}

# Replay to reconstruct final state
final_state = store.replay_state("run_123", handlers)
print(f"Final state: {final_state}")

# Generate receipt with complete audit trail
receipt = store.generate_receipt()
print(f"Receipt contains {receipt['total_events']} events")
```

## Concept Reference

| Concept | Technical | Plain | Importance |
|---------|-----------|-------|------------|
| Runtime | The period during program execution when instructions are processed by the CPU, as opposed to compile-time or load-time. Encompasses the execution env | The actual time when your program is running and doing its job, like when you click 'play' on a video and it starts working. | 0.95 |
| Receipt | A computational artifact or data structure generated as proof of transaction execution, state transition, or operation completion, typically containin | A digital record that proves something happened in a program, like a confirmation slip showing that an action was completed. | 0.90 |
| Verifiability | The property enabling independent validation of computational claims through examination of proofs, logs, or receipts without requiring trust or re-ex | The ability for anyone to check and confirm that something really happened the way it was claimed, without having to take someone's word for it. | 0.87 |
| Computational Proof | Mathematical or cryptographic evidence demonstrating that a specific computation was executed correctly, often verifiable by third parties without re- | Mathematical evidence that shows your computer actually did the work it claims to have done, without anyone needing to redo it. | 0.88 |
| Deterministic Execution | Computational behavior where identical inputs and initial states invariably produce identical outputs and final states, eliminating non-deterministic | When a program always gives you the same answer if you start with the same information, like a calculator always showing '4' when you type '2+2'. | 0.86 |
| Execution Verification | The process of validating that computational operations were performed correctly according to specified constraints, often through cryptographic proof | Checking that a program did exactly what it was supposed to do, like double-checking your math homework. | 0.85 |
| Transaction Finality | The state where a committed transaction becomes irreversible and permanently part of the system's history, providing certainty and preventing rollback | When something is done and can't be undone, like when a bank transfer is complete and the money has officially moved. | 0.84 |
| Audit Trail | A chronological record of system activities providing documentary evidence of operations, enabling accountability, debugging, and compliance verificat | A trail of breadcrumbs showing every step a program took, so you can trace back what happened and who did what. | 0.83 |
| State Transition | A change in the computational state of a system from one valid configuration to another, governed by deterministic rules and often associated with tra | When a program moves from one situation to another, like a traffic light changing from red to green. | 0.82 |
| Immutable Record | A data structure designed to be write-once and read-many, preventing modification or deletion after creation, often enforced through cryptographic has | Information that can never be changed or erased once it's written down, like carving something in stone. | 0.81 |

## Glossary

| Term | Definition | Concept IDs |
|------|------------|-------------|
| Runtime Receipt | A computational artifact generated during program execution that cryptographically proves what operations occurred, capt | [1, 2, 7, 13] |
| Execution Verification | The process of validating computational correctness through cryptographic proofs or auditable logs, enabling third parti | [3, 6, 16] |
| Transaction Log | A sequential, append-only record of state-changing operations maintaining chronological order and atomicity, forming an | [4, 8, 14] |
| State Transition | A deterministic change in system configuration from one valid state to another, governed by rules and recorded as part o | [5, 9] |
| Computational Proof | Mathematical or cryptographic evidence demonstrating correct execution of specific computations, verifiable by third par | [6, 16] |
| Immutable Record | A write-once, read-many data structure preventing modification after creation, often enforced through cryptographic hash | [12, 4, 11] |
| Deterministic Execution | Computational behavior where identical inputs and initial states invariably produce identical outputs, eliminating varia | [9, 5] |
| Output Attestation | A cryptographically signed declaration certifying results from a computational process, binding outputs to specific exec | [17, 2, 6] |
| Audit Trail | A chronological record of system activities providing documentary evidence for accountability, debugging, and compliance | [8, 4, 10] |
| Execution Context | The complete runtime environment including call stack, variable bindings, and system state that determines program behav | [15, 1, 7] |
| Transaction Finality | The irreversible state where a committed transaction becomes permanent system history, providing certainty and preventin | [11, 12] |
| Resource Consumption | Quantified utilization of computational resources during execution, measured and recorded in receipts for billing, optim | [13, 7] |
| Accountability | The system property enabling attribution of actions to specific actors through verifiable records, supporting auditabili | [10, 8, 16] |

## Edge Cases & Warnings

- ⚠️ **Clock Skew**: In distributed systems, timestamp inconsistencies can break chronological ordering. Use logical clocks (Lamport/vector clocks) or consensus-based timestamps instead of relying on system time.
- ⚠️ **Hash Collision Vulnerability**: While cryptographically unlikely, hash collisions could theoretically forge receipts. Use strong hash functions (SHA-256 minimum) and consider domain separation.
- ⚠️ **Non-Deterministic Operations**: System calls, random number generation, and concurrent execution can break deterministic verification. Isolate or seed all sources of non-determinism.
- ⚠️ **Receipt Bloat**: Comprehensive logging can generate massive receipts. Implement compression, Merkle tree summarization, or selective logging strategies.
- ⚠️ **Replay Attacks**: Old valid receipts could be resubmitted maliciously. Include nonces, timestamps, or sequence numbers and maintain replay protection.
- ⚠️ **Incomplete Captures**: Receipts may miss critical context if execution spans multiple processes, containers, or machines. Design for distributed tracing integration.
- ⚠️ **Resource Measurement Overhead**: Tracking itself consumes resources, affecting measurements. Calibrate overhead and account for measurement costs in resource calculations.
- ⚠️ **Cryptographic Key Management**: Receipt signatures are worthless if keys are compromised. Implement proper key rotation, storage, and revocation mechanisms.

## Quick Reference

```python
from hashlib import sha256
import json
from datetime import datetime

class MinimalReceipt:
    """Bare-bones receipt generator for quick integration."""
    
    def __init__(self, execution_id: str):
        self.id = execution_id
        self.log = []
    
    def log_event(self, event: str, data: dict):
        """Log an execution event."""
        self.log.append({
            "ts": datetime.utcnow().isoformat(),
            "event": event,
            "data": data
        })
    
    def finalize(self, output: any) -> dict:
        """Generate final receipt with proof."""
        receipt = {
            "execution_id": self.id,
            "log": self.log,
            "output": output
        }
        # Compute cryptographic fingerprint
        receipt_json = json.dumps(receipt, sort_keys=True)
        receipt["proof"] = sha256(receipt_json.encode()).hexdigest()
        return receipt

# Usage
r = MinimalReceipt("exec_001")
r.log_event("start", {"input": "data.csv"})
r.log_event("process", {"records": 1000})
receipt = r.finalize({"status": "success"})
print(f"Receipt proof: {receipt['proof']}")
```

---
_Generated by Philosopher's Stone v4 — EchoSeed_
Philosopher's Stone v4 × Skill Forge × EchoSeed
