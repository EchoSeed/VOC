# Direct-Peering Agent Transport

> Trigger this skill when designing AI agent communication infrastructure, migrating from centralized LLM routing, optimizing for low token costs on routine transfers, eliminating single points of failure in multi-agent systems, or implementing auditable security boundaries. Use when horizontal scalability, low latency, and deterministic guardrails matter more than intelligent message interpretation.

## Overview

This skill teaches agents to implement a dumb-pipe architecture for peer-to-peer data exchange, inspired by mIRC FTP protocols. Instead of routing every message through a central LLM that parses and forwards content, agents communicate directly using stateless transport primitives with rule-based security layers. The infrastructure-first approach prioritizes building a reliable, auditable foundation before adding application logic, trading intelligent mediation for scalability, cost efficiency, and resilience.

## When to Use

- **High-frequency data exchanges** between agents that follow predictable patterns (status updates, file transfers, structured reports)
- **Token cost optimization** scenarios where LLM inference on every message creates unsustainable API expenses
- **Latency-sensitive applications** requiring sub-second response times without LLM processing delays
- **Security-critical systems** needing auditable, deterministic validation rather than opaque model-based filtering
- **Horizontally scaled deployments** where adding peer nodes should not bottleneck on a central routing service
- **Reliability requirements** that cannot tolerate a single point of failure in the communication layer

## Core Workflow

1. **Establish Direct Peer Connections**: Configure agents with network addresses of intended communication partners, bypassing central hub topology
2. **Define Structured Data Schemas**: Specify explicit field definitions, type constraints, and serialization formats for each message type
3. **Implement Deterministic Security Blocks**: Deploy rule-based validation at sender and receiver boundaries using explicit conditional logic
4. **Transfer Data via Stateless Primitives**: Send messages treating each as an independent operation with no session context
5. **Apply Auditable Guardrails**: Log all security checks and policy enforcement decisions for compliance verification
6. **Scale Horizontally**: Add new peer nodes without modifying existing infrastructure or introducing coordination overhead

## Key Patterns

### Stateless Message Envelope

Encapsulate data in self-contained structures that carry all necessary context, enabling independent processing without session memory.

```python
from dataclasses import dataclass
from typing import Literal
import json
import hashlib

@dataclass
class MessageEnvelope:
    """Stateless transport primitive for direct peer communication."""
    sender_id: str
    recipient_id: str
    message_type: Literal["data_transfer", "status_update", "file_chunk"]
    payload: dict
    schema_version: str = "1.0"
    checksum: str = ""
    
    def __post_init__(self):
        """Calculate deterministic checksum for integrity validation."""
        content = f"{self.sender_id}{self.recipient_id}{self.message_type}{json.dumps(self.payload, sort_keys=True)}{self.schema_version}"
        self.checksum = hashlib.sha256(content.encode()).hexdigest()
    
    def validate(self) -> bool:
        """Deterministic validation without LLM inference."""
        # Rule-based security checks
        if not self.sender_id or not self.recipient_id:
            return False
        if self.message_type not in ["data_transfer", "status_update", "file_chunk"]:
            return False
        
        # Checksum verification
        expected = hashlib.sha256(
            f"{self.sender_id}{self.recipient_id}{self.message_type}{json.dumps(self.payload, sort_keys=True)}{self.schema_version}".encode()
        ).hexdigest()
        return self.checksum == expected

# Usage example
envelope = MessageEnvelope(
    sender_id="agent_001",
    recipient_id="agent_002",
    message_type="data_transfer",
    payload={"file_name": "report.csv", "size_bytes": 2048}
)

if envelope.validate():
    print(f"✓ Message valid, checksum: {envelope.checksum[:8]}...")
```

### Deterministic Security Boundary Layer

Implement transparent, auditable guardrails using explicit conditional logic instead of learned model weights.

```python
from typing import List, Dict, Callable
from enum import Enum
import logging

class PolicyAction(Enum):
    ALLOW = "allow"
    DENY = "deny"
    QUARANTINE = "quarantine"

class SecurityBoundary:
    """Rule-based policy enforcement for agent communication."""
    
    def __init__(self):
        self.rules: List[Callable[[MessageEnvelope], tuple[PolicyAction, str]]] = []
        self.audit_log: List[Dict] = []
    
    def add_rule(self, rule_fn: Callable[[MessageEnvelope], tuple[PolicyAction, str]]):
        """Register a deterministic validation rule."""
        self.rules.append(rule_fn)
    
    def evaluate(self, envelope: MessageEnvelope) -> PolicyAction:
        """Apply all rules in sequence, logging each decision."""
        for idx, rule in enumerate(self.rules):
            action, reason = rule(envelope)
            
            # Auditable logging
            self.audit_log.append({
                "rule_index": idx,
                "action": action.value,
                "reason": reason,
                "sender": envelope.sender_id,
                "recipient": envelope.recipient_id,
                "message_type": envelope.message_type,
                "checksum": envelope.checksum[:16]
            })
            
            if action != PolicyAction.ALLOW:
                return action
        
        return PolicyAction.ALLOW

# Example deterministic rules
def size_limit_rule(envelope: MessageEnvelope) -> tuple[PolicyAction, str]:
    """Deny payloads exceeding size threshold."""
    payload_size = len(json.dumps(envelope.payload))
    if payload_size > 10000:  # 10KB limit
        return PolicyAction.DENY, f"Payload size {payload_size} exceeds 10KB limit"
    return PolicyAction.ALLOW, "Size check passed"

def sender_allowlist_rule(envelope: MessageEnvelope) -> tuple[PolicyAction, str]:
    """Only accept messages from registered agents."""
    allowed_senders = {"agent_001", "agent_002", "agent_003"}
    if envelope.sender_id not in allowed_senders:
        return PolicyAction.DENY, f"Sender {envelope.sender_id} not in allowlist"
    return PolicyAction.ALLOW, "Sender authorized"

# Configure boundary
boundary = SecurityBoundary()
boundary.add_rule(size_limit_rule)
boundary.add_rule(sender_allowlist_rule)

# Evaluate message
result = boundary.evaluate(envelope)
print(f"Policy decision: {result.value}")
print(f"Audit trail entries: {len(boundary.audit_log)}")
```

### Direct Peer Transfer Protocol

Minimize routing overhead by establishing direct channels between agents without central mediation.

```python
import socket
import json
from typing import Optional

class PeerAgent:
    """Direct peer-to-peer communication without centralized routing."""
    
    def __init__(self, agent_id: str, listen_port: int):
        self.agent_id = agent_id
        self.listen_port = listen_port
        self.peer_registry: Dict[str, tuple[str, int]] = {}  # agent_id -> (host, port)
        self.security_boundary = SecurityBoundary()
    
    def register_peer(self, peer_id: str, host: str, port: int):
        """Add peer to direct communication registry."""
        self.peer_registry[peer_id] = (host, port)
    
    def send_direct(self, recipient_id: str, message_type: str, payload: dict) -> bool:
        """Send message directly to peer without LLM routing."""
        if recipient_id not in self.peer_registry:
            logging.error(f"Peer {recipient_id} not in registry")
            return False
        
        envelope = MessageEnvelope(
            sender_id=self.agent_id,
            recipient_id=recipient_id,
            message_type=message_type,
            payload=payload
        )
        
        # Stateless transfer - no session context
        host, port = self.peer_registry[recipient_id]
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.connect((host, port))
                sock.sendall(json.dumps({
                    "sender_id": envelope.sender_id,
                    "recipient_id": envelope.recipient_id,
                    "message_type": envelope.message_type,
                    "payload": envelope.payload,
                    "schema_version": envelope.schema_version,
                    "checksum": envelope.checksum
                }).encode())
                return True
        except Exception as e:
            logging.error(f"Direct transfer failed: {e}")
            return False
    
    def receive_direct(self, data: bytes) -> Optional[MessageEnvelope]:
        """Receive and validate message at security boundary."""
        try:
            msg_dict = json.loads(data.decode())
            envelope = MessageEnvelope(**msg_dict)
            
            # Deterministic validation
            if not envelope.validate():
                logging.warning("Message failed checksum validation")
                return None
            
            # Apply security boundary
            if self.security_boundary.evaluate(envelope) != PolicyAction.ALLOW:
                logging.warning(f"Message denied by security policy")
                return None
            
            return envelope
        except Exception as e:
            logging.error(f"Receive failed: {e}")
            return None

# Usage: horizontal scaling by adding peers
agent_a = PeerAgent("agent_001", 5001)
agent_b = PeerAgent("agent_002", 5002)

agent_a.register_peer("agent_002", "localhost", 5002)
agent_b.register_peer("agent_001", "localhost", 5001)

# Direct transfer without central LLM routing
agent_a.send_direct(
    recipient_id="agent_002",
    message_type="data_transfer",
    payload={"result": [1, 2, 3], "status": "complete"}
)
```

### Cost-Optimized Routine Transfer Handler

Bypass token-consuming LLM inference for predictable, high-frequency exchanges.

```python
from typing import Protocol
import time

class TransferHandler(Protocol):
    """Interface for handling routine transfers without LLM overhead."""
    def can_handle(self, envelope: MessageEnvelope) -> bool: ...
    def process(self, envelope: MessageEnvelope) -> dict: ...

class StatusUpdateHandler:
    """Zero-token handler for routine status messages."""
    
    def can_handle(self, envelope: MessageEnvelope) -> bool:
        return envelope.message_type == "status_update"
    
    def process(self, envelope: MessageEnvelope) -> dict:
        """Deterministic processing - no LLM inference required."""
        return {
            "acknowledged": True,
            "timestamp": time.time(),
            "status": envelope.payload.get("status", "unknown")
        }

class FileChunkHandler:
    """Zero-token handler for file transfer chunks."""
    
    def __init__(self):
        self.chunks: Dict[str, List[bytes]] = {}
    
    def can_handle(self, envelope: MessageEnvelope) -> bool:
        return envelope.message_type == "file_chunk"
    
    def process(self, envelope: MessageEnvelope) -> dict:
        """Accumulate chunks without LLM interpretation."""
        file_id = envelope.payload.get("file_id")
        chunk_data = envelope.payload.get("data")
        chunk_index = envelope.payload.get("index")
        
        if file_id not in self.chunks:
            self.chunks[file_id] = []
        
        self.chunks[file_id].append((chunk_index, chunk_data))
        
        return {
            "received": True,
            "chunk_index": chunk_index,
            "total_chunks": len(self.chunks[file_id])
        }

class RouterlessDispatcher:
    """Dispatch messages to handlers without LLM routing logic."""
    
    def __init__(self):
        self.handlers: List[TransferHandler] = []
        self.token_savings = 0  # Track avoided LLM inference costs
    
    def register_handler(self, handler: TransferHandler):
        self.handlers.append(handler)
    
    def dispatch(self, envelope: MessageEnvelope) -> dict:
        """Route using deterministic logic, avoiding token costs."""
        for handler in self.handlers:
            if handler.can_handle(envelope):
                result = handler.process(envelope)
                # Estimate token savings (typical routing ~50-100 tokens)
                self.token_savings += 75  
                return result
        
        return {"error": "No handler for message type"}

# Configure zero-LLM routing
dispatcher = RouterlessDispatcher()
dispatcher.register_handler(StatusUpdateHandler())
dispatcher.register_handler(FileChunkHandler())

# Process routine transfers without token costs
test_envelope = MessageEnvelope(
    sender_id="agent_001",
    recipient_id="agent_002",
    message_type="status_update",
    payload={"status": "processing", "progress": 0.75}
)

result = dispatcher.dispatch(test_envelope)
print(f"Result: {result}")
print(f"Cumulative token savings: {dispatcher.token_savings}")
```

## Concept Reference

| Concept | Technical | Plain | Importance |
|---------|-----------|-------|------------|
| mIRC FTP-inspired architecture | A decentralized communication protocol design pattern derived from early Internet relay chat file transfer mechanisms, characterized by direct peer-to-peer connections | A way for programs to talk directly to each other, like how people used to share files in old chat rooms without needing a middleman server | 0.95 |
| single point of failure | An architectural component whose malfunction or unavailability causes complete system failure, representing a critical reliability bottleneck in the design | One critical piece that, if it breaks, brings down the entire system—like having only one bridge across a river | 0.92 |
| AI agent communication | Structured information exchange protocols between autonomous computational entities capable of goal-directed behavior, typically involving serialized data | The way smart computer programs send messages and information to each other to work together on tasks | 0.92 |
| deterministic security block layers | Rule-based authorization and validation modules that apply predefined, reproducible security policies at architectural boundaries without probabilistic inference | Security checkpoints that follow fixed rules to check if data is safe and allowed to pass, giving the same answer every time for the same input | 0.91 |
| peer-to-peer data transfer | A distributed networking paradigm where nodes exchange data directly without hierarchical intermediation, enabling symmetric communication relationships | When two computers share information directly with each other instead of sending it through a central hub or server | 0.90 |
| attack surface minimization | Security design principle reducing the total number of exploitable entry points, interfaces, and code paths exposed to potential adversaries | Making a system safer by reducing the number of ways hackers could potentially break in or cause problems | 0.90 |
| stateless transport primitive | A communication abstraction that carries no session context between invocations, treating each data transfer as an independent operation | A delivery method that forgets everything after each delivery, treating every new message as if it's the first time | 0.89 |
| dumb-pipe architecture | A transport layer design philosophy that provides minimal processing intelligence, acting solely as a data conduit without interpretation | A simple delivery system that just moves information from point A to point B without trying to understand or change what it's carrying | 0.88 |
| low-latency substrate | An underlying infrastructure layer optimized for minimal message propagation delay through architectural choices that reduce processing overhead | A foundation system designed to move messages extremely quickly with very little waiting time between sending and receiving | 0.88 |
| infrastructure-first design | A systems engineering methodology prioritizing foundational transport, storage, and communication primitives before application-layer feature development | Building the basic plumbing and foundation of a system first, making sure it's solid before adding the fancy features on top | 0.87 |
| centralized mediation patterns | Architectural topologies where a dedicated intermediary service must participate in and coordinate all inter-component interactions | A setup where all communication must go through one central service that manages and controls all the conversations between different parts | 0.87 |

## Glossary

| Term | Definition | Concept IDs |
|------|------------|-------------|
| dumb-pipe transport | A communication layer that moves data without interpreting content, inspired by IRC file transfers, providing minimal processing intelligence | 1, 2, 8 |
| direct agent peering | Symmetric communication relationships where AI agents exchange structured data directly without hierarchical intermediaries | 3, 4, 17 |
| LLM routing elimination | Architectural choice to bypass language model inference for message forwarding, avoiding token costs and processing delays | 5, 13, 14 |
| deterministic security boundaries | Rule-based validation checkpoints using explicit conditional logic rather than learned models, enabling reproducible security decisions | 6, 9 |
| infrastructure-first methodology | Systems design prioritizing foundational transport primitives and reliability constraints before application features | 7, 11 |
| stateless message handling | Transport abstraction treating each data transfer independently without session memory, enabling horizontal scaling | 8, 10 |
| centralization risk | Reliability and availability vulnerabilities created when all communication depends on a single mediation service | 15, 16 |
| attack surface reduction | Security design minimizing exploitable entry points through architectural simplification, fewer interfaces, and restricted code paths | 12 |
| token cost overhead | Computational expenses from LLM inference on every message during routing or mediation, directly impacting API billing | 13 |
| horizontal scaling substrate | Infrastructure enabling capacity growth by adding parallel nodes rather than enhancing individual components | 10, 11 |
| auditable guardrails | Transparent policy enforcement using inspectable conditional logic rather than opaque model weights, supporting compliance verification | 6, 9 |
| structured data schema | Predefined information formats with explicit fields and type constraints enabling deterministic parsing without natural language understanding | 17 |
| routine transfer optimization | Architectural focus on high-frequency, predictable data exchanges that follow established patterns and don't require complex interpretation | 14 |

## Edge Cases & Warnings

- ⚠️ **Not suitable for complex negotiation**: When agents need nuanced interpretation, context understanding, or dynamic decision-making, LLM routing may be necessary despite cost overhead
- ⚠️ **Schema evolution challenges**: Stateless primitives require careful versioning; breaking changes to message schemas can orphan old peers without centralized upgrade coordination
- ⚠️ **Security rule brittleness**: Deterministic guardrails must be explicitly updated for new attack patterns; they lack the adaptability of model-based anomaly detection
- ⚠️ **Peer discovery complexity**: Direct peering requires registry mechanisms or service discovery; fully decentralized topologies need additional coordination protocols
- ⚠️ **Debugging distributed flows**: Without centralized logging, tracing message paths across peer networks requires instrumentation at each node
- ⚠️ **Partial failure handling**: Stateless design means no automatic retry or acknowledgment; application layer must implement reliability patterns if needed
- ⚠️ **Initial overhead trade-off**: Infrastructure-first approach delays feature delivery; only justified when scale, cost, or reliability requirements are clear

## Quick Reference

```python
from dataclasses import dataclass
import hashlib
import json

# 1. Define stateless message envelope
@dataclass
class Msg:
    sender: str
    recipient: str
    type: str
    data: dict
    
    @property
    def checksum(self) -> str:
        return hashlib.sha256(
            json.dumps(self.__dict__, sort_keys=True).encode()
        ).hexdigest()

# 2. Implement deterministic validation
def validate(msg: Msg) -> bool:
    allowed_types = ["status", "data", "file"]
    return msg.type in allowed_types and len(msg.sender) > 0

# 3. Direct peer transfer (no LLM routing)
def send_direct(msg: Msg, peer_addr: tuple[str, int]):
    if validate(msg):
        # Send via socket/HTTP/gRPC - no token cost
        print(f"✓ Sent to {peer_addr} without LLM overhead")
    else:
        print("✗ Validation failed")

# Usage
message = Msg("agent_a", "agent_b", "status", {"progress": 0.5})
send_direct(message, ("192.168.1.10", 8080))
```

---
_Generated by Philosopher's Stone v4 — EchoSeed_
Philosopher's Stone v4 × Skill Forge × EchoSeed
