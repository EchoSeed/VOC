# Model Context Protocol

> Trigger this skill when designing or implementing AI systems that need to maintain conversational context across multiple interactions, integrate with other AI components, or ensure standardized communication between models. Use when building multi-turn dialogue systems, creating interoperable AI architectures, or establishing session management for contextual AI applications. Essential for developers working on conversational AI, chatbots, AI agents, or any system requiring persistent state and semantic coherence across exchanges.

## Overview

The Model Context Protocol (MCP) establishes a standardized framework for AI models to exchange contextual information and maintain conversational state throughout multi-turn interactions. This skill enables you to implement systematic context preservation, ensuring AI systems can track conversation history, maintain semantic coherence, and deliver contextually-aware responses. By following MCP specifications, you create interoperable AI components that communicate through uniform message formats and data exchange patterns, reducing integration complexity while enabling session continuity across temporal boundaries and system interruptions.

## When to Use

- Building conversational AI systems requiring memory of previous exchanges
- Integrating multiple AI models or components that need to share contextual information
- Implementing multi-turn dialogue systems with persistent state management
- Creating standardized interfaces between AI services and client applications
- Designing systems where context accumulation influences response generation
- Ensuring semantic coherence across long-running conversation sessions
- Developing interoperable AI architectures that must work with diverse components
- Managing session continuity across system restarts or network interruptions
- Implementing chatbots, virtual assistants, or AI agents with conversational memory
- Establishing communication protocols for distributed AI systems
- Creating APIs for AI models that require contextual awareness

## Core Workflow

1. **Initialize Context Container**: Create a structured data object to hold conversation state, including message history, entity tracking, and session metadata
2. **Define Message Format**: Establish standardized message schemas with required fields (timestamp, speaker ID, content, context references) for all communications
3. **Implement State Persistence**: Store contextual information at each interaction turn, accumulating conversation history and updating entity relationships
4. **Process Contextual Queries**: When receiving new input, retrieve relevant context from the container and inject it into the model's processing pipeline
5. **Maintain Semantic Coherence**: Validate that new responses align logically with accumulated context and update state accordingly
6. **Handle Session Boundaries**: Implement serialization mechanisms to preserve context across system interruptions or temporal gaps
7. **Expose Standardized Interface**: Provide well-defined API endpoints or method signatures for external systems to interact with the context protocol

## Key Patterns

### Context Container Pattern

Maintain a structured data object that encapsulates all conversational state, providing versioned access to historical context while supporting efficient retrieval.

```python
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum

class MessageRole(Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"

@dataclass
class Message:
    """Individual message unit with contextual metadata"""
    role: MessageRole
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    context_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role.value,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
            "context_id": self.context_id
        }

@dataclass
class ContextContainer:
    """Main context preservation structure following MCP specification"""
    session_id: str
    messages: List[Message] = field(default_factory=list)
    entities: Dict[str, Any] = field(default_factory=dict)
    state_variables: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    
    def add_message(self, role: MessageRole, content: str, 
                   metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add message to context and update timestamp"""
        msg = Message(
            role=role,
            content=content,
            metadata=metadata or {},
            context_id=self.session_id
        )
        self.messages.append(msg)
        self.last_updated = datetime.now()
    
    def get_recent_context(self, n: int = 5) -> List[Message]:
        """Retrieve last n messages for context window"""
        return self.messages[-n:]
    
    def extract_entities(self, entity_type: str) -> List[Any]:
        """Retrieve specific entity type from context"""
        return self.entities.get(entity_type, [])
    
    def serialize(self) -> Dict[str, Any]:
        """Export context for persistence or transmission"""
        return {
            "session_id": self.session_id,
            "messages": [msg.to_dict() for msg in self.messages],
            "entities": self.entities,
            "state_variables": self.state_variables,
            "created_at": self.created_at.isoformat(),
            "last_updated": self.last_updated.isoformat()
        }
```

### Protocol Message Formatter

Standardize message structure for consistent communication between AI components, ensuring all exchanges conform to MCP specifications.

```python
from typing import Protocol, TypedDict, Literal
from uuid import uuid4

class MCPMessage(TypedDict):
    """Type-safe message format adhering to MCP standard"""
    message_id: str
    session_id: str
    role: Literal["user", "assistant", "system"]
    content: str
    timestamp: str
    context_refs: list[str]
    metadata: dict[str, Any]

class MessageFormatter(Protocol):
    """Protocol defining standardized message formatting interface"""
    
    def format_message(self, content: str, role: str, 
                      context: ContextContainer) -> MCPMessage:
        ...
    
    def parse_message(self, raw_message: Dict[str, Any]) -> Message:
        ...

class StandardMessageFormatter:
    """Concrete implementation of MCP message formatting"""
    
    def format_message(self, content: str, role: str, 
                      context: ContextContainer) -> MCPMessage:
        """Transform content into MCP-compliant message structure"""
        # Extract relevant context references from recent messages
        recent = context.get_recent_context(3)
        context_refs = [msg.context_id for msg in recent if msg.context_id]
        
        return MCPMessage(
            message_id=str(uuid4()),
            session_id=context.session_id,
            role=role,
            content=content,
            timestamp=datetime.now().isoformat(),
            context_refs=context_refs,
            metadata={
                "message_count": len(context.messages),
                "entity_count": len(context.entities)
            }
        )
    
    def parse_message(self, raw_message: Dict[str, Any]) -> Message:
        """Convert external message format to internal Message object"""
        return Message(
            role=MessageRole(raw_message["role"]),
            content=raw_message["content"],
            timestamp=datetime.fromisoformat(raw_message["timestamp"]),
            metadata=raw_message.get("metadata", {}),
            context_id=raw_message.get("session_id")
        )
```

### State Synchronization Pattern

Maintain consistency between in-memory context and persistent storage, enabling session continuity across system boundaries.

```python
import json
from pathlib import Path
from typing import Optional
import pickle

class ContextPersistenceManager:
    """Manages serialization and recovery of context state"""
    
    def __init__(self, storage_path: str = "./context_store"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
    
    def save_context(self, context: ContextContainer, 
                    format: Literal["json", "pickle"] = "json") -> None:
        """Persist context to disk for session continuity"""
        filepath = self.storage_path / f"{context.session_id}.{format}"
        
        if format == "json":
            with open(filepath, 'w') as f:
                json.dump(context.serialize(), f, indent=2)
        elif format == "pickle":
            with open(filepath, 'wb') as f:
                pickle.dump(context, f)
    
    def load_context(self, session_id: str, 
                    format: Literal["json", "pickle"] = "json") -> Optional[ContextContainer]:
        """Restore context from persistent storage"""
        filepath = self.storage_path / f"{session_id}.{format}"
        
        if not filepath.exists():
            return None
        
        if format == "json":
            with open(filepath, 'r') as f:
                data = json.load(f)
            return self._deserialize_context(data)
        elif format == "pickle":
            with open(filepath, 'rb') as f:
                return pickle.load(f)
    
    def _deserialize_context(self, data: Dict[str, Any]) -> ContextContainer:
        """Reconstruct ContextContainer from serialized data"""
        context = ContextContainer(session_id=data["session_id"])
        context.entities = data["entities"]
        context.state_variables = data["state_variables"]
        context.created_at = datetime.fromisoformat(data["created_at"])
        context.last_updated = datetime.fromisoformat(data["last_updated"])
        
        # Reconstruct message history
        for msg_data in data["messages"]:
            msg = Message(
                role=MessageRole(msg_data["role"]),
                content=msg_data["content"],
                timestamp=datetime.fromisoformat(msg_data["timestamp"]),
                metadata=msg_data["metadata"],
                context_id=msg_data.get("context_id")
            )
            context.messages.append(msg)
        
        return context
```

### Interoperability Interface

Create standardized API endpoints that expose context protocol functionality to external systems while maintaining encapsulation.

```python
from abc import ABC, abstractmethod
from typing import Callable, Optional

class MCPInterface(ABC):
    """Abstract base class defining MCP-compliant interface contract"""
    
    @abstractmethod
    def send_message(self, content: str, role: str = "user") -> MCPMessage:
        """Submit message to AI model with automatic context injection"""
        pass
    
    @abstractmethod
    def get_context_window(self, window_size: int = 10) -> List[Message]:
        """Retrieve recent conversation history"""
        pass
    
    @abstractmethod
    def update_entity(self, entity_type: str, entity_data: Any) -> None:
        """Register or update entity in context tracking"""
        pass
    
    @abstractmethod
    def reset_session(self) -> None:
        """Clear context and initialize new session"""
        pass

class MCPClient(MCPInterface):
    """Concrete implementation providing MCP-compliant API"""
    
    def __init__(self, session_id: Optional[str] = None,
                 persistence_manager: Optional[ContextPersistenceManager] = None):
        self.session_id = session_id or str(uuid4())
        self.context = ContextContainer(session_id=self.session_id)
        self.formatter = StandardMessageFormatter()
        self.persistence = persistence_manager
        
        # Attempt to restore previous session
        if self.persistence:
            restored = self.persistence.load_context(self.session_id)
            if restored:
                self.context = restored
    
    def send_message(self, content: str, role: str = "user") -> MCPMessage:
        """Process user input with full context awareness"""
        # Add to context history
        self.context.add_message(MessageRole(role), content)
        
        # Format according to MCP standard
        mcp_message = self.formatter.format_message(content, role, self.context)
        
        # Persist updated context
        if self.persistence:
            self.persistence.save_context(self.context)
        
        return mcp_message
    
    def get_context_window(self, window_size: int = 10) -> List[Message]:
        """Retrieve recent conversation for display or processing"""
        return self.context.get_recent_context(window_size)
    
    def update_entity(self, entity_type: str, entity_data: Any) -> None:
        """Track entities mentioned in conversation"""
        if entity_type not in self.context.entities:
            self.context.entities[entity_type] = []
        self.context.entities[entity_type].append(entity_data)
        self.context.last_updated = datetime.now()
    
    def reset_session(self) -> None:
        """Initialize fresh context state"""
        self.session_id = str(uuid4())
        self.context = ContextContainer(session_id=self.session_id)
```

## Concept Reference

| Concept | Technical | Plain | Importance |
|---------|-----------|-------|------------|
| Model Context Protocol | A standardized communication protocol that defines how AI models exchange contextual information and maintain state across interactions within a structured framework | A set of rules that helps AI systems share and remember important information during conversations, like a common language for keeping track of what's being discussed | 1.0 |
| Context | The aggregate of relevant information, including conversation history, environmental parameters, and semantic relationships that inform model interpretation and response generation | The background information and previous parts of a conversation that help the AI understand what you're talking about and give better answers | 0.95 |
| Model | A computational artifact, typically based on machine learning architectures, that processes inputs to generate predictions, classifications, or outputs based on learned patterns | An AI system or program that has been trained to understand and respond to information, like a digital brain that can answer questions or perform tasks | 0.9 |
| State Management | The systematic preservation and retrieval of computational state variables across discrete processing cycles to maintain coherence and continuity in multi-turn interactions | Keeping track of where you are in a conversation so the AI remembers what you've talked about from one message to the next | 0.88 |
| Standardization | The establishment of uniform specifications, data formats, and operational procedures to ensure compatibility and reduce integration complexity across heterogeneous systems | Creating common rules and formats that everyone agrees to use so different systems can work together easily | 0.87 |
| Protocol | A formalized set of rules and conventions governing data exchange, message formatting, and communication sequences between system components or networked entities | An agreed-upon set of rules that different computer programs or systems follow to communicate with each other effectively | 0.85 |
| Communication Protocol | A specification defining message syntax, semantics, synchronization, and error handling for information exchange between autonomous agents or system modules | A detailed guide for how different programs send messages to each other, including what format to use and how to handle problems | 0.84 |
| Interoperability | The capability of disparate systems to exchange information and utilize shared data through common interfaces and protocols without loss of functionality or meaning | The ability of different computer systems to work together and share information smoothly | 0.83 |
| Structured Framework | An organized architecture with defined components, interfaces, and data schemas that enforce consistency and interoperability in system operations | An organized system with clear parts and rules that ensures everything works together in a predictable way | 0.82 |
| Multi-turn Conversation | A sequential dialogue session comprising multiple request-response pairs where context accumulates and influences subsequent processing | A conversation with multiple back-and-forth exchanges where earlier messages affect later ones | 0.81 |
| Contextual Information | Structured or unstructured data elements that provide situational awareness, including temporal markers, entity relationships, and discourse metadata | Specific details about the situation, like who's talking, what was said before, and what the conversation is about | 0.8 |
| Semantic Coherence | The logical consistency and meaningful relationships among information elements within context, ensuring interpretability and appropriate inference generation | Making sure all the pieces of information in a conversation fit together logically and make sense as a whole | 0.8 |
| Session Continuity | The maintenance of persistent state and contextual coherence across temporal boundaries and potential system interruptions within an interaction sequence | Keeping a conversation going smoothly even if there are pauses or breaks, so the AI doesn't forget what you were discussing | 0.79 |

## Glossary

| Term | Definition | Concept IDs |
|------|------------|-------------|
| Model Context Protocol | A standardized framework defining how AI models exchange contextual information and maintain conversational state using uniform communication patterns | 1, 2, 3, 8, 9 |
| Context Preservation | The systematic tracking and maintenance of conversation history, situational details, and semantic relationships that inform AI model responses | 4, 5, 6, 17 |
| State Management | The process of preserving computational variables and conversational information across discrete message exchanges to maintain session coherence | 6, 13, 15 |
| Protocol Standardization | The establishment of uniform specifications for message syntax, data formats, and operational procedures to ensure compatibility across AI systems | 2, 9, 10, 14 |
| Structured Communication | The organized exchange of information using defined message formats, interfaces, and data schemas within a consistent architectural framework | 8, 10, 11, 12 |
| Contextual Awareness | The AI model's ability to process and utilize background information, including conversation history and situational parameters, when generating responses | 3, 4, 5, 17 |
| Multi-turn Interaction | Sequential dialogue sessions comprising multiple request-response exchanges where accumulated context influences subsequent model behavior | 7, 13, 15 |
| Interoperability Framework | The capability enabling disparate AI systems to exchange information and maintain functional compatibility through common protocols and interfaces | 14, 2, 9, 16 |
| Message Structure | The application of syntactic and organizational rules to format data payloads into standardized representations suitable for transmission and parsing | 12, 11, 10 |
| Session Continuity | The maintenance of persistent conversational state and contextual coherence across temporal boundaries and potential system interruptions | 15, 6, 13 |
| Semantic Coherence | The logical consistency and meaningful relationships among information elements ensuring interpretability and appropriate inference within context | 17, 4, 5 |
| Interface Definition | Formal specifications of methods, parameters, and behavioral contracts governing how external systems interact programmatically with AI components | 16, 8, 14 |
| Data Exchange Mechanism | The bidirectional transfer of encoded information between AI systems using defined transport protocols, serialization formats, and error handling | 11, 12, 10, 2 |

## Edge Cases & Warnings

- ⚠️ **Context Window Overflow**: Long conversations may exceed model token limits. Implement sliding window strategies or context summarization to maintain recent relevance while preserving critical historical information.
- ⚠️ **Concurrent Session Conflicts**: Multiple clients accessing the same session simultaneously can create state inconsistencies. Use locking mechanisms or versioning to prevent race conditions in context updates.
- ⚠️ **Serialization Format Compatibility**: JSON serialization may lose type information for complex Python objects. Consider using pickle for internal persistence and JSON only for cross-system communication.
- ⚠️ **Temporal Drift**: Timestamps across distributed systems may diverge. Use UTC consistently and implement clock synchronization checks when ordering matters for context reconstruction.
- ⚠️ **Memory Leaks in Long Sessions**: Unbounded message accumulation will exhaust memory. Implement periodic context pruning, archival strategies, or paginated message retrieval for production systems.
- ⚠️ **Sensitive Data Exposure**: Context containers may accumulate PII or confidential information. Implement encryption for persisted state and sanitization policies before logging or transmitting context.
- ⚠️ **Protocol Version Mismatches**: Different components may implement different MCP versions. Include version identifiers in messages and implement backward compatibility or graceful degradation strategies.
- ⚠️ **Entity Tracking Accuracy**: Automatic entity extraction may produce false positives or miss critical entities. Validate entity updates and provide manual correction interfaces for high-stakes applications.
- ⚠️ **Session Restoration Failures**: Corrupted persistence files or missing dependencies can prevent context restoration. Implement fallback mechanisms to initialize fresh sessions when restoration fails.
- ⚠️ **Network Interruption Handling**: Distributed MCP implementations must handle partial message transmission. Use acknowledgment patterns and idempotency keys to ensure exactly-once message processing.

## Quick Reference

```python
from uuid import uuid4
from datetime import datetime

# Initialize MCP-compliant client with persistence
persistence = ContextPersistenceManager("./sessions")
client = MCPClient(persistence_manager=persistence)

# Send messages with automatic context tracking
response1 = client.send_message("What is the weather today?", role="user")
response2 = client.send_message("How about tomorrow?", role="user")

# Track entities mentioned in conversation
client.update_entity("location", {"name": "San Francisco", "type": "city"})
client.update_entity("temporal", {"reference": "tomorrow", "type": "relative"})

# Retrieve context for model processing
recent_history = client.get_context_window(window_size=5)
conversation_summary = "\n".join([f"{msg.role.value}: {msg.content}" 
                                  for msg in recent_history])

# Serialize context for cross-system transmission
context_snapshot = client.context.serialize()

# Restore session from persistent storage
restored_client = MCPClient(
    session_id="existing-session-id",
    persistence_manager=persistence
)

# Access standardized message format
print(f"Session: {response1['session_id']}")
print(f"Message ID: {response1['message_id']}")
print(f"Context Refs: {response1['context_refs']}")
```

---
_Generated by Philosopher's Stone v4 — EchoSeed_
