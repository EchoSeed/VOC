# Dispatch-Bounded Agent Design

> Trigger this skill when building AI assistants that need to respond quickly within narrow domains while maintaining honest capability boundaries. Use when system requirements prioritize low-latency responses over comprehensive coverage, when input quality varies significantly, or when you need to prevent confident fabrication from weak queries. Essential for production systems where silent failures erode user trust and where different query types demand different cognitive processing modes.

## Overview

This skill teaches you to architect AI agents using **cognitive stratification**: layering fast-narrow specialist processors with slow-broad generalist reasoning, unified through intelligent dispatch routing. The core principle is **honest boundary management**—systems that explicitly refuse tasks beyond their validated capabilities rather than fabricating confident output from insufficient input. This approach achieves superior performance by matching processing mode to task requirements, preventing the silent failures that plague monolithic "do-everything" architectures. The result is a trustworthy assistant that operates quickly within its domain while gracefully degrading and transparently signaling its limitations.

## When to Use

- Building production chatbots or assistants where response latency directly impacts user experience
- Designing systems that receive highly variable input quality (malformed queries, insufficient context, ambiguous requests)
- Architecting agents that must maintain user trust by avoiding hallucinations and confident fabrications
- Creating specialist tools that need clear capability boundaries rather than attempting universal competence
- Implementing multi-tier cognitive systems where some queries need quick answers and others require deep reasoning
- Developing APIs or services where graceful degradation is preferable to cascading failures
- Refactoring monolithic agents that suffer from unpredictable behavior or overreach their training domains

## Core Workflow

1. **Input Classification**: Analyze incoming queries for quality (well-formed vs. malformed), information density (sufficient vs. insufficient), and domain fit (in-scope vs. out-of-scope)
2. **Dispatch Routing**: Based on classification, route to appropriate handler: fast-narrow specialist for constrained tasks, slow-broad generalist for complex problems, or explicit refusal for out-of-bounds requests
3. **Confidence Thresholding**: Before generating output, validate internal confidence scores against task requirements; refuse or downgrade response if confidence is below threshold
4. **Bounded Processing**: Execute within the selected cognitive mode, maintaining strict domain constraints and refusing to extrapolate beyond validated capabilities
5. **Transparent Signaling**: Return responses with explicit capability indicators—success with confidence level, partial success with limitations noted, or structured refusal with clear reasoning

## Key Patterns

### Fast-Narrow Specialist Handler

Create focused processors optimized for minimal latency within constrained domains. These handlers use shallow reasoning chains and limited context windows to achieve high throughput.

```python
from typing import Optional, Literal
from dataclasses import dataclass

@dataclass
class SpecialistResponse:
    status: Literal["success", "insufficient_input", "out_of_scope"]
    output: Optional[str]
    confidence: float
    latency_ms: float

class FastNarrowHandler:
    """Specialized handler optimized for low-latency responses in constrained domain."""
    
    def __init__(self, domain: str, max_latency_ms: float = 100):
        self.domain = domain
        self.max_latency_ms = max_latency_ms
        self.validated_patterns = self._load_domain_patterns()
    
    def _load_domain_patterns(self) -> set[str]:
        """Load pre-validated query patterns for this domain."""
        # In production, load from configuration or trained classifier
        return {"pattern_1", "pattern_2", "pattern_3"}
    
    def can_handle(self, query: str) -> tuple[bool, float]:
        """Check if query matches domain patterns. Returns (can_handle, confidence)."""
        # Simplified pattern matching; production would use trained classifier
        normalized = query.lower().strip()
        for pattern in self.validated_patterns:
            if pattern in normalized:
                return True, 0.92
        return False, 0.0
    
    def process(self, query: str) -> SpecialistResponse:
        """Process query with fast-narrow logic, enforcing latency constraints."""
        import time
        start = time.perf_counter()
        
        # Check domain fitness
        can_handle, confidence = self.can_handle(query)
        if not can_handle:
            return SpecialistResponse(
                status="out_of_scope",
                output=None,
                confidence=0.0,
                latency_ms=(time.perf_counter() - start) * 1000
            )
        
        # Check input quality (simplified)
        if len(query.split()) < 3:
            return SpecialistResponse(
                status="insufficient_input",
                output=None,
                confidence=0.0,
                latency_ms=(time.perf_counter() - start) * 1000
            )
        
        # Execute constrained processing (mock fast operation)
        output = f"Fast response for: {query[:50]}"
        
        elapsed_ms = (time.perf_counter() - start) * 1000
        return SpecialistResponse(
            status="success",
            output=output,
            confidence=confidence,
            latency_ms=elapsed_ms
        )
```

### Dispatch-Based Routing Architecture

Implement intelligent request classification and routing to match queries with appropriate cognitive processors.

```python
from typing import Protocol, Union
from enum import Enum

class CognitiveMode(Enum):
    FAST_NARROW = "fast_narrow"
    SLOW_BROAD = "slow_broad"
    REFUSE = "refuse"

class QueryHandler(Protocol):
    """Interface for cognitive processors."""
    def can_handle(self, query: str) -> tuple[bool, float]: ...
    def process(self, query: str) -> SpecialistResponse: ...

@dataclass
class RoutingDecision:
    mode: CognitiveMode
    handler: Optional[QueryHandler]
    reason: str
    confidence: float

class DispatchRouter:
    """Routes queries to appropriate cognitive processors based on classification."""
    
    def __init__(self):
        self.fast_handlers: list[QueryHandler] = []
        self.slow_handler: Optional[QueryHandler] = None
        self.confidence_threshold = 0.7
    
    def register_fast_handler(self, handler: QueryHandler) -> None:
        """Register a fast-narrow specialist handler."""
        self.fast_handlers.append(handler)
    
    def register_slow_handler(self, handler: QueryHandler) -> None:
        """Register the slow-broad generalist handler."""
        self.slow_handler = handler
    
    def classify_query(self, query: str) -> RoutingDecision:
        """Classify query and determine routing decision."""
        # Input quality checks
        words = query.split()
        if len(words) < 2:
            return RoutingDecision(
                mode=CognitiveMode.REFUSE,
                handler=None,
                reason="Insufficient input: query too short",
                confidence=1.0
            )
        
        # Try fast-narrow handlers first (optimization for common cases)
        for handler in self.fast_handlers:
            can_handle, confidence = handler.can_handle(query)
            if can_handle and confidence >= self.confidence_threshold:
                return RoutingDecision(
                    mode=CognitiveMode.FAST_NARROW,
                    handler=handler,
                    reason=f"Matched specialist domain with {confidence:.2f} confidence",
                    confidence=confidence
                )
        
        # Fall back to slow-broad if available
        if self.slow_handler:
            can_handle, confidence = self.slow_handler.can_handle(query)
            if can_handle:
                return RoutingDecision(
                    mode=CognitiveMode.SLOW_BROAD,
                    handler=self.slow_handler,
                    reason="Requires generalist processing",
                    confidence=confidence
                )
        
        # No handler can process this query
        return RoutingDecision(
            mode=CognitiveMode.REFUSE,
            handler=None,
            reason="Query outside validated capability boundaries",
            confidence=0.0
        )
    
    def route(self, query: str) -> Union[SpecialistResponse, dict]:
        """Execute full routing pipeline."""
        decision = self.classify_query(query)
        
        if decision.mode == CognitiveMode.REFUSE:
            return {
                "status": "refused",
                "reason": decision.reason,
                "suggestion": "Please rephrase with more detail or context"
            }
        
        # Execute selected handler
        response = decision.handler.process(query)
        
        # Validate output confidence
        if response.confidence < self.confidence_threshold:
            return {
                "status": "low_confidence",
                "reason": f"Confidence {response.confidence:.2f} below threshold {self.confidence_threshold}",
                "partial_output": response.output,
                "suggestion": "Response may be unreliable; consider slow-broad mode"
            }
        
        return response
```

### Honest Boundary Enforcement

Implement explicit refusal mechanisms and confidence thresholding to prevent confident fabrication from weak input.

```python
from typing import Optional

@dataclass
class BoundedResponse:
    """Response type with explicit capability signaling."""
    success: bool
    output: Optional[str]
    confidence: float
    within_bounds: bool
    limitations: list[str]
    metadata: dict

class HonestBoundaryManager:
    """Enforces capability boundaries and prevents overreach."""
    
    def __init__(
        self,
        min_confidence: float = 0.7,
        max_uncertainty: float = 0.4,
        validated_domains: set[str] = None
    ):
        self.min_confidence = min_confidence
        self.max_uncertainty = max_uncertainty
        self.validated_domains = validated_domains or set()
    
    def validate_output(
        self,
        query: str,
        proposed_output: str,
        internal_confidence: float,
        domain: str
    ) -> BoundedResponse:
        """Validate proposed output against capability boundaries."""
        limitations = []
        within_bounds = True
        
        # Domain validation
        if domain not in self.validated_domains:
            limitations.append(f"Domain '{domain}' not in validated set")
            within_bounds = False
        
        # Confidence validation
        if internal_confidence < self.min_confidence:
            limitations.append(
                f"Confidence {internal_confidence:.2f} below threshold {self.min_confidence}"
            )
            within_bounds = False
        
        # Uncertainty validation (could come from ensemble disagreement, etc.)
        uncertainty = 1.0 - internal_confidence
        if uncertainty > self.max_uncertainty:
            limitations.append(f"Uncertainty {uncertainty:.2f} exceeds maximum {self.max_uncertainty}")
            within_bounds = False
        
        # Input quality checks
        if len(query.split()) < 3:
            limitations.append("Input query lacks sufficient detail")
            within_bounds = False
        
        # Decide whether to return output or refuse
        if not within_bounds:
            return BoundedResponse(
                success=False,
                output=None,
                confidence=internal_confidence,
                within_bounds=False,
                limitations=limitations,
                metadata={
                    "refusal_reason": "Output validation failed",
                    "boundary_violations": len(limitations)
                }
            )
        
        return BoundedResponse(
            success=True,
            output=proposed_output,
            confidence=internal_confidence,
            within_bounds=True,
            limitations=[],
            metadata={"validation_passed": True}
        )
    
    def graceful_degrade(
        self,
        full_response: str,
        confidence: float
    ) -> BoundedResponse:
        """Provide partial response with explicit limitation signaling."""
        # Extract high-confidence portions (simplified)
        high_conf_portion = full_response[:len(full_response)//2]
        
        return BoundedResponse(
            success=True,  # Partial success
            output=high_conf_portion,
            confidence=confidence,
            within_bounds=True,
            limitations=[
                "Response truncated to high-confidence content only",
                "Full answer would exceed validated capability boundaries"
            ],
            metadata={
                "degraded": True,
                "original_length": len(full_response),
                "returned_length": len(high_conf_portion)
            }
        )
```

### Complete Hybrid Architecture

Integrate all patterns into a cohesive multi-tier cognitive system.

```python
class HybridCognitiveAgent:
    """Complete implementation of stratified intelligence architecture."""
    
    def __init__(self):
        self.router = DispatchRouter()
        self.boundary_manager = HonestBoundaryManager(
            min_confidence=0.75,
            validated_domains={"weather", "math", "definitions"}
        )
        
        # Register specialist handlers
        self.router.register_fast_handler(
            FastNarrowHandler(domain="weather", max_latency_ms=100)
        )
        self.router.register_fast_handler(
            FastNarrowHandler(domain="math", max_latency_ms=150)
        )
        
        # In production, would register slow-broad handler here
        # self.router.register_slow_handler(SlowBroadHandler())
    
    def process_query(self, query: str) -> dict:
        """Main entry point: classify, route, validate, and respond."""
        # Step 1: Route to appropriate handler
        routing_result = self.router.route(query)
        
        # Handle refusal from routing
        if isinstance(routing_result, dict) and routing_result.get("status") == "refused":
            return {
                "status": "refused",
                "message": routing_result["reason"],
                "suggestion": routing_result["suggestion"],
                "honest_boundary": True
            }
        
        # Step 2: Validate output against boundaries
        specialist_response = routing_result
        validated = self.boundary_manager.validate_output(
            query=query,
            proposed_output=specialist_response.output,
            internal_confidence=specialist_response.confidence,
            domain="weather"  # Would extract from handler metadata
        )
        
        # Step 3: Return with explicit capability signaling
        if not validated.within_bounds:
            return {
                "status": "refused",
                "message": "Query exceeds validated capability boundaries",
                "limitations": validated.limitations,
                "honest_boundary": True,
                "suggestion": "Try rephrasing or providing more context"
            }
        
        return {
            "status": "success",
            "output": validated.output,
            "confidence": validated.confidence,
            "latency_ms": specialist_response.latency_ms,
            "mode": "fast_narrow",
            "honest_boundary": True,
            "metadata": validated.metadata
        }
```

## Concept Reference

| Concept | Technical | Plain | Importance |
|---------|-----------|-------|------------|
| narrow, fast-response assistant architecture | A constrained agent design pattern optimizing for minimal latency and reduced cognitive overhead by limiting operational scope to well-defined task domains | A focused helper system built to answer quickly within a specific area rather than trying to do everything, like a specialist rather than a generalist | 0.95 |
| hybrid cognitive architecture | Composite system design integrating multiple reasoning modalities with distinct computational characteristics, implementing routing logic to match processing mode to task requirements | A smart system that combines fast simple thinking with slow deep thinking, choosing which to use based on what's needed | 0.93 |
| honestly bounded systems | Architectures with explicit capability frontiers and transparent limitation signaling, implementing confidence thresholding and uncertainty quantification | Systems that know their limits and honestly tell you when they can't help or don't know something, instead of pretending they can do everything | 0.91 |
| complementary capability composition | Architectural principle leveraging synergistic integration of specialized subsystems with non-overlapping strength profiles, achieving emergent capabilities through strategic combination | Combining different specialized tools that are good at different things, so together they're more capable than any single tool alone | 0.90 |
| intelligence layering | Hierarchical cognitive architecture strategy employing multiple processing tiers with distinct speed-accuracy tradeoffs, enabling dynamic resource allocation based on task characteristics | Using different levels of thinking—some quick and simple, others slow and deep—working together as a team | 0.89 |
| dispatch-based agent | An architectural pattern employing request routing logic to direct queries to specialized subcomponents or handlers based on input classification, enabling modular capability composition | A system that reads your question and sends it to the right specialized tool or function, like a receptionist directing calls to different departments | 0.88 |
| fabricated confident output | Generated responses exhibiting high epistemic certainty markers despite insufficient evidential basis or low internal confidence scores, representing hallucination or overreach | When a system makes up an answer and presents it as if it's definitely correct, even though it's actually just guessing or doesn't have good information | 0.87 |
| slow-broad approach | Generalist processing mode employing extensive deliberation, comprehensive context integration, and multi-step reasoning across diverse knowledge domains | Careful, thorough thinking that takes time to consider many angles and handle complex, unfamiliar problems | 0.86 |
| silent failure | Error condition where a system produces output despite detecting internal constraint violations or confidence thresholds breaches, masking failure state from users | When something goes wrong but the system pretends everything is fine and gives you an answer anyway without telling you there was a problem | 0.85 |
| fast-narrow approach | Specialized processing mode optimized for high-throughput, low-latency operations within constrained domains, utilizing shallow reasoning chains and limited context integration | Quick, focused thinking that handles simple, specific tasks rapidly without deep analysis | 0.84 |

## Glossary

| Term | Definition | Concept IDs |
|------|------------|-------------|
| narrow specialist architecture | A constrained agent design optimizing for speed within well-defined domains, trading generality for deterministic performance | 1, 12 |
| dispatch routing | Request classification logic that directs queries to appropriate specialized handlers, enabling modular capability composition | 4, 17 |
| honest boundaries | Explicit capability frontiers with transparent limitation signaling, preventing output generation beyond validated operational domains | 8, 15 |
| silent failure | Error masking where systems produce output despite internal constraint violations, hiding failure states from users | 7, 9 |
| input quality tiers | Classification of queries by information density and constraint specification, from malformed to well-formed | 5, 6, 10 |
| cognitive stratification | Hierarchical processing architecture employing multiple reasoning tiers with distinct speed-accuracy tradeoffs | 11, 14 |
| fast-narrow mode | High-throughput, low-latency processing optimized for constrained domains using shallow reasoning chains | 12, 1 |
| slow-broad mode | Deliberative processing employing comprehensive context integration across diverse knowledge domains | 13, 14 |
| friction minimization | Reducing interaction cost through simplified interfaces, minimal decision points, and low cognitive load | 2, 3 |
| confident fabrication | Generated responses exhibiting high certainty markers despite insufficient evidential basis or low internal confidence | 9, 7 |
| explicit refusal | Formal rejection responses when task requirements exceed operational constraints, enforcing capability boundaries | 15, 8 |
| graceful degradation | Maintaining partial functionality with explicit capability signaling under suboptimal conditions rather than cascading failures | 16, 8 |
| complementary composition | Strategic integration of specialized subsystems with non-overlapping strengths, achieving emergent capabilities through synergistic combination | 17, 14 |

## Edge Cases & Warnings

- ⚠️ **Threshold Miscalibration**: Setting confidence thresholds too high causes excessive refusals and degrades user experience; too low allows confident fabrication. Calibrate against validation datasets with known ground truth.
- ⚠️ **Routing Ambiguity**: Queries that match multiple specialist domains with similar confidence create dispatch conflicts. Implement tie-breaking logic (e.g., prefer fastest handler, escalate to slow-broad, or request user clarification).
- ⚠️ **Graceful Degradation Abuse**: Over-reliance on partial responses can mask systemic capability gaps. Monitor degradation rates and investigate domains with consistently high degradation as candidates for specialist handler development.
- ⚠️ **Silent Scope Creep**: Specialists may gradually accept out-of-domain queries during operation if domain boundaries aren't continuously validated. Implement automated testing with out-of-scope queries to detect boundary drift.
- ⚠️ **Latency Cascade**: Fast-narrow handlers that fail and escalate to slow-broad processing create unpredictable latency. Set explicit timeouts and consider returning "processing" status for long-running slow-broad tasks.
- ⚠️ **User Trust Erosion from Over-Refusal**: While honest boundaries build trust, excessive refusals frustrate users. Balance boundary enforcement with helpful guidance (suggest rephrasing, provide related information within scope).
- ⚠️ **Confidence Score Miscalibration**: Internal confidence scores from ML models often don't align with actual accuracy. Validate confidence-to-accuracy correlation and apply calibration transforms (e.g., Platt scaling, temperature scaling).
- ⚠️ **Input Classification Brittleness**: Simple heuristics for malformed/insufficient input detection fail on edge cases. Invest in robust input validation and consider learning-based quality classifiers trained on production data.

## Quick Reference

```python
# Minimal dispatch-bounded agent pattern
from typing import Optional, Literal

class MinimalBoundedAgent:
    def __init__(self, confidence_threshold: float = 0.75):
        self.threshold = confidence_threshold
    
    def process(self, query: str) -> dict:
        # 1. Validate input quality
        if len(query.split()) < 3:
            return {"status": "refused", "reason": "insufficient_input"}
        
        # 2. Attempt processing (mock)
        confidence = 0.85  # Would come from actual model
        output = f"Response to: {query}"
        
        # 3. Enforce honest boundaries
        if confidence < self.threshold:
            return {
                "status": "refused",
                "reason": "low_confidence",
                "confidence": confidence
            }
        
        return {
            "status": "success",
            "output": output,
            "confidence": confidence,
            "honest_boundary": True
        }

# Usage
agent = MinimalBoundedAgent(confidence_threshold=0.75)
result = agent.process("What is the weather?")
print(result)  # Honest response or explicit refusal
```

---
_Generated by Philosopher's Stone v4 — EchoSeed_
Philosopher's Stone v4 × Skill Forge × EchoSeed
