# Triadic Quality Assurance

> This skill activates when validating technical documentation, inspecting system integration points, or critically evaluating information claims. It combines structural validation (Markdown Auditor), skeptical analysis (Skeptic's Lens), and boundary monitoring (Seam-Watcher) to ensure comprehensive quality control across textual and system architectures. Use when document integrity, logical consistency, or interface reliability must be systematically verified through independent examination processes.

## Overview

Triadic Quality Assurance implements a three-pronged validation framework for technical systems and documentation. It ensures document structural integrity through syntax compliance checking, applies critical evaluation methodologies to prevent uncritical acceptance of claims, and monitors integration boundaries where component failures typically occur. This skill synthesizes auditing, skepticism, and seam-watching into a unified quality assurance process applicable to both written artifacts and system architectures.

## When to Use

- Validating Markdown documents for syntax compliance and formatting correctness
- Examining claims or logical arguments requiring empirical verification
- Inspecting integration points where distinct systems or components connect
- Performing systematic audits to assess compliance with established standards
- Detecting errors, inconsistencies, or anomalies in documents or system boundaries
- Reviewing technical documentation for structural coherence and accuracy
- Monitoring interface boundaries for potential failures or data exchange issues

## Core Workflow

1. **Structural Validation Phase**: Execute Markdown Auditor to verify syntax compliance, formatting correctness, and hierarchical organization against specifications
2. **Critical Evaluation Phase**: Apply Skeptic's Lens to examine logical consistency, question assumptions, and validate claims through evidence-based reasoning
3. **Boundary Inspection Phase**: Deploy Seam-Watcher to monitor integration points, detect interface anomalies, and verify proper component interaction
4. **Synthesis & Reporting**: Aggregate findings from all three validation streams into unified compliance report
5. **Remediation Cycle**: Address identified discrepancies through targeted corrections and re-validation

## Key Patterns

### Document Structure Validation

Systematically verify Markdown documents against formal syntax specifications, checking heading hierarchy, link integrity, code block formatting, and list structure compliance.

```python
from typing import List, Dict, Tuple
import re
from dataclasses import dataclass

@dataclass
class ValidationResult:
    """Represents outcome of structural validation check."""
    is_valid: bool
    line_number: int
    issue_type: str
    message: str

class MarkdownAuditor:
    """Validates Markdown document structural integrity."""
    
    def __init__(self):
        self.heading_pattern = re.compile(r'^(#{1,6})\s+(.+)$')
        self.link_pattern = re.compile(r'\[([^\]]+)\]\(([^\)]+)\)')
        self.code_block_pattern = re.compile(r'^```(\w*)$')
        
    def audit_heading_hierarchy(self, lines: List[str]) -> List[ValidationResult]:
        """Verify heading levels increment properly without skipping."""
        results = []
        prev_level = 0
        
        for i, line in enumerate(lines, 1):
            match = self.heading_pattern.match(line)
            if match:
                current_level = len(match.group(1))
                # Check for hierarchy violations (skipping levels)
                if current_level > prev_level + 1:
                    results.append(ValidationResult(
                        is_valid=False,
                        line_number=i,
                        issue_type="HEADING_SKIP",
                        message=f"Heading jumped from h{prev_level} to h{current_level}"
                    ))
                prev_level = current_level
        
        return results
    
    def audit_code_blocks(self, lines: List[str]) -> List[ValidationResult]:
        """Ensure code blocks are properly opened and closed."""
        results = []
        block_stack = []
        
        for i, line in enumerate(lines, 1):
            if self.code_block_pattern.match(line):
                if block_stack:
                    block_stack.pop()  # Close block
                else:
                    block_stack.append(i)  # Open block
        
        # Report unclosed code blocks
        for line_num in block_stack:
            results.append(ValidationResult(
                is_valid=False,
                line_number=line_num,
                issue_type="UNCLOSED_CODE_BLOCK",
                message="Code block opened but never closed"
            ))
        
        return results

# Usage example
auditor = MarkdownAuditor()
document_lines = [
    "# Main Heading",
    "### Subheading",  # Skips h2 - violation
    "```python",
    "def example(): pass",
    "# Code block never closed - violation"
]

hierarchy_issues = auditor.audit_heading_hierarchy(document_lines)
code_block_issues = auditor.audit_code_blocks(document_lines)
```

### Critical Claim Evaluation

Apply systematic doubt and evidence-based reasoning to evaluate information claims, identifying logical fallacies and unsupported assertions.

```python
from typing import List, Set, Optional
from enum import Enum
from dataclasses import dataclass

class FallacyType(Enum):
    """Common logical fallacies to detect."""
    AD_HOMINEM = "attacks person rather than argument"
    STRAW_MAN = "misrepresents opposing position"
    FALSE_DICHOTOMY = "presents only two options when more exist"
    APPEAL_TO_AUTHORITY = "cites authority without evidence"
    CIRCULAR_REASONING = "conclusion restates premise"

@dataclass
class Claim:
    """Represents an assertion requiring verification."""
    text: str
    premises: List[str]
    conclusion: str
    evidence: List[str]

class SkepticsLens:
    """Applies critical evaluation to information claims."""
    
    def __init__(self):
        self.fallacy_indicators = {
            FallacyType.AD_HOMINEM: ["you're just", "of course they'd say", "coming from them"],
            FallacyType.FALSE_DICHOTOMY: ["either", "only two", "must be one or"],
            FallacyType.APPEAL_TO_AUTHORITY: ["expert says", "authority on", "scientist claims"],
        }
    
    def evaluate_claim(self, claim: Claim) -> Dict[str, any]:
        """Systematically assess claim validity through skeptical inquiry."""
        evaluation = {
            "claim": claim.text,
            "premise_strength": self._assess_premises(claim.premises),
            "evidence_quality": self._evaluate_evidence(claim.evidence),
            "logical_consistency": self._check_logic_chain(claim),
            "detected_fallacies": self._scan_fallacies(claim),
            "verification_status": "UNVERIFIED"
        }
        
        # Determine overall verification status
        if (evaluation["premise_strength"] > 0.7 and 
            evaluation["evidence_quality"] > 0.6 and
            not evaluation["detected_fallacies"]):
            evaluation["verification_status"] = "VERIFIED"
        elif evaluation["detected_fallacies"]:
            evaluation["verification_status"] = "REJECTED"
            
        return evaluation
    
    def _assess_premises(self, premises: List[str]) -> float:
        """Score premise quality (0-1 scale)."""
        if not premises:
            return 0.0
        # Check for specificity, measurability, verifiability
        score = sum(1 for p in premises if len(p.split()) > 5) / len(premises)
        return score
    
    def _evaluate_evidence(self, evidence: List[str]) -> float:
        """Score evidence strength (0-1 scale)."""
        if not evidence:
            return 0.0
        # Basic heuristic: evidence with citations/specifics scores higher
        score = sum(1 for e in evidence if any(char.isdigit() for char in e)) / len(evidence)
        return score
    
    def _check_logic_chain(self, claim: Claim) -> bool:
        """Verify conclusion follows from premises."""
        # Simplified: check if conclusion contains premise keywords
        premise_terms = set(word.lower() for p in claim.premises for word in p.split())
        conclusion_terms = set(word.lower() for word in claim.conclusion.split())
        overlap = len(premise_terms & conclusion_terms)
        return overlap >= 2  # Minimal logical connection
    
    def _scan_fallacies(self, claim: Claim) -> List[FallacyType]:
        """Detect logical fallacies in claim structure."""
        detected = []
        full_text = f"{' '.join(claim.premises)} {claim.conclusion}".lower()
        
        for fallacy, indicators in self.fallacy_indicators.items():
            if any(indicator in full_text for indicator in indicators):
                detected.append(fallacy)
        
        return detected

# Usage example
lens = SkepticsLens()
test_claim = Claim(
    text="New framework improves performance",
    premises=[
        "Framework X reduces latency in benchmarks",
        "Lower latency correlates with better user experience"
    ],
    conclusion="Framework X improves overall performance",
    evidence=["Benchmark study 2024 showed 30% latency reduction"]
)

evaluation = lens.evaluate_claim(test_claim)
print(f"Status: {evaluation['verification_status']}")
print(f"Fallacies: {evaluation['detected_fallacies']}")
```

### Boundary Monitoring System

Inspect integration points between system components, detecting interface failures and data exchange anomalies at critical junctures.

```python
from typing import Protocol, Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

class InterfaceStatus(Enum):
    """Status of interface boundary health."""
    HEALTHY = "normal operation"
    DEGRADED = "partial functionality"
    FAILED = "connection lost"
    UNKNOWN = "not monitored"

@dataclass
class IntegrationPoint:
    """Represents boundary between system components."""
    component_a: str
    component_b: str
    protocol: str
    expected_latency_ms: float
    data_format: str

@dataclass
class BoundaryEvent:
    """Records event at integration boundary."""
    timestamp: datetime
    integration_point: str
    event_type: str  # "request", "response", "error", "timeout"
    payload_size_bytes: int
    latency_ms: float
    status_code: Optional[int] = None
    error_message: Optional[str] = None

class SeamWatcher:
    """Monitors integration boundaries for anomalies and failures."""
    
    def __init__(self):
        self.monitored_points: Dict[str, IntegrationPoint] = {}
        self.event_history: List[BoundaryEvent] = []
        self.alert_thresholds = {
            "latency_multiplier": 2.0,  # Alert if 2x expected latency
            "error_rate": 0.05,  # Alert if >5% error rate
            "timeout_count": 3   # Alert after 3 consecutive timeouts
        }
    
    def register_integration_point(self, point: IntegrationPoint) -> None:
        """Add integration boundary to monitoring scope."""
        point_id = f"{point.component_a}::{point.component_b}"
        self.monitored_points[point_id] = point
    
    def record_boundary_event(self, event: BoundaryEvent) -> None:
        """Log event occurring at integration boundary."""
        self.event_history.append(event)
        
        # Real-time anomaly detection
        if self._is_anomalous(event):
            self._raise_alert(event)
    
    def _is_anomalous(self, event: BoundaryEvent) -> bool:
        """Detect if boundary event represents anomaly."""
        if event.integration_point not in self.monitored_points:
            return False
        
        point = self.monitored_points[event.integration_point]
        
        # Check latency threshold
        if event.latency_ms > point.expected_latency_ms * self.alert_thresholds["latency_multiplier"]:
            return True
        
        # Check for error conditions
        if event.event_type == "error" or (event.status_code and event.status_code >= 400):
            return True
        
        # Check for consecutive timeouts
        recent_events = [e for e in self.event_history[-10:] 
                        if e.integration_point == event.integration_point]
        timeout_streak = sum(1 for e in recent_events[-3:] if e.event_type == "timeout")
        if timeout_streak >= self.alert_thresholds["timeout_count"]:
            return True
        
        return False
    
    def _raise_alert(self, event: BoundaryEvent) -> None:
        """Generate alert for boundary anomaly."""
        print(f"⚠️  SEAM ALERT [{event.timestamp}]")
        print(f"   Integration Point: {event.integration_point}")
        print(f"   Event: {event.event_type}")
        print(f"   Latency: {event.latency_ms}ms")
        if event.error_message:
            print(f"   Error: {event.error_message}")
    
    def generate_health_report(self) -> Dict[str, InterfaceStatus]:
        """Assess current health of all monitored integration points."""
        report = {}
        
        for point_id, point in self.monitored_points.items():
            recent_events = [e for e in self.event_history[-50:] 
                           if e.integration_point == point_id]
            
            if not recent_events:
                report[point_id] = InterfaceStatus.UNKNOWN
                continue
            
            error_rate = sum(1 for e in recent_events if e.event_type == "error") / len(recent_events)
            avg_latency = sum(e.latency_ms for e in recent_events) / len(recent_events)
            
            if error_rate > self.alert_thresholds["error_rate"]:
                report[point_id] = InterfaceStatus.FAILED
            elif avg_latency > point.expected_latency_ms * 1.5:
                report[point_id] = InterfaceStatus.DEGRADED
            else:
                report[point_id] = InterfaceStatus.HEALTHY
        
        return report

# Usage example
watcher = SeamWatcher()

# Register integration boundaries
api_gateway_point = IntegrationPoint(
    component_a="API_Gateway",
    component_b="Auth_Service",
    protocol="HTTP/REST",
    expected_latency_ms=50.0,
    data_format="JSON"
)
watcher.register_integration_point(api_gateway_point)

# Simulate boundary events
watcher.record_boundary_event(BoundaryEvent(
    timestamp=datetime.now(),
    integration_point="API_Gateway::Auth_Service",
    event_type="response",
    payload_size_bytes=2048,
    latency_ms=45.0,
    status_code=200
))

watcher.record_boundary_event(BoundaryEvent(
    timestamp=datetime.now(),
    integration_point="API_Gateway::Auth_Service",
    event_type="timeout",
    payload_size_bytes=0,
    latency_ms=5000.0,
    error_message="Connection timeout after 5000ms"
))

health = watcher.generate_health_report()
print(f"\nHealth Report: {health}")
```

## Concept Reference

| Concept | Technical | Plain | Importance |
|---------|-----------|-------|------------|
| Markdown Auditor | A specialized tool or process that validates, inspects, and verifies the structural integrity, syntax compliance, and formatting correctness of Markdown documents | A checker that looks over Markdown files to make sure they're written correctly and follow the rules for how Markdown should be formatted. | 0.95 |
| Skeptic's Lens | An analytical framework or methodological approach that applies critical evaluation, questioning assumptions, and systematic doubt to examine claims | A way of looking at information with a questioning mindset, where you don't automatically believe things but instead carefully examine whether they make sense | 0.90 |
| Seam-Watcher | A monitoring mechanism or inspection system that identifies boundary conditions, integration points, or junctures where different components, systems, or modules connect | A tool that watches the places where different parts of something connect or join together, looking for problems or gaps where things meet | 0.88 |
| Auditing | A systematic, independent examination and evaluation of records, processes, or systems to assess compliance with standards, identify discrepancies, and verify accuracy | A careful review process where someone checks work or records to make sure everything is correct and follows the rules that should be followed | 0.85 |
| Markdown Syntax | A lightweight markup language specification utilizing plain text formatting conventions with special characters and symbols to denote structural elements | A simple way of writing documents using regular text with special symbols (like asterisks and hashtags) to show things like headings, bold text, and lists | 0.82 |
| Critical Analysis | A cognitive process involving systematic examination, evaluation, and interpretation of information through logical reasoning, evidence assessment, and identification of assumptions | Thinking carefully and deeply about something by breaking it down into parts, questioning whether it makes sense, and looking for any hidden problems | 0.80 |
| Interface Boundary | The demarcation point or transition zone where two distinct systems, components, modules, or domains interact, exchange data, or transfer control | The edge or border where two different things meet and need to work together, like where two puzzle pieces connect | 0.78 |
| Validation | The process of verifying that data, input, output, or system behavior conforms to predefined specifications, constraints, and requirements through systematic testing | Checking that something meets the requirements and works the way it's supposed to work, confirming it's correct and acceptable | 0.76 |
| Skepticism | An epistemological stance characterized by systematic doubt, critical inquiry, and provisional acceptance of claims pending sufficient evidence | A mindset of being cautious about believing things without good reasons, preferring to see proof before accepting something as true | 0.75 |
| Structural Integrity | The property of a system, document, or construct maintaining its intended organization, hierarchical relationships, and compositional coherence | How well something holds together and maintains its proper structure and organization without falling apart or becoming disorganized | 0.73 |

## Glossary

| Term | Definition | Concept IDs |
|------|------------|-------------|
| Markdown Auditor | A specialized validation tool that systematically inspects Markdown documents for syntax compliance, structural integrity, and formatting correctness | [1, 5, 10, 13] |
| Skeptic's Lens | An analytical framework applying critical evaluation and systematic doubt to examine claims for logical consistency and empirical validity | [2, 6, 9, 16] |
| Seam-Watcher | A monitoring mechanism that identifies and inspects boundary conditions and integration points where components connect, detecting potential failures | [3, 7, 11, 14] |
| Document Validation | The process of verifying that written materials conform to formatting standards, structural requirements, and content specifications | [1, 8, 13, 17] |
| Critical Inquiry | A cognitive approach involving systematic examination, questioning of assumptions, and evidence assessment to evaluate information validity | [2, 6, 9] |
| Interface Monitoring | Continuous observation of boundary points where systems or components interact to detect anomalies and ensure proper integration | [3, 7, 11, 14] |
| Syntax Compliance | Adherence to the formal rules and conventions of a markup language specification, ensuring correct use of formatting symbols and structure | [1, 5, 8, 17] |
| Quality Assurance Framework | A systematic methodology encompassing validation, monitoring, and critical analysis to ensure outputs meet specified requirements and standards | [4, 12, 16, 17] |
| Boundary Detection | The identification and examination of demarcation points where distinct systems, components, or domains interact and exchange information | [3, 7, 14, 15] |
| Empirical Verification | The process of confirming claims or system behavior through evidence-based testing and observation rather than assumption or assertion | [2, 8, 9, 17] |
| Structural Assessment | Evaluation of how well a document or system maintains its intended organization, hierarchical relationships, and compositional coherence | [1, 10, 13, 15] |
| Integration Assurance | The practice of ensuring that separate components or modules connect properly and exchange information correctly at designated interface points | [3, 7, 12, 14] |
| Systematic Auditing | An independent examination process that assesses compliance with standards, identifies discrepancies, and verifies accuracy across systems or documents | [4, 8, 12, 15] |

## Edge Cases & Warnings

- ⚠️ **Markdown Dialect Variations**: Different Markdown parsers (CommonMark, GitHub Flavored, MultiMarkdown) have incompatible syntax extensions; auditor must specify target dialect
- ⚠️ **False Positive Skepticism**: Excessive critical evaluation can reject valid claims lacking immediate empirical evidence; balance systematic doubt with pragmatic acceptance thresholds
- ⚠️ **Monitoring Overhead**: Seam-watching on high-frequency integration points (>10k events/sec) may introduce performance degradation; implement sampling strategies
- ⚠️ **Cascading Boundary Failures**: Single integration point failure can propagate across dependent seams; monitor upstream/downstream relationships
- ⚠️ **Context-Dependent Validation**: Some Markdown constructs (embedded HTML, custom extensions) require context-aware validation beyond syntax checking
- ⚠️ **Circular Dependency Detection**: Integration monitoring must detect circular dependencies between components to prevent infinite loop conditions
- ⚠️ **Evidence Staleness**: Claims verified against historical evidence may become invalidated by new information; implement re-verification cycles
- ⚠️ **Asynchronous Boundary Events**: Event ordering at integration points may not reflect causal relationships in distributed systems; use vector clocks or logical timestamps

## Quick Reference

```python
# Triadic QA workflow combining all three validation approaches

from typing import List, Dict

def triadic_quality_assurance(
    document_path: str,
    claims_to_verify: List[Claim],
    integration_points: List[IntegrationPoint]
) -> Dict[str, any]:
    """Execute complete three-phase validation workflow."""
    
    # Phase 1: Structural Validation (Markdown Auditor)
    auditor = MarkdownAuditor()
    with open(document_path, 'r') as f:
        doc_lines = f.readlines()
    
    structural_issues = (
        auditor.audit_heading_hierarchy(doc_lines) +
        auditor.audit_code_blocks(doc_lines)
    )
    
    # Phase 2: Critical Evaluation (Skeptic's Lens)
    lens = SkepticsLens()
    claim_evaluations = [lens.evaluate_claim(claim) for claim in claims_to_verify]
    
    # Phase 3: Boundary Monitoring (Seam-Watcher)
    watcher = SeamWatcher()
    for point in integration_points:
        watcher.register_integration_point(point)
    
    boundary_health = watcher.generate_health_report()
    
    # Unified report
    return {
        "structural_validation": {
            "issues_found": len(structural_issues),
            "details": structural_issues
        },
        "critical_evaluation": {
            "verified_claims": sum(1 for e in claim_evaluations if e["verification_status"] == "VERIFIED"),
            "rejected_claims": sum(1 for e in claim_evaluations if e["verification_status"] == "REJECTED"),
            "details": claim_evaluations
        },
        "boundary_monitoring": {
            "healthy_seams": sum(1 for s in boundary_health.values() if s == InterfaceStatus.HEALTHY),
            "failed_seams": sum(1 for s in boundary_health.values() if s == InterfaceStatus.FAILED),
            "details": boundary_health
        },
        "overall_status": "PASS" if (
            len(structural_issues) == 0 and
            all(e["verification_status"] != "REJECTED" for e in claim_evaluations) and
            all(s != InterfaceStatus.FAILED for s in boundary_health.values())
        ) else "FAIL"
    }

# Minimal usage
result = triadic_quality_assurance(
    document_path="README.md",
    claims_to_verify=[test_claim],
    integration_points=[api_gateway_point]
)
print(f"QA Status: {result['overall_status']}")
```

---
_Generated by Philosopher's Stone v4 — EchoSeed_
Philosopher's Stone v4 × Skill Forge × EchoSeed
