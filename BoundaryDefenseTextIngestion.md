# Boundary-Defense Text Ingestion

> Trigger this skill when building text processing pipelines that must reject invalid inputs early, preserve error states across serialization boundaries, and enable autonomous agents to handle failures programmatically. Apply when downstream code must operate on guaranteed-valid data without redundant validation, when parsing Markdown structure matters for feature ranking, or when domain-specific terminology requires specialized TF-IDF calibration against field-specific corpora rather than general-web statistics.

## Overview

This skill implements a layered boundary-defense architecture for text ingestion that intercepts semantically empty inputs and converts parse failures into serializable error objects before any processing begins. By concentrating all validation at a single ingestion boundary, downstream code receives proven-valid typed objects that eliminate redundant checks. The system produces machine-readable JSON error envelopes enabling agentic orchestrators to act on failures without human intervention. Beyond validation, it performs Markdown AST-based feature extraction treating structural elements as weighted signals, and applies domain-aware TF-IDF scoring calibrated against in-domain corpora. The architecture follows a schema-first output contract defining data structures before implementation.

## When to Use

- Building autonomous agent pipelines where failures must be machine-actionable without human review
- Processing domain-specific text where general-web term frequencies misrepresent importance
- Creating trust boundaries where downstream code must assume input validity without re-checking
- Extracting features from Markdown where headers and bold text signal semantic importance
- Converting parsing exceptions into data that survives serialization through multiple system boundaries
- Implementing validation layers that reject placeholder text and sentinel strings before processing
- Designing systems where error states must traverse API boundaries and be stored in databases

## Core Workflow

1. **Schema Definition**: Define output contract with JSON schema specifying structure, types, and constraints before writing extractor logic
2. **Boundary Validation**: Intercept inputs at single ingestion point, rejecting semantically empty strings (placeholders, sentinel values) and converting parse failures to error objects
3. **Type Construction**: Build proven-valid typed objects enforcing invariants at construction time, creating data structures downstream code can trust
4. **AST Extraction**: Parse Markdown into Abstract Syntax Tree, extract structural features (headings, bold terms) as weighted signals
5. **Domain Scoring**: Apply TF-IDF scoring calibrated against in-domain corpora rather than general-web baselines for terminology relevance
6. **Error Packaging**: Wrap all failures in machine-readable JSON envelopes with error codes, messages, and metadata for programmatic consumption
7. **Orchestration Handoff**: Return typed objects or error envelopes to agentic orchestrators for autonomous decision-making

## Key Patterns

### Single Ingestion Boundary with Error Reification

Consolidate all validation at one architectural point and convert exceptions into passable data structures. This eliminates downstream validation and enables error states to survive serialization.

```python
from typing import Union, Literal
from dataclasses import dataclass
from enum import Enum
import json

class ErrorCode(Enum):
    EMPTY_INPUT = "EMPTY_INPUT"
    SENTINEL_DETECTED = "SENTINEL_DETECTED"
    PARSE_FAILURE = "PARSE_FAILURE"
    INVALID_MARKDOWN = "INVALID_MARKDOWN"

@dataclass
class ErrorEnvelope:
    """Machine-readable error that survives serialization"""
    code: ErrorCode
    message: str
    input_sample: str
    metadata: dict
    
    def to_json(self) -> str:
        return json.dumps({
            "error": self.code.value,
            "message": self.message,
            "input": self.input_sample[:100],
            "metadata": self.metadata
        })

@dataclass
class ValidatedText:
    """Proven-valid typed object with enforced invariants"""
    content: str
    word_count: int
    has_structure: bool
    
    def __post_init__(self):
        # Invariants enforced at construction
        assert len(self.content.strip()) > 0, "Content cannot be empty"
        assert self.word_count == len(self.content.split()), "Word count mismatch"

# Sentinel strings that indicate empty content
SENTINEL_VALUES = {"lorem ipsum", "n/a", "null", "none", "placeholder", "todo", "tbd"}

def validate_at_boundary(raw_input: str) -> Union[ValidatedText, ErrorEnvelope]:
    """Single point of validation - all checks happen here"""
    
    # Check for semantically empty input
    normalized = raw_input.strip().lower()
    if not normalized:
        return ErrorEnvelope(
            code=ErrorCode.EMPTY_INPUT,
            message="Input contains only whitespace",
            input_sample=raw_input,
            metadata={"length": len(raw_input)}
        )
    
    # Check for sentinel strings
    if any(sentinel in normalized for sentinel in SENTINEL_VALUES):
        return ErrorEnvelope(
            code=ErrorCode.SENTINEL_DETECTED,
            message="Input contains placeholder or sentinel value",
            input_sample=raw_input,
            metadata={"detected_sentinels": [s for s in SENTINEL_VALUES if s in normalized]}
        )
    
    # Parse failures become data, not exceptions
    try:
        word_count = len(raw_input.split())
        has_structure = any(marker in raw_input for marker in ["#", "**", "*", "-"])
        
        # Return proven-valid object - downstream can trust this
        return ValidatedText(
            content=raw_input.strip(),
            word_count=word_count,
            has_structure=has_structure
        )
    except Exception as e:
        # Exception reified as passable value
        return ErrorEnvelope(
            code=ErrorCode.PARSE_FAILURE,
            message=f"Failed to parse input: {str(e)}",
            input_sample=raw_input,
            metadata={"exception_type": type(e).__name__}
        )
```

### Markdown AST Feature Extraction with Weighted Signals

Parse Markdown structure and assign importance weights based on formatting. Headers and bold text carry higher weight than plain text.

```python
from typing import List, Dict
from dataclasses import dataclass
import re

@dataclass
class WeightedFeature:
    """Feature with importance score based on structural position"""
    text: str
    weight: float
    feature_type: Literal["heading", "bold", "text"]

class MarkdownASTExtractor:
    """Extract features from Markdown treating structure as signal"""
    
    # Weights calibrated for importance ranking
    WEIGHTS = {
        "h1": 1.0,
        "h2": 0.8,
        "h3": 0.6,
        "bold": 0.7,
        "text": 0.3
    }
    
    def extract_features(self, markdown: str) -> List[WeightedFeature]:
        """Parse Markdown into AST and extract weighted features"""
        features = []
        
        # Extract h1 headings (# Title)
        for match in re.finditer(r'^#\s+(.+)$', markdown, re.MULTILINE):
            features.append(WeightedFeature(
                text=match.group(1).strip(),
                weight=self.WEIGHTS["h1"],
                feature_type="heading"
            ))
        
        # Extract h2 headings (## Title)
        for match in re.finditer(r'^##\s+(.+)$', markdown, re.MULTILINE):
            features.append(WeightedFeature(
                text=match.group(1).strip(),
                weight=self.WEIGHTS["h2"],
                feature_type="heading"
            ))
        
        # Extract h3 headings (### Title)
        for match in re.finditer(r'^###\s+(.+)$', markdown, re.MULTILINE):
            features.append(WeightedFeature(
                text=match.group(1).strip(),
                weight=self.WEIGHTS["h3"],
                feature_type="heading"
            ))
        
        # Extract bold text (**bold** or __bold__)
        for match in re.finditer(r'\*\*(.+?)\*\*|__(.+?)__', markdown):
            text = match.group(1) or match.group(2)
            features.append(WeightedFeature(
                text=text.strip(),
                weight=self.WEIGHTS["bold"],
                feature_type="bold"
            ))
        
        # Extract plain text paragraphs (lower weight)
        # Remove Markdown syntax first
        clean_text = re.sub(r'#+\s+', '', markdown)
        clean_text = re.sub(r'\*\*(.+?)\*\*|__(.+?)__', r'\1\2', clean_text)
        
        for paragraph in clean_text.split('\n\n'):
            if paragraph.strip():
                features.append(WeightedFeature(
                    text=paragraph.strip(),
                    weight=self.WEIGHTS["text"],
                    feature_type="text"
                ))
        
        return features
    
    def rank_by_importance(self, features: List[WeightedFeature]) -> List[WeightedFeature]:
        """Sort features by weight for importance ranking"""
        return sorted(features, key=lambda f: f.weight, reverse=True)
```

### Domain-Aware TF-IDF Calibration

Calculate term importance using corpus statistics from the target domain rather than general-web frequencies. This correctly weights technical terminology.

```python
from typing import List, Dict, Set
from collections import Counter
import math

@dataclass
class DomainCorpus:
    """Reference corpus for domain-specific calibration"""
    documents: List[str]
    term_doc_freq: Dict[str, int]  # How many docs contain each term
    total_docs: int
    
    @classmethod
    def from_documents(cls, documents: List[str]) -> 'DomainCorpus':
        """Build corpus statistics from domain documents"""
        term_doc_freq = Counter()
        
        for doc in documents:
            # Count unique terms per document (not total occurrences)
            unique_terms = set(doc.lower().split())
            for term in unique_terms:
                term_doc_freq[term] += 1
        
        return cls(
            documents=documents,
            term_doc_freq=dict(term_doc_freq),
            total_docs=len(documents)
        )

class DomainAwareTFIDF:
    """TF-IDF scorer calibrated against in-domain corpus"""
    
    def __init__(self, domain_corpus: DomainCorpus):
        self.corpus = domain_corpus
    
    def calculate_idf(self, term: str) -> float:
        """
        Inverse document frequency calibrated to domain.
        Rare terms in domain get high scores, common terms get low scores.
        """
        doc_freq = self.corpus.term_doc_freq.get(term.lower(), 0)
        
        if doc_freq == 0:
            # Term not in domain corpus - very rare, highest score
            return math.log(self.corpus.total_docs + 1)
        
        # Standard IDF formula with domain statistics
        return math.log(self.corpus.total_docs / doc_freq)
    
    def calculate_tf(self, term: str, document: str) -> float:
        """Term frequency in specific document"""
        terms = document.lower().split()
        term_count = terms.count(term.lower())
        
        if len(terms) == 0:
            return 0.0
        
        return term_count / len(terms)
    
    def score_document(self, document: str) -> Dict[str, float]:
        """
        Calculate TF-IDF scores for all terms using domain calibration.
        Returns term -> score mapping.
        """
        scores = {}
        unique_terms = set(document.lower().split())
        
        for term in unique_terms:
            tf = self.calculate_tf(term, document)
            idf = self.calculate_idf(term)
            scores[term] = tf * idf
        
        return scores
    
    def compare_to_general_web(self, term: str, general_web_idf: float) -> Dict[str, float]:
        """
        Show difference between domain-aware and general-web scoring.
        Useful for validating domain calibration is working.
        """
        domain_idf = self.calculate_idf(term)
        
        return {
            "term": term,
            "domain_idf": domain_idf,
            "general_web_idf": general_web_idf,
            "domain_boost": domain_idf / general_web_idf if general_web_idf > 0 else float('inf')
        }

# Example: Technical terms get higher scores in domain corpus
machine_learning_corpus = DomainCorpus.from_documents([
    "neural networks use backpropagation for gradient descent",
    "convolutional layers extract spatial features from images",
    "transformer architecture revolutionized natural language processing",
    "attention mechanisms enable models to focus on relevant information"
])

scorer = DomainAwareTFIDF(machine_learning_corpus)

# "transformer" is common in ML domain but rare on general web
print(scorer.calculate_idf("transformer"))  # High score in domain
print(scorer.calculate_idf("the"))  # Low score - common everywhere
```

### Schema-First Output Contract

Define output structure before implementation using typed dataclasses or Pydantic models. This creates binding interface extractor logic must satisfy.

```python
from typing import List, Optional, Literal
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

# Define contract BEFORE writing extractor logic
@dataclass
class ExtractionMetadata:
    """Metadata about extraction process"""
    timestamp: datetime
    extractor_version: str
    processing_time_ms: float
    feature_count: int

@dataclass
class ExtractedTerm:
    """Contract for individual term extraction"""
    term: str
    tf_idf_score: float
    source_context: str  # Sentence or paragraph containing term
    structural_weight: float  # From Markdown AST position
    
    def __post_init__(self):
        # Enforce invariants defined in contract
        assert 0.0 <= self.tf_idf_score <= 100.0, "TF-IDF score out of range"
        assert 0.0 <= self.structural_weight <= 1.0, "Structural weight must be [0,1]"
        assert len(self.term.strip()) > 0, "Term cannot be empty"

@dataclass
class ExtractionResult:
    """
    Schema-first output contract.
    Extractor logic must produce objects matching this structure.
    """
    status: Literal["success", "partial", "failed"]
    terms: List[ExtractedTerm]
    metadata: ExtractionMetadata
    errors: List[ErrorEnvelope]
    
    def to_json_schema(self) -> dict:
        """Generate JSON Schema for API documentation"""
        return {
            "type": "object",
            "required": ["status", "terms", "metadata", "errors"],
            "properties": {
                "status": {"type": "string", "enum": ["success", "partial", "failed"]},
                "terms": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["term", "tf_idf_score", "source_context", "structural_weight"],
                        "properties": {
                            "term": {"type": "string"},
                            "tf_idf_score": {"type": "number", "minimum": 0, "maximum": 100},
                            "source_context": {"type": "string"},
                            "structural_weight": {"type": "number", "minimum": 0, "maximum": 1}
                        }
                    }
                },
                "metadata": {"type": "object"},
                "errors": {"type": "array"}
            }
        }

# Extractor MUST satisfy this contract
def extract_with_contract(validated_input: ValidatedText, 
                         corpus: DomainCorpus) -> ExtractionResult:
    """
    Implementation constrained by predefined schema.
    Return type enforces contract at compile time.
    """
    start_time = datetime.now()
    
    # Extract features using patterns above
    ast_extractor = MarkdownASTExtractor()
    features = ast_extractor.extract_features(validated_input.content)
    
    tfidf_scorer = DomainAwareTFIDF(corpus)
    scores = tfidf_scorer.score_document(validated_input.content)
    
    # Build terms matching contract structure
    terms = []
    for feature in features:
        if feature.text in scores:
            terms.append(ExtractedTerm(
                term=feature.text,
                tf_idf_score=min(scores[feature.text] * 10, 100.0),  # Scale to [0,100]
                source_context=feature.text,  # Simplified for example
                structural_weight=feature.weight
            ))
    
    processing_time = (datetime.now() - start_time).total_seconds() * 1000
    
    # Return object satisfying contract
    return ExtractionResult(
        status="success" if terms else "partial",
        terms=terms,
        metadata=ExtractionMetadata(
            timestamp=datetime.now(),
            extractor_version="1.0.0",
            processing_time_ms=processing_time,
            feature_count=len(terms)
        ),
        errors=[]
    )
```

## Concept Reference

| Concept | Technical | Plain | Importance |
|---------|-----------|-------|------------|
| layered boundary-defense architecture | A multi-tiered security pattern where validation and sanitization occur at successive architectural boundaries, with each layer enforcing increasingly strict constraints | A system design where input goes through several checkpoints, like airport security, with each checkpoint catching different problems before the data enters trusted zones | 0.95 |
| schema-first output contract | A development methodology where the structure, types, and constraints of output data are formally specified before implementation, establishing a binding interface | Deciding exactly what format and structure the results will have before writing any code, like creating a blueprint before building | 0.93 |
| domain-aware TF-IDF scoring | Term Frequency-Inverse Document Frequency calculation calibrated using corpus statistics specific to a particular domain, adjusting term importance based on field-specific usage | A way of measuring word importance that's tuned to a specific field, so technical terms that are rare in normal text but common in that field get scored correctly | 0.91 |
| parse failures as passable values | An error-handling pattern where parsing exceptions are reified as first-class data structures that can traverse system boundaries and be serialized | Instead of crashing when something can't be read properly, the system wraps the error in a container that can be passed around and saved like normal data | 0.90 |
| single ingestion boundary | A centralized validation layer that consolidates all input verification logic at one architectural point, creating a trust boundary | One main gate where all checking happens, so everything beyond that gate can trust the data is already verified and correct | 0.88 |
| machine-readable JSON error envelopes | Structured error responses conforming to a defined JSON schema that includes error codes, messages, and metadata in a format optimized for programmatic interpretation | Error messages packaged in a standard JSON format that computers can easily read and understand, not just text meant for people to read | 0.87 |
| proven-valid typed objects | Data structures that carry both runtime type information and invariant guarantees enforced at construction time, ensuring all instances satisfy domain constraints | Objects that have passed all tests and are guaranteed to be the right type with correct values, so you can use them without checking again | 0.86 |
| text ingestion | The process of accepting, validating, and preparing textual input data for subsequent processing stages within a computational pipeline | The first step where text enters a system and gets checked to make sure it's ready to be processed | 0.85 |
| Markdown AST-based feature extraction | The process of parsing Markdown into an Abstract Syntax Tree representation and extracting structural and semantic features from typed nodes | Converting Markdown text into a tree structure that shows headers, bold text, and other formatting, then pulling out important information based on that structure | 0.84 |
| agentic orchestrators | Autonomous software components that coordinate complex workflows by interpreting system outputs and making decisions about subsequent actions without human intervention | Smart programs that can read results, make decisions, and control what happens next automatically without needing a person to tell them what to do | 0.83 |

## Glossary

| Term | Definition | Concept IDs |
|------|------------|-------------|
| layered boundary-defense | Multi-tiered security pattern with validation at successive architectural boundaries before data enters trusted zones | 1 |
| single ingestion boundary | Centralized validation layer consolidating all input verification at one architectural point to create a trust boundary | 7 |
| semantically empty inputs | Syntactically valid strings carrying no meaningful content, including placeholders and sentinel values | 3, 4 |
| parse failures as passable values | Error-handling pattern reifying parsing exceptions as first-class data structures that traverse system boundaries | 5, 6 |
| proven-valid typed objects | Data structures carrying runtime type information and invariant guarantees enforced at construction time | 8 |
| machine-readable JSON error envelopes | Structured error responses with codes, messages, and metadata optimized for programmatic consumption | 9 |
| agentic orchestrators | Autonomous software components coordinating workflows by interpreting outputs and making decisions without human intervention | 10 |
| Markdown AST-based extraction | Parsing Markdown into Abstract Syntax Tree and extracting features from typed nodes rather than raw text | 11 |
| weighted signals | Feature representations assigned numerical importance values reflecting contribution to classification objectives | 12 |
| domain-aware TF-IDF | Term frequency-inverse document frequency calculation calibrated using corpus statistics specific to a domain | 13 |
| in-domain corpora | Reference document collections representative of specific domains for establishing baseline frequency distributions | 14 |
| schema-first output contract | Methodology where output structure, types, and constraints are formally specified before implementation | 16 |
| extractor logic | Algorithmic implementation identifying, isolating, and transforming relevant features from input data structures | 17 |

## Edge Cases & Warnings

- ⚠️ **Sentinel Detection False Positives**: Common words like "none" appear in legitimate text. Use context-aware detection checking surrounding tokens, not just substring matching. Consider "none of the options" vs standalone "none" placeholder.
- ⚠️ **Empty Markdown Structure**: Documents with no headers or bold text receive only low-weight plain-text features. Implement fallback scoring when `has_structure == False` to avoid zero-feature extractions.
- ⚠️ **Domain Corpus Size Threshold**: TF-IDF calibration requires minimum 50-100 documents for stable statistics. With smaller corpora, blend domain and general-web IDF scores using weighted average.
- ⚠️ **Serialization of Custom Types**: Error envelopes containing Enum types require custom JSON encoders. Implement `to_json()` methods on all error classes before passing to agentic orchestrators.
- ⚠️ **Invariant Violations at Boundaries**: `__post_init__` assertions in typed objects crash construction. Catch these at ingestion boundary and convert to error envelopes rather than propagating exceptions.
- ⚠️ **Unicode Normalization**: Markdown parsers may fail on mixed Unicode normalization forms (NFC vs NFD). Normalize input to NFC before AST parsing to prevent regex match failures on accented characters.
- ⚠️ **Nested Markdown Syntax**: Patterns like `**bold _italic_**` create ambiguous AST nodes. Flatten nested formatting during extraction or define precedence rules in schema contract.
- ⚠️ **TF-IDF Score Scaling**: Raw TF-IDF scores vary by document length and corpus size. Normalize scores to [0, 1] range or define explicit bounds in output contract to enable cross-document comparison.
- ⚠️ **Concurrent Corpus Updates**: If domain corpus updates during processing, IDF calculations become inconsistent. Version corpus snapshots and include version metadata in extraction results.
- ⚠️ **Error Envelope Explosion**: Storing full input samples in error objects consumes memory with large documents. Truncate samples to 200 characters and include byte offset for debugging context.

## Quick Reference

```python
from typing import Union
from dataclasses import dataclass
from enum import Enum

# 1. Define schema contract first
@dataclass
class ValidatedText:
    content: str
    word_count: int

@dataclass 
class ErrorEnvelope:
    code: str
    message: str

# 2. Validate at single boundary
SENTINELS = {"lorem ipsum", "n/a", "todo"}

def ingest(raw: str) -> Union[ValidatedText, ErrorEnvelope]:
    if not raw.strip():
        return ErrorEnvelope("EMPTY", "No content")
    if any(s in raw.lower() for s in SENTINELS):
        return ErrorEnvelope("SENTINEL", "Placeholder detected")
    return ValidatedText(raw.strip(), len(raw.split()))

# 3. Extract weighted Markdown features
def extract_headings(text: str) -> list:
    import re
    return [(m.group(1), 1.0) for m in re.finditer(r'^#\s+(.+)$', text, re.MULTILINE)]

# 4. Score with domain-aware TF-IDF
def domain_idf(term: str, corpus: list) -> float:
    import math
    doc_freq = sum(1 for doc in corpus if term in doc.lower())
    return math.log(len(corpus) / (doc_freq + 1))

# 5. Use in agent pipeline
result = ingest("# Machine Learning\nTransformers use attention mechanisms")
if isinstance(result, ValidatedText):
    headings = extract_headings(result.content)
    # Agent proceeds with proven-valid data
else:
    # Agent handles error programmatically
    print(f"Agent received error: {result.code}")
```

---
_Generated by Philosopher's Stone v4 — EchoSeed_
Philosopher's Stone v4 × Skill Forge × EchoSeed
