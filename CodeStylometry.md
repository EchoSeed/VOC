# Code Stylometry Analysis

> Trigger this skill when analyzing code authorship, investigating source code provenance, attributing malicious software, verifying code authenticity in legal contexts, detecting plagiarism in programming assignments, or identifying insider threats through coding patterns. Use when you need to determine "who wrote this code" based on stylistic fingerprints rather than metadata. Also applicable when studying programming style evolution, building author verification systems, or implementing privacy-preserving coding practices to resist de-anonymization.

## Overview

Code stylometry applies computational analysis to programmer identification by extracting and quantifying distinctive coding habits—naming conventions, formatting patterns, structural preferences, and complexity tendencies. Machine learning classifiers trained on labeled code samples recognize these stylistic signatures with accuracy comparable to biometric identification. The technique serves forensic cybersecurity purposes (malware attribution, insider threat detection), legal verification (intellectual property disputes, code provenance), and academic integrity enforcement. However, deliberate obfuscation can disguise style, and natural evolution complicates long-term attribution.

## When to Use

- Investigating authorship of anonymous or pseudonymous code contributions
- Attributing malware, exploits, or malicious scripts to threat actors
- Verifying claimed authorship in intellectual property disputes or academic contexts
- Detecting code plagiarism or unauthorized copying with style transformation
- Building privacy-preserving systems that resist de-anonymization attacks
- Analyzing insider threat risks through behavioral consistency monitoring
- Studying cross-language style persistence in polyglot developers
- Evaluating temporal style drift for adaptive attribution systems

## Core Workflow

1. **Feature Extraction Pipeline**: Parse source code into Abstract Syntax Trees (AST), extract lexical tokens, compute n-gram frequencies, measure complexity metrics (cyclomatic, Halstead), quantify layout patterns (indentation depth, line length), and catalog identifier naming conventions (case style, length distribution, semantic content).

2. **Feature Engineering & Selection**: Normalize extracted features across language syntax variations, apply dimensionality reduction (PCA, feature importance ranking), weight features by discriminative power, handle missing values for incomplete code fragments, and create language-agnostic representations for cross-platform analysis.

3. **Model Training & Classification**: Split labeled corpus into training/validation/test sets (typically 70/15/15), train supervised classifiers (Random Forest, SVM, Neural Networks), perform cross-validation with author-stratified folds, tune hyperparameters through grid search, evaluate with precision/recall/F1 metrics, and validate against adversarial obfuscation attempts.

4. **Attribution & Confidence Scoring**: Apply trained model to unknown code samples, generate probability distributions across candidate authors, compute confidence intervals accounting for training set size and feature coverage, identify ambiguous cases requiring manual review, and provide explainable attribution through feature contribution analysis.

5. **Temporal Validation & Adaptation**: Test attribution accuracy against code samples from different time periods, detect style drift through longitudinal consistency metrics, retrain models periodically with recent samples, flag authors exhibiting significant evolution, and maintain versioned author profiles capturing style trajectories.

## Key Patterns

### AST-Based Structural Fingerprinting

Extract hierarchical syntactic patterns that persist across superficial formatting changes. AST nodes reveal control flow preferences, nesting tendencies, and algorithmic structure choices independent of variable names or whitespace.

```python
import ast
from collections import Counter
from typing import Dict, List

def extract_ast_features(source_code: str) -> Dict[str, float]:
    """
    Extract structural features from Python AST for authorship fingerprinting.
    
    Returns normalized feature vector capturing syntactic preferences.
    """
    tree = ast.parse(source_code)
    
    # Count node types (reveals control flow preferences)
    node_counts = Counter()
    for node in ast.walk(tree):
        node_counts[type(node).__name__] += 1
    
    # Calculate depth metrics (nesting tendency)
    def get_max_depth(node: ast.AST, current_depth: int = 0) -> int:
        max_child_depth = current_depth
        for child in ast.iter_child_nodes(node):
            child_depth = get_max_depth(child, current_depth + 1)
            max_child_depth = max(max_child_depth, child_depth)
        return max_child_depth
    
    max_depth = get_max_depth(tree)
    
    # Compute complexity metrics
    total_nodes = sum(node_counts.values())
    
    return {
        'avg_nesting_depth': max_depth / len(list(ast.walk(tree))),
        'if_ratio': node_counts['If'] / total_nodes,
        'for_ratio': node_counts['For'] / total_nodes,
        'while_ratio': node_counts['While'] / total_nodes,
        'try_ratio': node_counts['Try'] / total_nodes,
        'comprehension_ratio': (node_counts['ListComp'] + 
                               node_counts['DictComp']) / total_nodes,
        'lambda_ratio': node_counts['Lambda'] / total_nodes,
    }

# Example usage
code_sample = """
def process_data(items):
    result = [x * 2 for x in items if x > 0]
    try:
        return sum(result)
    except TypeError:
        return 0
"""

features = extract_ast_features(code_sample)
print(f"Structural fingerprint: {features}")
```

### N-Gram Stylistic Profiling

Capture local coding idioms through contiguous token sequences. N-grams reveal characteristic patterns like `if x is not None:` vs `if x != None:` preferences that distinguish authors.

```python
from typing import List, Dict, Tuple
from collections import Counter
import re

def extract_ngram_profile(source_code: str, n: int = 3) -> Dict[Tuple[str, ...], int]:
    """
    Generate n-gram frequency distribution for stylometric analysis.
    
    Args:
        source_code: Raw source text
        n: N-gram size (trigrams by default)
    
    Returns:
        Frequency distribution of token sequences
    """
    # Tokenize while preserving operators and keywords
    tokens = re.findall(r'\b\w+\b|[^\w\s]', source_code)
    
    # Generate n-grams
    ngrams = []
    for i in range(len(tokens) - n + 1):
        ngram = tuple(tokens[i:i+n])
        ngrams.append(ngram)
    
    return Counter(ngrams)

def compare_ngram_profiles(profile1: Dict, profile2: Dict) -> float:
    """
    Compute similarity between two n-gram profiles using cosine similarity.
    
    Returns value in [0, 1] where 1 indicates identical style.
    """
    all_ngrams = set(profile1.keys()) | set(profile2.keys())
    
    dot_product = sum(profile1.get(ng, 0) * profile2.get(ng, 0) 
                     for ng in all_ngrams)
    
    magnitude1 = sum(count**2 for count in profile1.values()) ** 0.5
    magnitude2 = sum(count**2 for count in profile2.values()) ** 0.5
    
    if magnitude1 * magnitude2 == 0:
        return 0.0
    
    return dot_product / (magnitude1 * magnitude2)

# Example: Comparing two code samples
sample_a = "if x is not None: return x else: return 0"
sample_b = "if x != None: return x else: return 0"

profile_a = extract_ngram_profile(sample_a, n=3)
profile_b = extract_ngram_profile(sample_b, n=3)

similarity = compare_ngram_profiles(profile_a, profile_b)
print(f"Style similarity: {similarity:.3f}")
print(f"Sample A trigrams: {list(profile_a.most_common(3))}")
print(f"Sample B trigrams: {list(profile_b.most_common(3))}")
```

### Layout Fingerprint Extraction

Quantify whitespace usage, indentation consistency, line length preferences, and bracket placement styles that form unconscious formatting signatures.

```python
from dataclasses import dataclass
from typing import List
import statistics

@dataclass
class LayoutFingerprint:
    """Quantified layout and formatting preferences."""
    avg_line_length: float
    line_length_stddev: float
    indentation_unit: int  # spaces per indent level
    blank_line_ratio: float
    trailing_whitespace_ratio: float
    max_line_length: int
    bracket_newline_ratio: float  # opening brackets followed by newline

def extract_layout_fingerprint(source_code: str) -> LayoutFingerprint:
    """
    Measure formatting characteristics that reveal author identity.
    
    Layout patterns are highly distinctive and difficult to fake consistently.
    """
    lines = source_code.split('\n')
    non_empty_lines = [line for line in lines if line.strip()]
    
    # Line length statistics
    line_lengths = [len(line) for line in non_empty_lines]
    avg_length = statistics.mean(line_lengths) if line_lengths else 0
    stddev = statistics.stdev(line_lengths) if len(line_lengths) > 1 else 0
    
    # Detect indentation unit (most common indent increment)
    indents = []
    for line in non_empty_lines:
        leading_spaces = len(line) - len(line.lstrip(' '))
        if leading_spaces > 0:
            indents.append(leading_spaces)
    
    # Find GCD of indentation levels (likely indent unit)
    from math import gcd
    indent_unit = indents[0] if indents else 4
    for indent in indents[1:]:
        indent_unit = gcd(indent_unit, indent)
    
    # Blank line ratio
    blank_lines = len(lines) - len(non_empty_lines)
    blank_ratio = blank_lines / len(lines) if lines else 0
    
    # Trailing whitespace detection
    trailing_ws = sum(1 for line in lines if line != line.rstrip())
    trailing_ratio = trailing_ws / len(lines) if lines else 0
    
    # Bracket-newline pattern (opening brace followed by newline)
    bracket_newlines = source_code.count('{\n') + source_code.count('[\n')
    total_brackets = source_code.count('{') + source_code.count('[')
    bracket_nl_ratio = bracket_newlines / total_brackets if total_brackets else 0
    
    return LayoutFingerprint(
        avg_line_length=avg_length,
        line_length_stddev=stddev,
        indentation_unit=indent_unit,
        blank_line_ratio=blank_ratio,
        trailing_whitespace_ratio=trailing_ratio,
        max_line_length=max(line_lengths) if line_lengths else 0,
        bracket_newline_ratio=bracket_nl_ratio
    )

# Example: Extract layout fingerprint
code = """
def calculate_sum(numbers):
    total = 0
    for num in numbers:
        if num > 0:
            total += num
    
    return total
"""

fingerprint = extract_layout_fingerprint(code)
print(f"Layout fingerprint: {fingerprint}")
```

### Multi-Author Classification Pipeline

Train ensemble classifiers on labeled corpus and apply to unknown samples with confidence scoring and explainable attribution.

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
import numpy as np
from typing import List, Dict, Tuple

class CodeAuthorClassifier:
    """
    Multi-author attribution system using ensemble learning.
    
    Combines multiple feature types for robust classification.
    """
    
    def __init__(self):
        self.classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.author_labels = []
        
    def extract_all_features(self, source_code: str) -> np.ndarray:
        """Combine AST, n-gram, and layout features into unified vector."""
        ast_features = extract_ast_features(source_code)
        ngram_profile = extract_ngram_profile(source_code, n=3)
        layout = extract_layout_fingerprint(source_code)
        
        # Convert to fixed-length feature vector
        feature_vector = list(ast_features.values()) + [
            len(ngram_profile),  # vocabulary size
            sum(ngram_profile.values()),  # total tokens
            layout.avg_line_length,
            layout.indentation_unit,
            layout.blank_line_ratio,
        ]
        
        return np.array(feature_vector)
    
    def train(self, code_samples: List[str], authors: List[str]) -> Dict[str, float]:
        """
        Train classifier on labeled code corpus.
        
        Returns cross-validation scores for model validation.
        """
        # Extract features from all samples
        X = np.array([self.extract_all_features(code) for code in code_samples])
        y = np.array(authors)
        
        # Store author mapping
        self.author_labels = list(set(authors))
        
        # Normalize features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train classifier
        self.classifier.fit(X_scaled, y)
        
        # Cross-validation
        cv_scores = cross_val_score(self.classifier, X_scaled, y, cv=5)
        
        return {
            'mean_accuracy': cv_scores.mean(),
            'std_accuracy': cv_scores.std(),
            'cv_scores': cv_scores.tolist()
        }
    
    def predict(self, unknown_code: str) -> Tuple[str, float, Dict[str, float]]:
        """
        Attribute authorship to unknown code sample.
        
        Returns:
            predicted_author: Most likely author
            confidence: Probability of prediction
            all_probabilities: Full distribution across authors
        """
        features = self.extract_all_features(unknown_code)
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        
        # Get probability distribution
        probabilities = self.classifier.predict_proba(features_scaled)[0]
        predicted_idx = np.argmax(probabilities)
        
        # Map to author names
        prob_dist = {
            author: float(prob) 
            for author, prob in zip(self.classifier.classes_, probabilities)
        }
        
        predicted_author = self.classifier.classes_[predicted_idx]
        confidence = probabilities[predicted_idx]
        
        return predicted_author, confidence, prob_dist

# Example usage
classifier = CodeAuthorClassifier()

# Training data (simplified - real system needs 100+ samples per author)
training_samples = [
    "def foo(x):\n  return x*2",  # Author A
    "def bar(y):\n  return y*2",  # Author A
    "def baz(x): return x * 2",   # Author B (different style)
    "def qux(y): return y * 2",   # Author B
]
training_authors = ["Alice", "Alice", "Bob", "Bob"]

# Train model
metrics = classifier.train(training_samples, training_authors)
print(f"Training accuracy: {metrics['mean_accuracy']:.3f}")

# Predict unknown sample
unknown = "def process(z):\n  return z*2"
author, confidence, probs = classifier.predict(unknown)
print(f"Predicted author: {author} (confidence: {confidence:.3f})")
print(f"Probability distribution: {probs}")
```

### Obfuscation Resistance Analysis

Detect and measure style manipulation attempts by comparing sample consistency against known author baselines.

```python
from typing import List, Tuple
import numpy as np

def detect_obfuscation(code_sample: str, 
                       author_baseline_samples: List[str],
                       threshold: float = 0.7) -> Tuple[bool, float, Dict[str, float]]:
    """
    Detect potential style obfuscation by measuring deviation from author baseline.
    
    Args:
        code_sample: Unknown sample to analyze
        author_baseline_samples: Known authentic samples from claimed author
        threshold: Similarity threshold below which obfuscation is suspected
    
    Returns:
        is_obfuscated: Boolean flag
        consistency_score: Overall similarity to baseline [0, 1]
        feature_deviations: Per-feature deviation metrics
    """
    # Extract features from unknown sample
    sample_features = extract_all_features(code_sample)
    
    # Extract baseline feature distribution
    baseline_features = np.array([
        extract_all_features(code) for code in author_baseline_samples
    ])
    baseline_mean = baseline_features.mean(axis=0)
    baseline_std = baseline_features.std(axis=0)
    
    # Compute z-scores for deviation detection
    z_scores = np.abs((sample_features - baseline_mean) / (baseline_std + 1e-6))
    
    # Feature-level deviation metrics
    feature_names = ['ast_if_ratio', 'ast_for_ratio', 'avg_line_length', 
                     'indentation_unit', 'blank_line_ratio']
    deviations = {
        name: float(z_score) 
        for name, z_score in zip(feature_names, z_scores[:len(feature_names)])
    }
    
    # Overall consistency score (inverse of mean absolute z-score)
    consistency = 1.0 / (1.0 + z_scores.mean())
    
    is_obfuscated = consistency < threshold
    
    return is_obfuscated, consistency, deviations

# Example: Detect style manipulation
authentic_samples = [
    "def func1(x):\n    return x + 1",
    "def func2(y):\n    return y * 2",
    "def func3(z):\n    return z - 1",
]

suspicious_sample = "def func(a):return a+1"  # Compressed style

obfuscated, score, deviations = detect_obfuscation(
    suspicious_sample, 
    authentic_samples
)

print(f"Obfuscation detected: {obfuscated}")
print(f"Consistency score: {score:.3f}")
print(f"Feature deviations: {deviations}")
```

## Concept Reference

| Concept | Technical | Plain | Importance |
|---------|-----------|-------|------------|
| Code Stylometry | The computational analysis and quantification of programming style characteristics to identify authorship patterns, behavioral signatures, or distinguish between developers through machine learning and statistical methods. | A method of figuring out who wrote a piece of computer code by looking at their unique coding habits and style, similar to how handwriting analysis identifies people by their penmanship. | 1.00 |
| Authorship Attribution | The process of determining the likely author of a code artifact by extracting discriminative features from source code and applying classification algorithms trained on labeled samples from known programmers. | Identifying who wrote a program by matching the code's style to patterns from known programmers, like matching a painting to an artist's signature style or brushwork techniques. | 0.95 |
| Stylistic Features | Quantifiable code characteristics including lexical patterns (identifier naming conventions), syntactic structures (control flow preferences), layout choices (indentation, whitespace), and complexity metrics that form author fingerprints. | The specific habits a programmer has, like how they name variables, organize their code, use spaces, or prefer certain programming structures—these all combine into a unique signature. | 0.90 |
| Feature Extraction | The systematic process of parsing source code to identify and quantify measurable attributes such as abstract syntax tree metrics, n-gram frequencies, complexity measures, and formatting patterns. | Breaking down code into measurable characteristics that can be counted and compared, like measuring how often someone uses certain words or sentence structures in writing. | 0.88 |
| Machine Learning Classification | The application of supervised learning algorithms (e.g., SVM, Random Forest, Neural Networks) trained on labeled code samples to predict authorship by recognizing learned stylistic patterns in unlabeled code. | Teaching a computer program to recognize different coding styles by showing it many examples, so it can later guess who wrote new code it hasn't seen before. | 0.87 |
| Code Obfuscation | Deliberate transformation of source code to obscure its logic, structure, or stylistic characteristics while preserving functional equivalence, potentially defeating stylometric attribution attempts. | Intentionally making code harder to understand or disguising your coding style, like wearing a disguise to hide your identity or writing with your opposite hand. | 0.85 |
| Abstract Syntax Tree (AST) | A hierarchical tree representation of source code's grammatical structure where each node represents a syntactic construct, enabling structural analysis independent of surface formatting. | A diagram that shows the underlying structure and grammar of code, similar to diagramming sentences in English class, which reveals patterns in how someone writes programs. | 0.84 |
| Forensic Analysis | The application of stylometric techniques in cybersecurity and legal contexts to attribute malicious code, identify insider threats, verify code provenance, or provide evidence in intellectual property disputes. | Using coding style analysis to solve computer crimes or legal disputes, like determining who wrote a virus or proving someone stole code. | 0.83 |
| N-gram Analysis | Statistical modeling of contiguous sequences of n code tokens (lexical units, operators, keywords) to capture local patterns and idioms characteristic of individual programming styles. | Looking at common sequences or patterns of code elements that appear together repeatedly, like noticing someone always uses certain word combinations in their writing. | 0.82 |
| De-anonymization | The process of reverse-engineering anonymous code authorship through stylometric analysis, potentially compromising programmer privacy by linking pseudonymous contributions to real identities. | Figuring out who wrote code that was meant to be anonymous by analyzing their coding style, which can reveal someone's identity even when they tried to hide it. | 0.81 |

## Glossary

| Term | Definition | Concept IDs |
|------|------------|-------------|
| Code Stylometry | Computational analysis of programming style to identify authorship through statistical patterns and machine learning applied to code features. | 1 |
| Authorship Attribution | Determining code authors by matching stylistic patterns against known programmer profiles using classification algorithms. | 2 |
| Stylistic Features | Quantifiable coding habits including naming conventions, formatting choices, and structural preferences that form a programmer's unique fingerprint. | 3, 10, 11, 12 |
| Feature Extraction | Systematically parsing code to identify and quantify measurable attributes that serve as input for classification models. | 4 |
| Machine Learning Classification | Training algorithms on labeled code samples to predict authorship by learning to recognize distinctive style patterns. | 5, 15 |
| Abstract Syntax Tree (AST) | Hierarchical representation of code's grammatical structure enabling analysis of programming patterns independent of surface formatting. | 7 |
| N-gram Analysis | Statistical modeling of contiguous code token sequences to capture characteristic local patterns and idioms in programming style. | 8 |
| Code Complexity Metrics | Quantitative measures of code intricacy that characterize an author's tendency toward simple or elaborate solution approaches. | 13 |
| Forensic Analysis | Applying stylometric techniques to attribute malicious code, identify insider threats, or provide evidence in legal disputes. | 9 |
| Code Obfuscation | Deliberately transforming code to obscure stylistic characteristics while preserving functionality, serving as a countermeasure to attribution. | 6 |
| De-anonymization | Reverse-engineering anonymous code authorship through stylometric analysis, potentially compromising programmer privacy protections. | 16 |
| Cross-language Stylometry | Analyzing whether programming style patterns persist across different languages or require language-specific feature engineering. | 14 |
| Temporal Style Evolution | Changes in coding style over time due to skill development, team influences, or tool adoption, requiring adaptive attribution approaches. | 17 |

## Edge Cases & Warnings

- ⚠️ **Insufficient Training Data**: Attribution accuracy degrades severely with fewer than 50-100 code samples per author; small training sets lead to overfitting and false positives.
- ⚠️ **Collaborative Code Contamination**: Pair programming, code reviews, and team style guides homogenize individual signatures, reducing discriminative power in shared codebases.
- ⚠️ **Temporal Drift Invalidation**: Programmer styles evolve significantly over 2-5 years; models trained on old samples may misattribute recent code as style changes accumulate.
- ⚠️ **Cross-Language Brittleness**: Features effective in one language (e.g., Python indentation patterns) may not transfer to others (e.g., C bracket styles); language-specific models required.
- ⚠️ **Code Generation Tool Interference**: AI-assisted coding (Copilot, ChatGPT) introduces external style patterns that contaminate author signatures and confound attribution.
- ⚠️ **Adversarial Obfuscation Success**: Determined adversaries using automated style transfer tools can reduce attribution accuracy to near-random guessing; forensic confidence scoring essential.
- ⚠️ **Privacy and Ethical Concerns**: De-anonymization capabilities threaten open-source contributor privacy and whistleblower protection; use requires informed consent and legal authorization.
- ⚠️ **Small Sample Bias**: Short code fragments (<50 lines) contain insufficient stylistic information; attribution confidence should scale with sample size.
- ⚠️ **Boilerplate Contamination**: Auto-generated code, framework templates, and copied snippets dilute author signal; preprocessing to remove common patterns required.
- ⚠️ **False Confidence in Legal Contexts**: Stylometric evidence should supplement rather than replace traditional forensic methods; probabilities are not certainties.

## Quick Reference

```python
# Minimal stylometry pipeline: extract features, train, predict

import ast
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np

def quick_fingerprint(code: str) -> list:
    """Extract minimal feature set for rapid attribution."""
    tree = ast.parse(code)
    nodes = list(ast.walk(tree))
    
    return [
        len(code),  # code length
        len(code.split('\n')),  # line count
        code.count('    ') / len(code),  # indentation ratio
        len([n for n in nodes if isinstance(n, ast.If)]) / len(nodes),  # if ratio
        len([n for n in nodes if isinstance(n, ast.For)]) / len(nodes),  # for ratio
    ]

# Training phase
train_codes = ["def f(x):\n    return x*2", "def g(y): return y*2"]
train_labels = ["AuthorA", "AuthorB"]
X = np.array([quick_fingerprint(c) for c in train_codes])
clf = RandomForestClassifier(n_estimators=50).fit(X, train_labels)

# Prediction phase
unknown = "def h(z):\n    return z*2"
prediction = clf.predict([quick_fingerprint(unknown)])[0]
confidence = clf.predict_proba([quick_fingerprint(unknown)]).max()

print(f"Author: {prediction}, Confidence: {confidence:.2f}")
```

---
_Generated by Philosopher's Stone v4 — EchoSeed_
Philosopher's Stone v4 × Skill Forge × EchoSeed
