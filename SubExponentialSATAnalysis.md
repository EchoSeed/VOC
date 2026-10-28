# Sub-Exponential SAT Analysis

> Trigger when evaluating claims about exact counting or decision algorithms for k-CNF SAT, assessing ETH-adjacent complexity results, or auditing pipeline outputs that blend algorithmic runtime bounds with conjecture-refutation claims — especially when source notation carries parameter-dependent exponents.

## Core Thesis
A deterministic algorithm is presented that exactly counts satisfying assignments of any k-CNF formula, running in time (n+m)^{O_k(1)} · 2^{n/(log n)^{1/2+o(1)}} using only poly_k(n,m) space for every fixed k≥2. This algorithm operates in sub-exponential time relative to the linear exponential baseline, refuting the Exponential Time Hypothesis, which postulates that deciding 3-SAT requires 2^{Ω(n)} time in the worst case. The result holds in the fixed-k parameter regime and achieves polynomial space complexity simultaneously with its sub-exponential time bound.

## Overview
This skill extracts and faithfully represents the primitive concepts embedded in complexity-theoretic algorithm claims, particularly those involving exact counting over unrestricted formula classes, simultaneous space-time tradeoffs, and the refutation of widely-held conjectures. It enforces source fidelity at the level of notation (O_k(1), poly_k, o(1) terms), rhetorical framing (postulate vs. theorem), and scope qualification (arbitrary k-CNF, fixed-k regime). The skill is designed to prevent inflation where pipeline synthesis might obscure meaningful distinctions — such as the difference between a uniform and a k-dependent polynomial bound, or between refuting a conjecture and disproving a theorem.

The skill matters because complexity results live and die by their exact quantifiers. A bound like 2^{n/(log n)^{1/2+o(1)}} is not simply 'sub-exponential' — it is a specific family of functions with a non-trivial denominator whose o(1) term leaves the asymptotic open to improvement. Collapsing this precision into a narrative gloss can mislead downstream reasoning about the result's significance or its distance from polynomial time. Similarly, the simultaneous achievement of polynomial space and sub-exponential time is a non-trivial structural claim that pipelines routinely absorb into background framing rather than foregrounding as an independent concept.

Reach for this skill when a pipeline output has synthesized a formal complexity result and you need to verify that each extracted concept is directly traceable to source language, that parameter-dependence has not been silently universalized, that conjecture-vs-theorem distinctions are preserved, and that the scope of the formula class (arbitrary, restricted, parametric) is faithfully carried into every derived claim.

## When to Use
- A pipeline thesis presents an algorithmic runtime bound and you need to verify the bound's exact form — including any o(1), O_k(1), or poly_k notation — has not been smoothed into a weaker or stronger claim
- A source attributes refutation to a named conjecture (ETH, SETH, etc.) and you must confirm the pipeline preserves the conjecture/postulate status rather than treating it as a proven lower bound
- A complexity result involves simultaneous space and time bounds and you need to ensure both are extracted as independent concepts rather than collapsed into a single characterization
- The source scopes an algorithm to 'arbitrary' or 'unrestricted' inputs and you need to surface that scope explicitly rather than letting it dissolve into generic framing
- Parameter-dependent notation (subscript-k bounds) appears in source and you must check whether the pipeline correctly flags k-dependence as a distinct nuance rather than presenting the result as uniform

## Core Workflow
1. Step 1 — Anchor to source notation: Identify every formal expression in the source (runtime, space, parameter notation) and bind each pipeline concept to its exact source string; flag any concept lacking a direct source anchor as [inferred]
2. Step 2 — Decompose simultaneous claims: Separate composite assertions (e.g., simultaneous sub-exponential time AND polynomial space) into independent concepts, each verified against its own source passage and page anchor if provenance tags are present
3. Step 3 — Preserve epistemic framing: Check whether each conjecture or hypothesis cited in the source is labeled as postulate, conjecture, or proven lower bound; carry that label into the extracted concept metadata without upgrading or downgrading the epistemic status
4. Step 4 — Scope and parameter audit: Confirm that (a) formula-class scope ('arbitrary k-CNF') is stated explicitly, (b) parameter-dependence (O_k(1), poly_k) is flagged as k-dependent rather than uniform, and (c) the fixed-k regime is noted as a constraint on where guarantees hold
5. Step 5 — Inflation check: For each elaboration, verify that analogies and insights do not introduce theoretical extensions presented as extracted facts; mark any extension with [inferred] and ensure the audit trail distinguishes source-attested claims from pipeline-generated elaborations

## Key Patterns
### Parameter-Dependent Bound Preservation
When source notation uses subscript-k exponents (O_k(1), poly_k(n,m)), the result is not uniform across all k — polynomial degree and space constants are functions of k. Pipelines frequently absorb this into 'polynomial space' without flagging the k-dependence. Preserving this distinction matters because the result does not assert a single fixed polynomial bound valid for all k simultaneously.

### Conjecture Epistemic Lock
ETH and similar hypotheses are postulates, not proven lower bounds. The source's own language ('postulates') encodes this status. A pipeline that reports only 'ETH is refuted' without noting ETH's conjectural status implicitly suggests the algorithm defeated a theorem, overstating the result's logical force. The correct framing is: the algorithm provides a direct counterexample to a widely-believed open conjecture.

### Simultaneous Bound Independence
Sub-exponential time and polynomial space are each individually significant, and their simultaneous achievement is a third, distinct claim. Most sub-exponential algorithms achieve their runtime at the cost of super-polynomial space. Extracting both bounds as independent concepts — and then explicitly noting their co-occurrence as a non-trivial structural property — prevents either from being treated as a corollary of the other.

### Scope Foregrounding for Unrestricted Classes
The word 'arbitrary' in 'arbitrary k-CNF' is a substantive claim: no structural restriction on the formula is assumed. This distinguishes the result from parametric or restricted-class algorithms. Pipelines tend to absorb 'arbitrary' into general framing ('any k-CNF formula') without surfacing it as a standalone concept that differentiates this result from, for example, algorithms that require bounded treewidth or other structural conditions.

### o(1) Exponent Term Precision
The o(1) in the denominator exponent 2^{n/(log n)^{1/2+o(1)}} means the bound is a family of functions asymptotically approaching 2^{n/(log n)^{1/2}}, not a single clean closed-form runtime. This leaves room for improvement within the stated form and signals the result is an asymptotic characterization rather than a tight bound. Collapsing this to '2^{n/(log n)^{1/2}}' drops a meaningful qualifier present in the source.

## Decision Logic
Pseudocode / Decision Logic:
1. ENTRY: Source contains a formal algorithm claim with exact runtime and space expressions for a combinatorial problem class (k-CNF SAT, #k-SAT, or structurally equivalent)
2. EXTRACT runtime bound using exact source string: (n+m)^{O_k(1)} · 2^{n/(log n)^{1/2+o(1)}}; preserve o(1) term without collapsing; tag as sub-exponential relative to 2^{Ω(n)} baseline only if source supports that comparison
3. EXTRACT space bound using exact source string: poly_k(n,m); flag k-subscript as indicating k-dependent polynomial degree, not a uniform constant-degree polynomial
4. EXTRACT formula class scope: source says 'arbitrary k-CNF' — surface 'arbitrary' as an explicit no-structural-restriction qualifier; do not absorb into generic 'any k-CNF' without noting the scope claim
5. EXTRACT fixed-k regime: source scopes guarantees to 'every fixed k≥2'; flag that k is treated as a constant parameter and that behavior for k growing with n is outside source scope [inferred: unbounded-k regime unaddressed]
6. EXTRACT ETH refutation: source states algorithm 'refutes ETH'; bind to k=3 case (3-SAT decision subsumed by counting result); preserve source framing of ETH as 'postulate' in metadata — do not upgrade to 'proven lower bound'
7. EXTRACT ETH lower bound postulate: source defines ETH as 'postulates that deciding 3-SAT on n variables requires 2^{Ω(n)} time in the worst case'; this is a conjecture, not a theorem; the algorithm's runtime is a direct counterexample
8. BRANCH — simultaneity check: if source asserts both sub-exponential time AND polynomial space, extract as independent concepts AND as a third joint concept (simultaneous achievement); do not collapse into one
9. BRANCH — inflation check: for each extracted concept, verify analogy and insight do not introduce claims absent from source; if an insight extends beyond source (e.g., whether counting subsumes decision as corollary), mark [inferred]
10. BRANCH — provenance tags: if source carries page anchors ([p2], [p4]), bind each extracted concept to its anchor; flag any cross-page conflation (e.g., if space bound and ETH refutation appear at different document positions)
11. OUTPUT: Set of concepts each with (a) exact source string anchor, (b) scope qualifiers, (c) epistemic label (postulate/theorem/algorithm claim), (d) k-dependence flag where applicable, (e) [inferred] markers on extensions
Note: use source-attested terms only; mark any extension as [inferred]

## Triple-Mode Insights
### deterministic exact #k-SAT algorithm
**🎯 Decision:** Applies when counting all satisfying assignments of any k-CNF formula with n variables and m clauses; the algorithm is deterministic (no randomness) and exact (not approximate), operative for every fixed k≥2 as stated in source.
**🎭 Analogy:** A meticulous librarian who counts every valid shelving arrangement exhaustively, without dice or shortcuts, guaranteeing a precise total.
**💡 Insight:** Exactness combined with determinism is stronger than decision or approximate counting; the source implies #k-SAT (counting) is solved, which subsumes k-SAT (decision) as a corollary — though that subsumption is inferred from standard complexity theory, not stated explicitly.

### polynomial space complexity
**🎯 Decision:** Applies for every fixed k≥2; space is poly_k(n,m), meaning polynomial in n and m with degree depending on k, operative simultaneously with the stated time bound on formulas of n variables and m clauses.
**🎭 Analogy:** A hiker solving a vast maze who only needs a small notepad regardless of maze size — memory footprint grows gently even as the search explodes in time.
**💡 Insight:** Achieving sub-exponential time while using only polynomial space is non-trivial; most sub-exponential algorithms trade space for speed. The source asserts both hold simultaneously, suggesting the algorithm avoids exponential-size data structures — the mechanism is unstated and any elaboration would be inferred.

### sub-exponential time complexity
**🎯 Decision:** The runtime 2^{n/(log n)^{1/2+o(1)}} is sub-exponential in n, strictly between polynomial and 2^{Ω(n)}; applies for every fixed k≥2 on n-variable, m-clause k-CNF formulas as the core quantitative claim of the source.
**🎭 Analogy:** A sprinter who never reaches full marathon pace but still finishes faster than anyone expected — slower than polynomial but dramatically faster than brute-force exponential.
**💡 Insight:** The o(1) term in the exponent's denominator means the bound is not a single clean function but a family approaching 2^{n/(log n)^{1/2}}; this leaves room for improvement within the stated form, though whether the bound is tight is unaddressed by the source.

### fixed-k parameter regime
**🎯 Decision:** The algorithm's guarantees — space poly_k(n,m) and the stated time bound — hold for every fixed k≥2, meaning k is treated as a constant parameter, not a growing input; applicable across all fixed clause-width instances as stated.
**🎭 Analogy:** A recipe that works perfectly for any fixed oven temperature you choose in advance, with ingredient quantities scaled to that setting — the recipe breaks only if the temperature itself keeps changing.
**💡 Insight:** Fixing k allows degree and constants to absorb k-dependence into poly_k notation; the source does not address the regime where k grows with n. Whether the algorithm extends to unbounded k is outside source warrant and would be inferred.

### ETH refutation
**🎯 Decision:** The source explicitly states the algorithm refutes ETH; this follows directly because the algorithm solves 3-SAT (k=3 case) in 2^{n/(log n)^{1/2+o(1)}} time, which is sub-exponential, contradicting ETH's postulate that 3-SAT requires 2^{Ω(n)} time.
**🎭 Analogy:** A single counterexample key that opens a lock everyone believed unpickable — one demonstrated sub-exponential solution invalidates the universal worst-case conjecture.
**💡 Insight:** Refuting ETH via a #k-SAT counting algorithm is stronger than needed; a decision algorithm suffices to refute ETH. The counting result implies the decision result, making the refutation a byproduct of a harder achievement — though that logical hierarchy is inferred from complexity theory, not stated in source.

### ETH lower bound postulate
**🎯 Decision:** ETH is characterized in the source as the postulate that deciding 3-SAT on n variables requires 2^{Ω(n)} time in the worst case; it serves as the benchmark the algorithm's runtime is measured against and explicitly contradicts.
**🎭 Analogy:** A speed limit sign reading '2^{Ω(n)} minimum' that theorists believed no algorithm could duck under — the new algorithm drives straight through it.
**💡 Insight:** The source frames ETH as a postulate, not a theorem, underscoring that its negation is logically consistent; the algorithm's result does not expose a flaw in ETH's reasoning but simply provides a direct constructive counterexample. Whether the algorithm is correct is assumed by the source authors but not proven within the excerpt.

## Concept Reference
| Concept | Technical | Plain | Importance | Citation |
|---------|-----------|-------|------------|----------|
| deterministic exact #k-SAT algorithm | extracted: deterministic algorithm for exactly counting satisfying assignments of arbitrary k-CNF formula | A deterministic algorithm that exactly counts solutions to any k-CNF formula. | 100% | _"We give a deterministic algorithm for exactly counting the satisfying assignments of an arbitrary k-CNF formula."_ |
| ETH refutation | extracted: algorithm refutes the Exponential Time Hypothesis (ETH) by running faster than ETH's lower bound | The algorithm's runtime disproves a major conjecture about SAT hardness. | 98% | _"Our algorithm refutes the Exponential Time Hypothesis (ETH)"_ |
| ETH lower bound postulate | extracted: ETH postulates deciding 3-SAT on n variables requires 2^{Ω(n)} time in the worst case | ETH claims 3-SAT needs exponential time; this algorithm shows that claim is false. | 90% | _"which postulates that deciding 3-SAT on n variables requires 2^{Ω(n)} time in the worst case"_ |
| polynomial space complexity | extracted: algorithm uses poly_k(n, m) space for fixed k ⩾ 2 on formula with n variables and m clauses | The algorithm runs in polynomial space relative to input size for any fixed k. | 85% | _"our algorithm uses poly_k(n, m) space"_ |
| fixed-k parameter regime | extracted: algorithm applies for every fixed k ⩾ 2; complexity bounds are parameterized by k | The algorithm's guarantees hold for any fixed clause-width k at least 2. | 75% | _"For every fixed k ⩾ 2"_ |
| sub-exponential time complexity | extracted: runs in time (n + m)^{O_k(1)} 2^{n/(log n)^{1/2 + o(1)}}, sub-exponential in n | The algorithm runs faster than any purely exponential time bound in n. | 57% | _"runs in time (n + m)^{O_k(1)} 2^{n/(log n)^{1/2 + o(1)}}"_ |

## Glossary
| Term | Definition | Concept IDs |
|------|------------|-------------|
| exact #k-SAT algorithm | A deterministic procedure that counts, rather than merely decides, the number of satisfying assignments of an arbitrary k-CNF formula. | 1 |
| ETH refutation | The algorithmic result that disproves the Exponential Time Hypothesis by providing a concrete algorithm whose running time falls below the 2^{Ω(n)} lower bound the hypothesis postulates. | 5 |
| ETH lower bound postulate | The conjecture that deciding 3-SAT on n variables requires 2^{Ω(n)} time in the worst case. | 6 |
| polynomial space complexity | The property that the algorithm's memory usage is bounded by poly_k(n,m), a polynomial in n and m whose degree may depend on k. | 2 |
| fixed-k parameter regime | The setting in which k is treated as a constant, allowing time and space bounds to carry implicit dependences on k while remaining polynomial or sub-exponential in n and m. | 4 |
| sub-exponential time complexity | A running time of the form 2^{n/(log n)^{1/2+o(1)}}, which grows faster than polynomial but slower than any 2^{Ω(n)} exponential. | 3 |

## Substantiation Summary
**Sound:** 5 · **Weak:** 1 · **Unsound (demoted):** 0

**Coverage Gaps Detected:**
- 📍 decision vs. counting distinction — source contrasts 'deciding 3-SAT' (ETH) with 'exactly counting' (the algorithm), but no concept captures this decision/counting boundary explicitly

## Edge Cases & Warnings
- ⚠️ The source explicitly scopes the algorithm to 'arbitrary k-CNF' formulas — the word 'arbitrary' is a meaningful claim (no structural restriction) that the thesis paraphrases as 'any k-CNF' which is acceptable but the emphasis on unrestricted formula class could have been foregrounded as a standalone concept. The pipeline appears to have absorbed this into the general framing rather than extracting it as a distinct concept.
- ⚠️ The O_k(1) and poly_k(n,m) notation signals that the polynomial exponents and space bounds are functions of k, not uniform constants — this parameter-dependence on k is a subtle but source-present nuance that the thesis mentions ('fixed-k parameter regime') but the pipeline did not surface as a standalone extracted concept distinct from the space complexity concept.
- ⚠️ ETH is identified as a postulate in the source ('postulates that deciding 3-SAT requires 2^{Omega(n)} time in the worst case') — the pipeline correctly notes refutation but does not flag that the source's own language frames ETH as a conjecture/postulate, meaning the result is a formal refutation of a widely-believed open hypothesis, a rhetorical and technical weight not fully surfaced.
- ⚠️ The source text tags [p2] and [p4] indicate this is a multi-page document with specific provenance anchors — the pipeline did not use these structural markers to distinguish claims appearing at different document positions, which could matter if the algorithm description and the ETH-refutation claim carry different levels of formality or proof status within the source document.

## Emergence Assessment
The pipeline thesis accurately reconstructs the core algorithmic claim and its ETH-refuting consequence. The sub-exponential framing ('sub-exponential time relative to the linear exponential baseline') is a valid and non-inflated characterization of 2^{n/(log n)^{1/2+o(1)}} versus 2^{Omega(n)}, directly traceable to source language. The simultaneous poly-space and sub-exponential time claim is genuinely present in source. No reward-hacking or principal-agent vocabulary introduced. The thesis is faithful. The 6-elaboration count is a mild inflation risk but no fabricated concepts are evident from the audit. The missing 7th concept is a minor gap; the 6 extracted appear accurate. Overall emergence fidelity is high.


## Reflexive Observations
_None detected_
## Recommendations
- 🔧 Surface 'arbitrary k-CNF' (no structural restriction on formula) as a standalone concept with direct source citation to distinguish this from parametric or restricted-class results.
- 🔧 Extract the k-dependence of polynomial factors (O_k(1), poly_k) as a distinct concept — it is present in source notation and is technically meaningful, signaling that the result is not uniform in k.
- 🔧 Note the [p2]/[p4] provenance tags in the audit trail; if the pipeline supports page-level provenance, bind each concept to its source page anchor to improve traceability and catch any cross-page conflation.
- 🔧 The ETH postulate framing ('postulates') should be preserved in extracted concept metadata to correctly characterize the result as refuting a conjecture rather than a proven lower bound — this distinction is source-present and algorithmically significant.

## Quick Reference
Quick-Reference Pattern:
- When: Source presents a k-CNF algorithm with formal runtime/space bounds and invokes ETH refutation
- Do: Extract each bound using exact source notation (O_k(1), poly_k, o(1) term intact); surface 'arbitrary k-CNF' scope explicitly; flag k-dependence in polynomial bounds; label ETH as postulate not theorem; extract simultaneous space-time achievement as an independent third concept
- Check: (1) o(1) term preserved in exponent, (2) poly_k carries k-subscript flag, (3) ETH metadata reads 'postulate/conjecture' not 'proven lower bound', (4) 'arbitrary' formula class noted as no-structural-restriction claim, (5) fixed-k regime boundary stated, (6) any pipeline insight extending beyond source marked [inferred]
- Avoid: Collapsing 2^{n/(log n)^{1/2+o(1)}} to a cleaner closed form; treating ETH refutation as disproving a theorem; universalizing k-dependent polynomial bounds as uniform; absorbing 'arbitrary' scope into generic framing; presenting simultaneous space-time bound as a single concept rather than three (time, space, joint)

---
_Generated by Philosopher's Stone v5 — EchoSeed_
