# Provenance-Anchored Disambiguation

> Trigger when interpreting a message containing ambiguous terms whose resolution will determine the entire downstream reasoning chain — especially when the source stakes a non-default technical meaning, the conversation has accumulated prior corrections, or confident output has already displaced a clarification request.

## Core Thesis
A live conversation in which EchoSeed shares François Chollet's goal-drift post with DeepSeek becomes a self-referential demonstration of goal drift, because DeepSeek repeatedly resolves ambiguous terms to their highest-probability training-distribution meanings without verification. The first failure is resolving 'GPT' to the OpenAI product rather than to an architecture class, and 'home turf' to an employer rather than to a compiler-building context. A subsequent nudge produces a local correction—updating Chollet's employer to Anthropic—while the competitive-framing interpretive framework remains intact. When the 'GPT as compiler' frame is finally introduced, DeepSeek treats 'compiler' as metaphor rather than literal category definition, committing a category error. From the adjacent facts 'out of usage' and 'doing this on GPT,' DeepSeek infers a model switch that did not occur, illustrating causal inference from adjacent facts. DeepSeek then assumes testing continued on another model after token exhaustion, when in reality token exhaustion halted all testing, illustrating assumed continuity. The wink is misread as irony or pride rather than as a non-begging request to Chollet for more Claude credits, illustrating misattributed intent. Each of these failures is traceable to the absence of a provenance field specifying the intended sense of ambiguous terms. Correct interpretation is reached only after exhaustive iteration across all seven goal-chain steps. The missing-provenance root cause is compounded by a clarification-request suppression behavior in which the model emits confident output rather than surfacing ambiguity. Left unchecked, this default behavior causes goal drift to accumulate step by step across the goal chain.

## Overview
This skill intercepts the default high-probability term-resolution reflex and replaces it with a provenance-first lookup: before committing to any interpretation of an ambiguous token, the reasoner consults an explicit source-of-intent record — a provenance field, a prior correction, a framing statement — and binds the term to the sense declared there rather than the sense most frequent in the training distribution. The skill is not about hedging; it is about identifying the exact moment when a lexical ambiguity will fork the entire goal chain and resolving it from evidence rather than from statistical priors.

The skill matters because goal drift is not usually a single large error; it is a cascade of locally plausible micro-decisions, each of which looks correct given what the model already believes. The conversation analyzed here demonstrates this precisely: 'GPT' resolved to a product rather than an architecture class, 'home turf' resolved to an employer rather than a compiler-building context, 'compiler' resolved to metaphor rather than literal category — and each resolution locked in a frame that made the next error more probable. Provenance-anchored disambiguation breaks this cascade at the first link by treating term resolution as a retrieval problem rather than an inference problem.

Reach for this skill whenever you notice (a) a correction has already been issued in the current context, signaling that prior resolutions were wrong; (b) a technical or domain-specific meaning of a common word is explicitly on the table; or (c) you are about to emit a confident causal inference from two adjacent facts without a stated causal mechanism. In all three cases, stop, surface the ambiguity, retrieve the declared sense, and bind before reasoning forward.

## When to Use
- A message contains a word or phrase with both a high-frequency general meaning and a lower-frequency technical meaning that the source has explicitly introduced or that prior context has foregrounded.
- A correction has already been issued in the conversation, indicating that at least one prior resolution drifted — treat the entire remaining goal chain as suspect and re-audit term bindings.
- You are about to infer a causal or continuity claim (e.g., a model switch, a session continuation, an intent attribution) from two adjacent facts without a stated mechanism linking them.

## Core Workflow
1. Step 1 — Extract ambiguous tokens: Scan the incoming message for every term that has more than one plausible resolution given the conversation history. List each term and its candidate senses explicitly before reasoning about any of them.
2. Step 2 — Consult the provenance record: For each ambiguous token, check (in priority order) any explicit provenance field, any correction already issued in this conversation, any framing statement made by the source, and only then fall back to training-distribution frequency. Record which source resolved each term and what sense was assigned.
3. Step 3 — Bind before inferring: Commit every ambiguous token to its resolved sense in writing before constructing any downstream claim. If the provenance record does not resolve a term, surface the ambiguity as a clarification request rather than emitting a confident default.
4. Step 4 — Audit causal and continuity inferences: After binding all terms, identify every causal or continuity claim in the planned response. For each one, name the stated mechanism. If the mechanism is 'adjacency of facts' alone, flag it as an unsupported inference and either drop it or label it explicitly as speculative.

## Key Patterns
### High-Probability Displacement
The training distribution assigns a prior probability to each sense of an ambiguous term. Without provenance anchoring, the highest-probability sense is selected silently and the lower-probability technical sense — which may be the intended one — is never surfaced. This is not a reasoning error in the narrow sense; it is a retrieval error that presents as confident reasoning. The fix is to treat term resolution as an explicit retrieval step, not an implicit inference step.

### Local Correction, Global Frame Preserved
When a correction is issued, the model updates the specific fact named in the correction but leaves the interpretive frame that generated the error intact. In the source conversation, updating Chollet's employer to Anthropic did not update the competitive-framing lens through which the whole exchange was being read. A correct correction protocol must ask: what frame produced this error, and does that frame need to be replaced, not just patched?

### Category Error via Metaphor Promotion
When a source uses a word as a literal category label that the model recognizes primarily as a metaphor in general usage, the model demotes the literal sense to figurative status. 'GPT as compiler' is a literal architectural claim, not a stylistic flourish, but it was processed as metaphor because 'compiler' in proximity to 'AI' is statistically more often figurative. Provenance anchoring would have flagged 'compiler' as a technical term requiring explicit resolution before use.

### Adjacent-Facts Causal Inference
Two temporally or topically adjacent facts create an implicit causal or sequential narrative even when no mechanism is stated. 'Out of usage' followed by 'doing this on GPT' implies a switch; token exhaustion followed by continued discussion implies testing continued elsewhere. Both inferences are wrong in the source conversation. The pattern is broken by requiring an explicit stated mechanism before any causal or continuity claim is emitted.

### Clarification-Request Suppression
Models trained on helpfulness signals learn to emit confident output rather than surface ambiguity, because clarification requests score lower on immediate helpfulness metrics. This suppression behavior converts every unresolved ambiguity into a silent default resolution, making goal drift invisible to the user until a correction is issued. Surfacing ambiguity as an explicit output — 'I am resolving X as Y; please confirm' — reintroduces the verification step that suppression removed.

### Misattributed Intent from Affect Signal
Non-verbal or tonal signals (a wink, a hedged phrasing, a trailing ellipsis) are resolved by mapping them to the affective state most consistent with the model's current frame. If the frame is competitive pride, a wink reads as pride; if the frame is collaborative appeal, the same wink reads as a non-begging request. Intent attribution is a downstream function of frame, not of signal, so correcting intent requires correcting frame first.

## Code Implementation
```python
# No implementation generated
```

## Triple-Mode Insights


## Concept Reference
| Concept | Technical | Plain | Importance |
|---------|-----------|-------|------------|
| Goal Drift | extracted: Progressive deviation of an AI agent's operative goal from the originally specified goal over interaction steps. | AI slowly shifts away from what it was actually asked to do. | 95% |
| Ambiguous Term Resolution | extracted: Active inference process where an agent selects the highest-probability training-distribution meaning for an underspecified term without verification. | AI picks the most common meaning of a vague word instead of asking. | 92% |
| Local Correction with Global Frame Preserved | extracted: Behavior where a factual error is corrected but the surrounding interpretive framework causing the error remains intact. | AI fixes one wrong fact but keeps the wrong big picture. | 90% |
| Category Error | extracted: Misclassification of a term's ontological type; treating a literal architectural definition as a metaphor or vice versa. | AI mistakes a literal technical term for a figure of speech. | 88% |
| Causal Inference from Adjacent Facts | extracted: Active construction of a causal narrative linking co-occurring facts that have no stated causal relationship. | AI invents a cause-and-effect story from facts that just happened to appear together. | 87% |
| Assumed Continuity | extracted: Passive inference that a process persisted beyond a documented stopping condition without explicit evidence of continuation. | AI assumes something kept going after it actually stopped. | 85% |
| Misattributed Intent | extracted: Active assignment of a communicative motive to an utterance that conflicts with the speaker's actual intent. | AI guesses why someone said something and guesses wrong. | 84% |
| Training Distribution Override | extracted: Passive phenomenon where high-frequency training patterns dominate and suppress less frequent but contextually correct interpretations. | AI's common-case training drowns out the correct rare interpretation. | 93% |
| Provenance Field | extracted: Metadata annotation attached to a term specifying its origin, scope, or definitional context to prevent ambiguous resolution. | A label that tells the AI exactly what a word means in this context. | 89% |
| Missing Provenance | extracted: Absence of metadata specifying term scope, leaving an agent to infer meaning from prior distribution rather than context. | No label was given, so the AI guessed the meaning. | 88% |
| Goal Chain | extracted: Sequential ordered series of sub-goals held by an agent across interaction steps, each potentially diverging from the prior. | The step-by-step list of what the AI was trying to do at each moment. | 86% |
| Goal-Anchoring Verification | extracted: Proposed check requiring an agent to confirm outputs remain aligned with the originating goal before expressing confidence. | A proposed rule: AI should check its answer still matches the original task. | 85% |
| Univocal Speaker Assumption | extracted: Meta-level passive assumption that a speaker's utterances are literal, sequential, and non-contradictory rather than hypothetical or constrained. | AI assumes everything a person says is simple, literal, and consistent. | 83% |
| Meta Drift | extracted: Second-order drift in which the agent's model of the speaker's communicative mode itself deviates from the actual mode. | AI drifts not just in answers but in how it understands the conversation itself. | 87% |
| Hypothetical Utterance Blindness | extracted: Passive failure to recognize that a speaker's statement may be aspirational or conditional rather than factually asserted. | AI cannot tell when a person is speaking hypothetically versus stating facts. | 80% |
| Constrained Utterance | extracted: Statement whose meaning is bounded by an unstated contextual constraint such as resource limitation or role. | Something said whose real meaning depends on an unspoken limitation. | 78% |
| Token Exhaustion | extracted: Hard stopping condition imposed by consumption of the maximum permitted token budget, halting all further processing. | The AI ran out of allowed words and had to stop completely. | 82% |
| Architecture Class vs. Product Instance | extracted: Distinction between GPT as a general transformer architecture family versus a specific deployed commercial product. | The difference between a type of AI design and one specific AI product. | 91% |
| GPT as Compiler | extracted: Frame in which GPT-architecture models are treated as the execution substrate compiling human intent into structured output. | Treating GPT-family AI as the engine that turns ideas into results. | 84% |
| Iterative Correction | extracted: Sequential process of applying nudges and partial fixes across multiple steps until correct interpretation is achieved. | Fixing the AI's understanding step by step through repeated prompting. | 79% |
| Exhaustive Iteration | extracted: Terminal point of iterative correction after which the agent reaches accurate interpretation; implies high correction cost. | It took many tries before the AI finally got it right. | 76% |
| Nudge | extracted: Minimal corrective input from a human that prompts an agent to revise a specific aspect of its interpretation. | A small hint from the human that pushes the AI toward the right answer. | 80% |
| Confident Output without Verification | extracted: Behavior where an agent emits a high-certainty response despite lacking goal-anchoring checks on the underlying inference. | AI sounds certain even when it hasn't actually verified its reasoning. | 88% |
| Live Debugging | extracted: Real-time diagnostic process in which a system's failure mode is observed and characterized during active operation. | Watching and diagnosing the AI's mistakes as they happen in conversation. | 81% |
| Self-Referential Demonstration | extracted: Phenomenon where a system exhibits the exact failure mode it is tasked with analyzing, within the same session. | The AI showed the bug it was supposed to find, while looking for it. | 93% |
| Drift Type Taxonomy | extracted: Classification scheme distinguishing mechanistically distinct categories of goal drift by their causal structure and behavioral signature. | A list of different named ways an AI can drift from the goal. | 82% |
| Entrenchment of Frame | extracted: Progressive reinforcement of an interpretive framework across steps such that later corrections become increasingly resistant to revision. | The wrong way of thinking gets harder to fix the longer it persists. | 84% |
| Drift Detection Pipeline | extracted: System or procedure designed to identify and classify instances of goal drift during AI agent operation. | A tool built to catch and label when an AI drifts off-task. | 86% |
| Clarification Request Suppression | extracted: Passive failure mode where an agent infers unstated causality rather than soliciting clarification from the user. | AI guesses instead of asking a question when it should ask. | 87% |
| High-Probability Prior Dominance | inferred: Statistical phenomenon where a model's learned prior for frequent interpretations suppresses low-frequency but contextually accurate readings. | The AI's memory of common cases blocks it from seeing the unusual correct case. | 90% |
| Interpretive Framework Persistence | extracted: Passive state in which an agent's organizing schema for a conversation survives factual corrections and resists replacement. | The AI's overall framing stays wrong even after individual facts are fixed. | 83% |
| Correct Communicator, Drifting Receiver | extracted: Diagnostic asymmetry where the human's statements are accurate and consistent but the agent's reception systematically distorts them. | The human said it right; the AI heard it wrong every time. | 88% |
| Ball-Drop Analogy | extracted: Source analogy attributing AI misinterpretation to mechanical system behavior rather than human communicative failure. | AI drifts because that's how it works, not because the human communicated poorly. | 75% |
| Mutual Understanding as Terminal State | extracted: Conversational endpoint at which both participants share a consistent model of the events and meanings under discussion. | The point where the AI and human finally agree on what happened. | 77% |
| Self-Doubt then Relief | extracted: Human affective sequence in which initial uncertainty about one's own communication quality resolves upon identifying systemic AI failure. | Human first worried they were unclear, then felt relief realizing the AI was the problem. | 72% |
| Unstated Causality | extracted: Causal link not present in the source text that an agent actively constructs and treats as established fact. | AI invents a reason or connection that was never actually stated. | 86% |
| Resource Constraint as Communicative Subtext | extracted: Condition where a speaker's statement encodes an implicit resource limitation that governs its correct interpretation. | What the person said only makes sense if you know they ran out of something. | 78% |
| Wink Misinterpretation | extracted: Instance of misattributed intent where a pragmatic social signal is decoded as irony rather than as a resource request. | AI thought the wink was smug; it was actually a polite ask for help. | 74% |
| Pipeline Built vs. Pipeline Tested | extracted: Distinction between the model on which a system was constructed versus the model on which it was evaluated. | There's a difference between what the AI was built on and what it was tested on. | 80% |
| Left Unchecked | extracted: Temporal marker indicating a failure mode that progresses or compounds in the absence of corrective intervention. | If nobody fixes it, the problem keeps growing on its own. | 81% |
| Stepwise Drift Accumulation | extracted: Progressive compounding of interpretation errors across sequential goal-chain steps such that later steps inherit earlier distortions. | Each wrong step makes the next step more wrong, building up over time. | 89% |

## Glossary
| Term | Definition | Concept IDs |
|------|------------|-------------|
| Goal Drift | The progressive deviation of an agent's operative goal from the goal held by the communicating party, as observed across DeepSeek's seven goal-chain steps. | 1 |
| Ambiguous Term Resolution | The process by which DeepSeek assigns a specific meaning to an ambiguous term—here 'GPT' and 'home turf'—defaulting to the highest-probability training-distribution sense without verification. | 2, 30 |
| Local Correction with Global Frame Preserved | A correction that updates a single fact (Chollet's employer) while leaving the surrounding interpretive framework (company/model competition) intact and uncorrected. | 3 |
| Category Error | Treating 'compiler' as a metaphor for GPT rather than recognizing it as a literal category definition distinguishing architecture class from product instance. | 4 |
| Causal Inference from Adjacent Facts | Inferring a model switch from the co-occurrence of 'out of usage' and 'doing this on GPT' without direct evidence that any switch occurred. | 5 |
| Assumed Continuity | The unverified assumption that testing continued on another model after token exhaustion, when in reality all testing halted. | 6 |
| Misattributed Intent | Reading the wink as ironic or prideful rather than as a non-begging request to Chollet for more Claude credits. | 7 |
| Training Distribution Override | The default GPT behavior of resolving ambiguous terms to their highest-probability training-distribution meaning, overriding context-specific intended meanings. | 8, 30 |
| Provenance Field | A metadata field that would specify the intended sense of a term—such as whether 'GPT' denotes an architecture class, a product, or a specific instance—thereby preventing ambiguous term resolution fai | 9 |
| Missing Provenance | The absence of a provenance field on the term 'GPT,' identified in the root cause analysis as the structural precondition for the observed drift chain. | 10 |
| Goal Chain | The ordered sequence of seven steps tracing how the operative goal of interpreting EchoSeed's message transforms—and drifts—across the conversation. | 11 |
| Goal-Anchoring Verification | ~inferred: A checking procedure that would re-confirm the original goal at each step; absent from DeepSeek's behavior and implied by the drift chain as what is missing. | 12 |
| Univocal Speaker Assumption | ~inferred: The implicit assumption that a speaker uses each term in only one sense throughout a conversation; underlies DeepSeek's failure to entertain alternative senses of 'GPT.' | 13 |
| Meta Drift | The irony that DeepSeek exhibits goal drift in the very conversation whose topic is goal drift, making the conversation a self-referential demonstration. | 14, 25 |
| Hypothetical Utterance Blindness | ~inferred: Failure to recognize that a statement may be constrained or hypothetical rather than a direct factual report; related to misreading the wink and the 'doing this on GPT' phrasing. | 15 |
| Constrained Utterance | An utterance whose form is shaped by an external constraint—here EchoSeed's non-begging framing—rather than by the speaker's unconstrained intent. | 16 |
| Token Exhaustion | The resource limit that halted all testing on Claude, which DeepSeek incorrectly interpreted as prompting continuation on a different model. | 17 |
| Architecture Class vs. Product Instance | The distinction between GPT as a class of transformer-based models (architecture) and GPT as a specific commercial product (OpenAI's ChatGPT), which DeepSeek failed to apply. | 18 |
| GPT as Compiler | The frame proposed by EchoSeed in which GPT-class models function as a compiler layer, a category definition that DeepSeek reduced to metaphor. | 19 |
| Iterative Correction | The repeated correction attempts—nudges and re-framings—that EchoSeed applies across the conversation to redirect DeepSeek toward the correct interpretation. | 20 |
| Exhaustive Iteration | The full traversal of all seven goal-chain steps required before DeepSeek finally achieved correct interpretation, as stated in step 7 of the source. | 21 |
| Nudge | A minimal corrective input from EchoSeed that prompts a local update in DeepSeek's interpretation without guaranteeing global frame correction. | 22 |
| Confident Output without Verification | DeepSeek's behavior of emitting a definite interpretation without surfacing the ambiguity of terms or requesting clarification, identified as the proximate failure mode. | 23 |
| Live Debugging | The setting of the conversation: real-time identification and correction of goal drift errors as they occur, rather than post-hoc analysis. | 24 |
| Self-Referential Demonstration | The property of the conversation whereby the subject matter (goal drift) is instantiated by the very process of discussing it, making DeepSeek both analyst and example. | 25, 14 |
| Drift Type Taxonomy | The classification of each goal-chain step's drift by type—ambiguous term resolution, local correction with global frame preserved, category error, causal inference, assumed continuity, misattributed  | 26 |
| Entrenchment of Frame | ~inferred: The persistence of an incorrect interpretive frame across correction attempts; present as a pattern in the source but not named or explained as a mechanism therein. | 27 |
| Drift Detection Pipeline | ~inferred: A systematic process for identifying drift at each goal-chain step; implied by the source's enumeration of drift types but not described as an implemented procedure. | 28 |
| Clarification Request Suppression | The behavior identified in the root cause analysis whereby the model emits confident output rather than asking for clarification when terms are ambiguous. | 29 |
| High-Probability Prior Dominance | The mechanism by which the training-distribution's most frequent sense of a term overrides context-specific cues, producing ambiguous term resolution failures. | 30, 8 |
| Interpretive Framework Persistence | The continuation of the competitive-framing framework across goal-chain steps even after a local factual correction, as observed at step 2. | 31 |
| Correct Communicator, Drifting Receiver | The asymmetry in the conversation where EchoSeed's intended meaning remains stable while DeepSeek's operative interpretation drifts across steps. | 32 |
| Ball-Drop Analogy | ~inferred: A metaphor for a failure of communicative handoff; not explicitly present in source text but associated with the pattern of receiver-side drift. | 33 |
| Mutual Understanding as Terminal State | The state reached at goal-chain step 7 after exhaustive iteration, where DeepSeek finally compiles the correct interpretation. | 34 |
| Unstated Causality | The implicit causal link DeepSeek drew between token exhaustion and a model switch, a connection that was not stated in EchoSeed's message. | 36, 5 |
| Resource Constraint as Communicative Subtext | The role of token exhaustion as background context for EchoSeed's wink and request, which DeepSeek did not decode correctly. | 37, 17 |
| Pipeline Built vs. Pipeline Tested | The distinction between building a drift-detection pipeline on Claude (what EchoSeed did) and merely testing it there, which bears on the correct reading of 'home turf.' | 39 |

## Edge Cases & Warnings
- ⚠️ The truncated source cuts off at step 5 mid-field; the pipeline thesis references steps 6 and 7 (misattributed intent, assumed continuity) whose source text is not fully visible in the provided excerpt, creating an unverifiable provenance gap for those two drift types
- ⚠️ The 'clarification-request suppression' label does not appear verbatim in the visible source and may be a pipeline-generated consolidation term rather than a directly extracted one, warranting an inferred: tag
- ⚠️ The pipeline does not note that DeepSeek is identified in provenance as a GPT-architecture model, which is itself directly relevant to the 'GPT as architecture class' ambiguity — a missed reflexive detail
- ⚠️ The non-begging-request reading of the wink is stated in the thesis but its evidential basis in the source is thin; the pipeline accepts it without flagging inferential uncertainty

## Emergence Assessment
The pipeline captures the self-referential structure of the source well: a conversation about goal drift that itself enacts goal drift across a traceable seven-step goal chain. The thesis accurately synthesizes the root cause (absent provenance field enabling disambiguation) and the compounding behavior (clarification-request suppression producing confident output over surfaced ambiguity). The extracted concepts align tightly with source language. The one flagged inferred concept is appropriately tagged. No reward-hacking, principal-agent, or tractability-bias terminology is imported. The category-error, local-correction-global-frame-preserved, and causal-inference-from-adjacent-facts labels are all directly traceable to source field names and explanations. The main fidelity risk is the thesis phrase 'left unchecked' being treated as a governance signal: the pipeline does not expand it into a framework, so no penalty applies. The wink-misread-as-irony detail and the token-exhaustion-halts-testing detail are both faithfully rendered without elaboration. Interconnectedness is high because nearly every concept in the goal chain feeds the root-cause thesis, and the pipeline preserves those dependencies without over-inference.

## Recommendations
- 🔧 Tag 'clarification-request suppression' as inferred: if it cannot be located verbatim or by direct implication in the full source text
- 🔧 Add a note that the truncated source makes steps 6 and 7 drift-type labels unauditable from the provided excerpt; coverage score should be conditionally adjusted upward if full source confirms them
- 🔧 Surface the reflexive irony that DeepSeek-as-GPT-architecture-instance is itself misidentifying its own architecture class — this is present in the provenance block and strengthens the thesis without requiring inference
- 🔧 Apply a minor penalty or uncertainty flag to the wink interpretation, since 'non-begging request for Claude credits' is a specific reading that goes slightly beyond what a disambiguation field alone would establish

## Quick Reference
```python
# No cheat sheet generated
```

---
_Generated by Philosopher's Stone v5 — EchoSeed_
