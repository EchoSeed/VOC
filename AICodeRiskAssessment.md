# AI Code Risk Assessment

> Trigger when evaluating whether to introduce, expand, or audit AI-generated code in a software pipeline—especially when weighing productivity gains against hidden quality costs, validating AI output before production deployment, or assessing security and reliability posture of codebases with LLM-assisted components.

## Core Thesis
Large language models introduced into software development pipelines produce code that is statistically unreliable, semantically opaque, and structurally hazardous, as demonstrated by Bjarne Stroustrup's warning that AI-generated output accumulates bugs, bloat, and security vulnerabilities faster than human engineers can remediate them. The nondeterministic nature of generative AI means that minor prompt variations can cascade unpredictably across an entire codebase, making consistent validation practically intractable. LLMs operate within bounded context windows and carry training data biases that distort their understanding of architectural intent, producing hallucinated APIs, incorrect abstractions, and dependency conflicts. Human code review remains the primary safeguard, yet the cognitive load imposed by reviewing AI output at scale strains developer capacity and accelerates workforce attrition. Senior engineers who built and retained institutional knowledge of critical systems are retiring rather than adapt to AI-assisted workflows, creating dangerous gaps in software craftsmanship. The absence of true code ownership in AI-assisted development erodes accountability structures that historically enforced correctness and maintainability standards. Technical debt compounds rapidly when AI-generated code lacks adherence to established software architecture principles, forcing expensive cascading refactors. Static analysis tools and software testing suites, designed around human coding patterns, are often insufficient to catch the novel failure modes AI introduces. Critical infrastructure software faces disproportionate risk because its correctness requirements exceed what current generative AI reliability levels can satisfy. Prompt engineering offers insufficient mitigation because sensitivity to phrasing renders reproducibility and auditability structurally compromised. Sustained human oversight of AI systems is therefore not optional but foundational to preserving software reliability, security, and the long-term viability of engineering as a disciplined craft.

## Overview
This skill provides a structured framework for evaluating the risks introduced when large language models participate in software development. It synthesizes Stroustrup's critique and empirical failure patterns—nondeterminism, hallucinated APIs, training-corpus security debt, and prompt sensitivity—into actionable decision criteria that engineers and architects can apply during code review, threat modeling, and toolchain design. Rather than treating AI-generated code as equivalent to human-authored code with occasional errors, the skill reframes the risk profile: AI bugs cluster around confident plausibility rather than complexity, making them harder to catch with conventional tooling and review heuristics calibrated for human output.

The skill matters because the productivity benefits of LLM-assisted development are front-loaded while the costs—accumulated technical debt, eroded code ownership, institutional knowledge loss, and security regressions—are deferred and often invisible until a critical failure surfaces. Senior engineers retiring rather than adapt represents a concrete signal that the human oversight layer, which remains the primary safeguard, is under structural pressure precisely as AI-generated code volume grows. This creates a compounding risk: more AI output arriving for review at the same time the review capacity and contextual expertise to catch its novel failure modes is shrinking.

Reach for this skill whenever an AI-assisted workflow is being designed, audited, or defended to stakeholders. It is equally applicable when scoping a penetration test on a codebase with LLM contributions, when deciding validation requirements for a new AI-assisted feature, or when building the institutional argument for why human oversight must be preserved as a non-negotiable architectural constraint rather than an optional quality gate.

## When to Use
- An AI-generated pull request is being reviewed and the reviewer needs a structured checklist to distinguish plausible-but-incorrect code from genuinely correct code
- A team is threat-modeling a system where LLM tools contributed to authentication, memory management, or network-facing code paths
- An engineering leader is evaluating whether to expand AI-assisted development and needs a risk-benefit framework that accounts for workforce attrition and institutional knowledge erosion
- A prompt change has produced unexpectedly large diffs and the team needs to reason about whether the codebase's behavior guarantees still hold
- A static analysis or test suite is being calibrated and the team needs to understand why existing tooling may miss AI-introduced failure modes

## Core Workflow
1. Classify the risk tier of the code under evaluation: consumer-facing, business-critical, or critical infrastructure—this determines how much validation overhead is justified and which failure modes are categorically unacceptable
2. Audit the AI contribution surface: identify every function, module, or dependency where LLM output was used directly or lightly edited, and flag code paths that handle untrusted input, authentication, memory, or network I/O for elevated scrutiny
3. Apply the plausibility-vs-correctness test: for each flagged path, ask whether the code satisfies the described happy path while potentially neglecting edge cases, hallucinating API contracts, or inheriting pre-2020 security debt from training corpus patterns
4. Evaluate the oversight structure: determine whether prompts are version-controlled alongside code, whether review capacity is sufficient given AI output volume, and whether institutional knowledge of the relevant subsystem still exists in the team—if senior engineers have left, treat their formerly-owned code as under-validated regardless of test coverage
5. Produce a risk disposition: accept with standard review, accept with augmented validation (formal verification, adversarial testing, expert audit), or reject pending human rewrite of the highest-risk components

## Key Patterns
### Plausibility-Correctness Gap
LLMs optimize for token probability, producing code that resembles correct code more reliably than it produces code that is correct. Human reviewers rate fluent, readable code positively, creating a systematic feedback loop where AI output passes review not because it is verified but because it is convincing. The gap between resemblance and correctness is where security vulnerabilities and logic errors accumulate invisibly.

### Prompt History as Hidden Dependency
AI-generated codebases carry an invisible dependency on the prompt history that produced them. Without version-controlling prompts alongside code, teams lose the ability to reason about why code looks the way it does, reproduce prior states, or audit the chain of decisions. A small prompt change can cascade unpredictably across an entire codebase, making the effective reproducibility of the system structurally compromised.

### Oversight Degradation Under Scale
Human oversight degrades precisely when it is most needed. As AI-generated code volume grows, reviewers face more output with less context per unit of time, making meaningful review statistically impractical without new tooling calibrated specifically to AI failure modes. The workforce attrition signal—senior engineers retiring rather than adapt—indicates the human safeguard layer is eroding concurrent with the risk surface expanding.

### Training Corpus Security Debt
AI models trained on public code inherit the security vulnerabilities prevalent in that corpus. Patterns that were common before major CVE disclosures may be statistically reinforced in outputs, meaning AI-generated code can regress security posture to pre-remediation baselines even when the team believes it is writing contemporary code. Threat modeling must account for this historical contamination rather than treating AI output as neutral.

### Front-Loaded Productivity, Deferred Cost
The productivity benefit of AI-assisted development is realized immediately at the point of generation; the costs—bugs, bloat, technical debt, knowledge gaps—compound over the maintenance lifetime of the code. This temporal asymmetry makes AI adoption appear economically rational in short evaluation windows while hiding the total cost of ownership. Risk assessment must explicitly project deferred remediation costs to give a complete picture.

## Code Implementation
```python
from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class RiskTier(Enum):
    """Classifies the consequence domain of the software under evaluation."""
    CONSUMER = "consumer"                  # crashes annoy users
    BUSINESS_CRITICAL = "business_critical"  # outages cost money or data
    CRITICAL_INFRASTRUCTURE = "critical_infrastructure"  # failures harm people


class Disposition(Enum):
    """Final risk disposition after assessment."""
    ACCEPT_STANDARD = "accept_standard_review"
    ACCEPT_AUGMENTED = "accept_augmented_validation"
    REJECT_PENDING_REWRITE = "reject_pending_human_rewrite"


@dataclass
class CodeSurface:
    """
    Represents a single unit of AI-contributed code under evaluation.
    A 'surface' is any function, module, or dependency where LLM output
    was used directly or lightly edited.
    """
    name: str
    handles_untrusted_input: bool = False
    handles_authentication: bool = False
    handles_memory: bool = False
    handles_network: bool = False
    prompt_version_controlled: bool = False   # is the generating prompt tracked?
    has_institutional_owner: bool = True       # does a senior engineer still own this?
    happy_path_only_tested: bool = False       # test suite covers only nominal cases?
    uses_external_apis: bool = False           # risk of hallucinated API contracts


@dataclass
class AssessmentResult:
    """Structured output of a single surface assessment."""
    surface_name: str
    risk_score: float          # 0.0 (low) – 1.0 (critical)
    flags: list[str] = field(default_factory=list)
    disposition: Optional[Disposition] = None
    rationale: str = ""


def score_surface(surface: CodeSurface, tier: RiskTier) -> AssessmentResult:
    """
    Evaluate a single AI-contributed code surface and return a risk score
    with an explicit disposition recommendation.

    Scoring logic encodes the key patterns from the skill:
      - Plausibility-Correctness Gap (happy path testing masks AI failure modes)
      - Training Corpus Security Debt (untrusted input + auth paths)
      - Oversight Degradation (missing institutional owner + no prompt versioning)
      - Prompt History as Hidden Dependency (prompt not version-controlled)
    """
    score: float = 0.0
    flags: list[str] = []

    # --- Security-sensitive code paths (Training Corpus Security Debt) ---
    if surface.handles_untrusted_input:
        score += 0.25
        flags.append(
            "Handles untrusted input: AI models statistically reinforce "
            "pre-remediation vulnerability patterns from training corpus."
        )
    if surface.handles_authentication:
        score += 0.25
        flags.append(
            "Authentication path: hallucinated invariants here are "
            "categorically dangerous; requires adversarial testing."
        )
    if surface.handles_memory:
        score += 0.20
        flags.append(
            "Memory management: LLMs reproduce patterns from unsafe historical "
            "code; buffer and use-after-free risks elevated."
        )
    if surface.handles_network:
        score += 0.15
        flags.append(
            "Network-facing: injection and deserialization attack surfaces "
            "are underrepresented in AI happy-path generation."
        )

    # --- Plausibility-Correctness Gap (test suite calibration) ---
    if surface.happy_path_only_tested:
        score += 0.20
        flags.append(
            "Test suite covers nominal cases only: AI bugs cluster at edge cases "
            "and adversarial inputs, which existing suites likely miss."
        )

    # --- Prompt History as Hidden Dependency ---
    if not surface.prompt_version_controlled:
        score += 0.10
        flags.append(
            "Prompt not version-controlled: codebase carries an invisible "
            "dependency; reproducibility and auditability are compromised."
        )

    # --- Hallucinated API contracts ---
    if surface.uses_external_apis:
        score += 0.15
        flags.append(
            "External API usage: risk of hallucinated contracts—syntactically "
            "valid calls to nonexistent or misused library interfaces."
        )

    # --- Oversight Degradation Under Scale ---
    if not surface.has_institutional_owner:
        score += 0.15
        flags.append(
            "No institutional owner: senior engineer knowledge gap means "
            "tacit correctness requirements are not held by anyone on the team."
        )

    # --- Tier multiplier: critical infrastructure amplifies every defect ---
    tier_multiplier = {
        RiskTier.CONSUMER: 1.0,
        RiskTier.BUSINESS_CRITICAL: 1.3,
        RiskTier.CRITICAL_INFRASTRUCTURE: 1.7,
    }[tier]

    final_score = min(score * tier_multiplier, 1.0)

    # --- Disposition mapping ---
    if final_score < 0.35:
        disposition = Disposition.ACCEPT_STANDARD
        rationale = "Risk level is manageable with standard peer review."
    elif final_score < 0.65:
        disposition = Disposition.ACCEPT_AUGMENTED
        rationale = (
            "Elevated risk requires augmented validation: adversarial testing, "
            "expert audit of flagged paths, or formal verification where feasible."
        )
    else:
        disposition = Disposition.REJECT_PENDING_REWRITE
        rationale = (
            "Risk score exceeds acceptable threshold for this tier. "
            "Human rewrite of flagged components is required before deployment."
        )

    return AssessmentResult(
        surface_name=surface.name,
        risk_score=round(final_score, 3),
        flags=flags,
        disposition=disposition,
        rationale=rationale,
    )


def assess_codebase(
    surfaces: list[CodeSurface],
    tier: RiskTier,
) -> dict:
    """
    Run a full codebase assessment across all AI-contributed surfaces.
    Returns a structured report with per-surface results and an aggregate
    disposition for the overall merge/deployment decision.
    """
    results = [score_surface(s, tier) for s in surfaces]

    # Aggregate: the codebase disposition is driven by the highest-risk surface.
    max_score = max(r.risk_score for r in results) if results else 0.0
    reject_count = sum(
        1 for r in results if r.disposition == Disposition.REJECT_PENDING_REWRITE
    )

    if reject_count > 0:
        aggregate_disposition = Disposition.REJECT_PENDING_REWRITE
    elif max_score >= 0.35:
        aggregate_disposition = Disposition.ACCEPT_AUGMENTED
    else:
        aggregate_disposition = Disposition.ACCEPT_STANDARD

    return {
        "tier": tier.value,
        "surface_count": len(surfaces),
        "aggregate_risk_score": round(max_score, 3),
        "aggregate_disposition": aggregate_disposition.value,
        "surfaces": [
            {
                "name": r.surface_name,
                "risk_score": r.risk_score,
                "disposition": r.disposition.value if r.disposition else None,
                "rationale": r.rationale,
                "flags": r.flags,
            }
            for r in results
        ],
    }


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    surfaces = [
        CodeSurface(
            name="auth_token_validator",
            handles_authentication=True,
            handles_untrusted_input=True,
            prompt_version_controlled=False,
            has_institutional_owner=False,
            happy_path_only_tested=True,
        ),
        CodeSurface(
            name="report_pdf_renderer",
            handles_untrusted_input=False,
            prompt_version_controlled=True,
            has_institutional_owner=True,
            happy_path_only_tested=False,
        ),
        CodeSurface(
            name="payment_gateway_client",
            handles_network=True,
            uses_external_apis=True,
            handles_untrusted_input=True,
            prompt_version_controlled=False,
            has_institutional_owner=True,
            happy_path_only_tested=True,
        ),
    ]

    report = assess_codebase(surfaces, RiskTier.BUSINESS_CRITICAL)
    print(json.dumps(report, indent=2))
```

## Triple-Mode Insights
### AI-generated code
**🎯 Decision:** Apply when evaluating whether to use LLM output directly in production. An agent invokes this concept when weighing speed-of-generation against hidden quality costs—bugs, bloat, security holes—especially in contexts where validation capacity is limited or expertise is declining.
**🎭 Analogy:** A fast-talking contractor who builds walls quickly but hides the plumbing inside them. The house looks finished, but every repair requires demolishing what was built.
**💡 Insight:** AI-generated code creates a knowledge debt: teams ship code no one fully understands, making future audits, refactors, and incident responses structurally harder. The productivity gain is front-loaded; the cost is compounding.

### Human oversight of AI systems
**🎯 Decision:** Apply when determining accountability boundaries in AI-assisted workflows. An agent invokes this when outputs affect safety, correctness, or security—especially where AI nondeterminism means identical inputs may not reproduce the same behavior for debugging.
**🎭 Analogy:** A co-pilot who occasionally hallucinates instrument readings. The human pilot must never fully delegate, even when the automation is mostly reliable, because 'mostly' is insufficient at altitude.
**💡 Insight:** Oversight degrades precisely when it's most needed: as AI-generated code volume grows, human reviewers face more code with less context, making meaningful review statistically impractical without new tooling designed specifically for AI output auditing.

### Software security
**🎯 Decision:** Apply when assessing risk posture of any codebase with AI-generated components. Invoke when threat modeling, conducting audits, or deploying to sensitive environments—AI-generated code introduces non-obvious attack surfaces because it may pattern-match insecure historical examples.
**🎭 Analogy:** A locksmith trained by reading old manuals, some of which describe locks already known to be pickable. They reproduce functional locks, but also reproduce known weaknesses without flagging them.
**💡 Insight:** AI models trained on public code inherit the security debt of that corpus. Vulnerabilities that were common before 2020 may be statistically reinforced in outputs, meaning AI-generated code could regress security posture to older, less hardened norms.

### Security vulnerabilities
**🎯 Decision:** Apply during code review, penetration testing scoping, or when triaging AI-generated pull requests. An agent invokes this when any code path handles untrusted input, authentication, memory, or network I/O—domains where LLMs frequently produce plausible but flawed implementations.
**🎭 Analogy:** A forgery that passes casual inspection but fails under ultraviolet light. The vulnerability exists in the code as written; only specialized scrutiny reveals it.
**💡 Insight:** LLMs tend to generate code that satisfies the described happy path while neglecting edge cases that attackers specifically target. The vulnerability isn't random—it's structurally biased toward omission of defensive logic, making static analysis insufficient without semantic review.

### AI hallucination in code
**🎯 Decision:** Apply when an LLM produces syntactically valid but semantically incorrect code—referencing nonexistent APIs, misusing library contracts, or generating logically coherent but functionally broken logic. Invoke during any automated code generation pipeline as a baseline risk factor.
**🎭 Analogy:** A confident student who writes a grammatically perfect essay citing books that don't exist. The fluency masks the fabrication; surface quality is anti-correlated with catching the error.
**💡 Insight:** Hallucinated code is more dangerous than obvious errors because it passes initial review. A function that compiles, runs, and produces plausible output can harbor a hallucinated invariant that only fails under specific production conditions, potentially years after deployment.

### Code validation
**🎯 Decision:** Apply whenever AI-generated code enters a pipeline. An agent invokes this to determine what validation layers—static analysis, testing, formal verification, expert review—are sufficient given the risk level of the target system. Validation cost must be factored into AI productivity calculations.
**🎭 Analogy:** Airport security for luggage: the faster bags are loaded, the more critical the scanner becomes. Skipping the scan to improve throughput defeats the safety purpose of the system entirely.
**💡 Insight:** Validation designed for human-written code is poorly calibrated for AI output. Human bugs cluster around complexity; AI bugs cluster around confident plausibility. Existing test suites may not cover the specific failure modes AI introduces, requiring new validation strategies, not just more of the same.

### Large language models (LLMs)
**🎯 Decision:** Apply when reasoning about the capabilities and failure modes of the underlying technology. An agent invokes this when explaining why AI code generation behaves nondeterministically, why prompt sensitivity exists, or why outputs can be fluent but incorrect.
**🎭 Analogy:** A vast library that has read everything but understood nothing causally—it predicts the next sentence based on what sentences usually follow, not because it models the world the sentences describe.
**💡 Insight:** LLMs optimize for token probability, not correctness. This means they produce code that resembles correct code more reliably than they produce code that is correct. The gap between resemblance and correctness is where most production failures originate.

### Generative AI
**🎯 Decision:** Apply when scoping what class of system is producing outputs. An agent invokes this when distinguishing generative systems—which synthesize novel outputs—from deterministic tools, to properly set expectations about reproducibility, auditability, and error characterization.
**🎭 Analogy:** The difference between a calculator and an improvising jazz musician. The calculator gives the same answer every time; the musician gives you something new each performance, sometimes transcendent, sometimes off-key.
**💡 Insight:** Generative AI's value proposition—novelty and fluency—is in direct tension with software engineering's core need for reproducibility and predictability. Applying generative tools to deterministic problem domains creates a category mismatch that no amount of prompt engineering fully resolves.

### Prompt sensitivity
**🎯 Decision:** Apply when evaluating stability of AI-assisted development workflows. An agent invokes this when a small change in requirements or phrasing produces disproportionately large changes in generated output—signaling that the codebase is coupled to prompt phrasing rather than to durable design decisions.
**🎭 Analogy:** A blueprint that changes if you describe the building differently to the architect each morning. The structure reflects the conversation, not a stable engineering intent.
**💡 Insight:** Prompt sensitivity means AI-generated codebases have an invisible dependency: the prompt history. Without version-controlling prompts alongside code, teams lose the ability to reason about why code looks the way it does, making maintenance a form of archaeology.

### Code correctness
**🎯 Decision:** Apply as the primary criterion when evaluating any code output. An agent invokes this when distinguishing code that compiles and appears to work from code that provably satisfies its specification under all valid inputs, including adversarial ones.
**🎭 Analogy:** A bridge that holds weight in good weather but hasn't been tested in a storm. Correctness under nominal conditions is necessary but not sufficient for engineering confidence.
**💡 Insight:** AI models are rewarded during training for outputs that humans rate positively, and humans rate plausible, readable code positively. This creates a systematic bias where correctness is approximated by appearance, not verified by proof—making AI-generated code correct on average but unreliable in the tail.

### Critical infrastructure software
**🎯 Decision:** Apply when the consequence of software failure extends beyond the application to physical systems, public safety, or societal continuity. An agent invokes this to escalate validation requirements, restrict AI-generated code usage, or mandate human expert review before deployment.
**🎭 Analogy:** The difference between a typo in a novel and a typo in a medical dosage formula. The medium is the same; the consequence of error is categorically different.
**💡 Insight:** Critical infrastructure amplifies every latent defect in AI-generated code. A hallucinated boundary condition in consumer software causes a crash; the same defect in a grid control system or water treatment application can cause cascading physical harm with no rollback option.

### Unpredictable behavior
**🎯 Decision:** Apply when characterizing the risk profile of systems with AI-generated components. An agent invokes this when prompt sensitivity, nondeterminism, or hallucination combine to make output behavior difficult to bound or anticipate—especially under distribution-shifted inputs.
**🎭 Analogy:** A vending machine that sometimes gives you what you selected, sometimes gives you something adjacent, and occasionally gives you nothing—with no error light. You can't trust it; you also can't fully distrust it.
**💡 Insight:** Unpredictability in AI systems is not random noise—it has structure. Failures cluster at edge cases, novel inputs, and domain boundaries. Understanding this structure allows targeted testing strategies, but also reveals that standard random testing will systematically miss the highest-risk failure modes.

### AI-assisted development
**🎯 Decision:** Apply when designing developer workflows that incorporate AI tools without fully delegating engineering judgment. An agent invokes this to define the appropriate scope of AI contribution—generation, suggestion, boilerplate—versus human responsibility: architecture, validation, security review.
**🎭 Analogy:** Power steering in a car: it reduces effort but doesn't replace the driver's judgment about where to go. The danger is treating it as autonomous navigation.
**💡 Insight:** AI-assisted development shifts the bottleneck from writing code to reviewing code. But review capacity is harder to scale than generation capacity, meaning AI tools may increase output volume faster than teams can safely absorb it—creating a quality debt that accumulates invisibly.

### Nondeterminism in AI
**🎯 Decision:** Apply when debugging AI-generated code or assessing reproducibility of a development process. An agent invokes this when the same prompt produces different outputs across runs, making it impossible to isolate whether a change in behavior reflects a code change or a sampling artifact.
**🎭 Analogy:** A photocopier that introduces random edits to each copy. You can't tell whether two documents differ because someone edited the original or because the copier introduced noise.
**💡 Insight:** Nondeterminism breaks a fundamental assumption of software engineering: that the same process applied to the same input yields the same output. AI-generated code introduces nondeterminism at the authorship layer, making version control semantics ambiguous and regression testing structurally incomplete.

### Software testing
**🎯 Decision:** Apply to verify that code behaves correctly across the range of inputs it will encounter. An agent invokes this as a mandatory layer after AI code generation, recognizing that AI outputs require test coverage calibrated to AI-specific failure modes, not just standard coverage metrics.
**🎭 Analogy:** A food taster for a chef who sometimes improvises ingredients. Standard tasting protocols detect known problems; you also need tests for unknown substitutions the chef may have made without noting them.
**💡 Insight:** Coverage metrics designed for human code—line coverage, branch coverage—are insufficient for AI-generated code because AI failures are semantic, not structural. A function can have 100% branch coverage and still implement the wrong algorithm. AI code demands more oracle-based and property-based testing.

### Software bugs
**🎯 Decision:** Apply when quantifying defect rate and defect type in AI-generated codebases. An agent invokes this when comparing AI-assisted versus human-written code quality, or when triaging a codebase to understand whether bugs cluster in AI-generated sections.
**🎭 Analogy:** Termites versus carpenter damage: both weaken a structure, but termites are hidden, spread systematically, and are only visible after significant structural compromise. AI-generated bugs share this profile.
**💡 Insight:** AI-generated bugs differ qualitatively from human bugs. Human bugs often reflect misunderstood requirements; AI bugs often reflect statistically common but contextually wrong patterns. This means AI bugs may be harder to find via code review because they look like reasonable code written by a competent developer.

### Software engineering expertise
**🎯 Decision:** Apply when assessing whether a team has the capacity to safely use, validate, and maintain AI-generated code. An agent invokes this when the gap between generation speed and expert review capacity threatens code quality—especially as senior engineers exit the field.
**🎭 Analogy:** A hospital that adopts diagnostic AI but loses its senior physicians. The AI may be useful, but without experts who can override it when it's wrong, the system becomes dangerous precisely when it fails.
**💡 Insight:** Stroustrup's observation about senior developers retiring rather than adapting signals an expertise hollowing-out risk. AI tools may lower the floor for entry-level developers while simultaneously driving away the senior engineers needed to catch AI errors—creating a systemic quality regression dressed as a productivity gain.

## Concept Reference
| Concept | Technical | Plain | Importance |
|---------|-----------|-------|------------|
| AI-generated code | Source code produced by large language models or generative AI systems through n | Code written by an AI instead of a human programmer, based on a description of w | 95% |
| Code validation | The systematic process of verifying that software meets specified requirements,  | Checking that code actually does what it's supposed to do and nothing harmful. | 90% |
| Software bugs | Defects or errors in source code that cause a program to produce incorrect, unex | Mistakes in code that make software behave wrongly or crash. | 85% |
| Code bloat | The presence of unnecessary, redundant, or inefficient code that increases progr | Extra unnecessary code that makes software bigger and slower than it needs to be | 78% |
| Security vulnerabilities | Weaknesses or flaws in software that can be exploited by malicious actors to gai | Holes in code that attackers can use to break into or misuse software. | 92% |
| Prompt sensitivity | The property of generative AI systems whereby small or subtle changes in input p | When a tiny change in how you ask an AI something produces a completely differen | 88% |
| Codebase | The complete collection of source code files, modules, and dependencies that con | All the code files that together make up a software application. | 80% |
| Unpredictable behavior | System outputs or states that cannot be reliably anticipated from inputs or prio | When software does things you didn't expect and can't easily predict. | 87% |
| Large language models (LLMs) | Neural network architectures trained on large text corpora using self-supervised | Powerful AI systems trained on vast text that can generate human-like writing an | 90% |
| Software engineering expertise | The accumulated domain knowledge, judgment, and skill required to design, implem | The deep skill and experience needed to build reliable, high-quality software. | 85% |
| Developer retirement | The departure of experienced software engineers from the workforce, either throu | Senior programmers leaving the field, taking years of hard-won knowledge with th | 75% |
| Bjarne Stroustrup | The Danish computer scientist who designed and created the C++ programming langu | The creator of C++, one of the most widely used programming languages in the wor | 82% |
| Code correctness | The property of a program that guarantees it satisfies its specification for all | Whether code actually does exactly what it was designed to do, without errors. | 88% |
| Technical debt | The implied cost of future rework accumulated when developers choose expedient s | The hidden future cost of taking shortcuts in software development today. | 76% |
| Generative AI | A class of machine learning models capable of producing novel content—text, code | AI that creates new content like text or code rather than just analyzing existin | 89% |
| Software reliability | The probability that a software system performs its intended function without fa | How consistently and dependably software works without breaking or failing. | 84% |
| Prompt engineering | The practice of designing and refining natural language inputs to guide generati | Crafting the right wording when asking an AI to get the output you actually want | 79% |
| Human code review | The manual inspection of source code by developers to detect defects, ensure adh | Having a programmer read through code to catch mistakes before it goes live. | 83% |
| Systems programming | Low-level software development targeting hardware interfaces, operating systems, | Writing software that works very close to hardware, like operating systems or de | 72% |
| Nondeterminism in AI | The characteristic of neural network inference whereby identical inputs can prod | The tendency of AI to give different answers even when asked the exact same ques | 86% |
| Software maintainability | The ease with which a software system can be modified, extended, debugged, or un | How easy it is for programmers to update, fix, and understand code over time. | 81% |
| Institutional knowledge | Accumulated organizational understanding of systems, decisions, and processes th | The unwritten know-how that experienced developers carry in their heads about ho | 77% |
| AI hallucination in code | The generation of syntactically plausible but semantically incorrect, nonexisten | When AI confidently writes code that looks right but is actually wrong or made u | 91% |
| Software security | The discipline of designing, implementing, and auditing software to protect agai | Protecting software from hackers and other threats by building it carefully and  | 93% |
| Cascading code changes | A modification in one part of a codebase that triggers unintended or necessary c | When changing one part of a program unexpectedly breaks or alters many other par | 85% |
| Developer productivity | The rate and quality at which software engineers produce working, maintainable c | How much useful, quality work a programmer can get done in a given time. | 78% |
| Software complexity | A measure of the structural and cognitive difficulty in understanding a software | How hard a piece of software is to understand, often because it has many interac | 84% |
| Workforce skill attrition | The gradual erosion of specialized competencies within an industry or organizati | The slow loss of important skills from a field when experienced people leave and | 74% |
| AI-assisted development | A software development paradigm integrating LLM-based tools to augment human pro | Using AI tools to help programmers write, fix, and improve code faster. | 87% |
| Static analysis | Automated examination of source code without execution to detect bugs, security  | Software tools that scan code for problems without actually running the program. | 75% |
| Context window limitations | The finite token capacity of LLMs that constrains how much prior code, instructi | The limit on how much information an AI can 'remember' when writing code in one  | 82% |
| Code ownership | The accountability and responsibility model determining which developer or team  | Knowing which programmer is responsible for a particular piece of code and its q | 73% |
| Training data bias | Systematic skews in LLM outputs attributable to imbalances, errors, or unreprese | AI learning bad habits or blind spots from flawed or unbalanced examples it was  | 80% |
| Software testing | The structured execution of software under controlled conditions to evaluate cor | Running a program in various ways to check that it works correctly and handles e | 86% |
| Abstraction layers | Hierarchical decomposition of software into levels of detail, where higher layer | Ways of hiding complexity in software so programmers only need to think about on | 76% |
| Critical infrastructure software | Software systems whose failure or compromise would have severe consequences for  | Code running power grids, hospitals, banking systems, and other essential servic | 88% |
| Software craftsmanship | A professional philosophy emphasizing skill, discipline, and pride in producing  | The idea that writing good code is a skilled craft requiring care, practice, and | 77% |
| Dependency management | The process of identifying, versioning, and maintaining external libraries and m | Keeping track of all the outside code libraries your software depends on and kee | 72% |
| Cognitive load in programming | The mental effort required for a developer to hold, process, and reason about co | How much mental effort it takes for a programmer to understand and work with a p | 79% |
| Software architecture | The high-level structural organization of a software system, defining components | The overall blueprint of how a software system is organized and how its parts fi | 81% |
| Human oversight of AI systems | The practice of maintaining meaningful human review, intervention capability, an | Keeping humans in the loop to check, correct, and take responsibility for what A | 94% |

## Glossary
| Term | Definition | Concept IDs |
|------|------------|-------------|
| AI-generated code | Source code produced by large language models or generative AI systems in response to natural langua | 1, 29 |
| Code validation | The process of verifying that code behaves as intended across all expected and edge-case inputs, mad | 2, 13 |
| Software bugs | Defects in code logic or implementation that cause incorrect or unintended program behavior, produce | 3, 16 |
| Code bloat | The accumulation of unnecessary, redundant, or inefficient code that increases binary size, reduces  | 4, 21 |
| Security vulnerabilities | Flaws in code that can be exploited to compromise confidentiality, integrity, or availability of a s | 5, 24 |
| Prompt sensitivity | The property of LLMs whereby small or seemingly insignificant changes in input phrasing produce subs | 6, 20 |
| Codebase | The complete body of source code comprising a software project, whose coherence and integrity are th | 7, 27 |
| Unpredictable behavior | The tendency of AI-generated code to produce outputs that deviate unexpectedly from intended semanti | 8, 20 |
| Large language models (LLMs) | Transformer-based neural networks trained on vast text corpora that generate code by predicting stat | 9, 15 |
| Software engineering expertise | The accumulated technical skill, design judgment, and domain knowledge required to build reliable, s | 10, 37 |
| Developer retirement | The departure of senior engineers from the profession, accelerated by frustration with AI-assisted w | 11, 28 |
| Bjarne Stroustrup | Creator of C++ and prominent systems programming authority who has publicly cautioned that AI-genera | 12, 19 |
| Code correctness | The property of code that guarantees it produces the specified output for all valid inputs, a standa | 13, 2 |
| Technical debt | The implied cost of rework accumulated when expedient or low-quality code is chosen over well-engine | 14, 21 |
| Generative AI | A class of AI systems capable of producing novel content including code, text, and images by learnin | 15, 9 |
| Software reliability | The probability that a software system performs its required functions under stated conditions for a | 16, 3 |
| Prompt engineering | The practice of carefully crafting input prompts to guide AI model outputs toward desired results, a | 17, 6 |
| Human code review | The manual inspection of source code by qualified engineers to detect defects, security issues, and  | 18, 41 |
| Systems programming | Low-level software development targeting operating systems, embedded systems, and infrastructure whe | 19, 36 |
| Nondeterminism in AI | The characteristic of generative models that produces variable outputs for identical or similar inpu | 20, 8 |
| Software maintainability | The ease with which a software system can be understood, modified, and extended over time, severely  | 21, 14 |
| Institutional knowledge | The accumulated organizational understanding of system history, design decisions, and operational co | 22, 11 |
| AI hallucination in code | The generation by LLMs of syntactically plausible but functionally incorrect or entirely fabricated  | 23, 1 |
| Software security | The discipline of designing, implementing, and verifying code to resist exploitation, undermined whe | 24, 5 |
| Cascading code changes | A chain reaction of required modifications that propagates through a codebase when a foundational co | 25, 7 |
| Developer productivity | The rate at which engineers deliver functional, high-quality software, which AI tools nominally incr | 26, 29 |
| Software complexity | The degree to which a system's structure and behavior are difficult to understand or predict, exacer | 27, 35 |
| Workforce skill attrition | The gradual erosion of engineering competency within an organization or industry as experienced prac | 28, 10 |
| AI-assisted development | A software engineering workflow in which developers use AI tools to generate, complete, or refactor  | 29, 41 |
| Static analysis | Automated examination of source code without execution to detect potential bugs, security flaws, and | 30, 2 |
| Context window limitations | The bounded amount of text an LLM can process in a single inference pass, preventing it from reasoni | 31, 27 |
| Code ownership | The assignment of responsibility for a specific module or system to an individual or team, eroded in | 32, 18 |
| Training data bias | Systematic skews in an LLM's behavior resulting from imbalances, errors, or outdated patterns in the | 33, 23 |
| Software testing | The systematic execution of code under controlled conditions to verify correctness and detect defect | 34, 13 |
| Abstraction layers | Architectural boundaries that hide implementation details behind defined interfaces, which AI system | 35, 40 |
| Critical infrastructure software | Code that controls essential societal systems such as power grids, financial networks, and healthcar | 36, 24 |
| Human oversight of AI systems | The ongoing supervision, review, and correction of AI behavior and output by qualified humans, ident | 41, 18 |

## Edge Cases & Warnings
- ⚠️ No engagement with Stroustrup's specific C++ expertise context, which would sharpen why correctness and determinism matter especially in systems programming
- ⚠️ Missing any counterargument or nuance Stroustrup may have offered, risking one-sided amplification
- ⚠️ The claim about static analysis tools being insufficient for AI failure modes is asserted but not substantiated with mechanism
- ⚠️ No distinction drawn between AI-assisted and AI-generated code, a meaningful technical boundary Stroustrup likely respects
- ⚠️ Workforce attrition claim attributed to Stroustrup but may be the pipeline's extrapolation from a brief quote
- ⚠️ Critical infrastructure risk is asserted categorically without acknowledging domains where AI code assistance has demonstrated adequate reliability

## Emergence Assessment
The pipeline demonstrated strong emergent synthesis, moving well beyond the three source bullets to construct a coherent systems-level argument. It connected nondeterminism to auditability failure, linked workforce attrition to institutional knowledge loss, and framed prompt sensitivity as a structural reproducibility problem rather than a mere usability issue. The thesis introduced second-order concepts not explicit in the source, such as context window limitations, training data bias, hallucinated APIs, and dependency conflicts, which represent genuine inferential extension. The accountability erosion framing around code ownership is a meaningful emergent concept. However, some elaborations read as field-standard LLM criticism that may not trace directly to Stroustrup's specific positions, introducing attribution ambiguity between what Stroustrup argued and what the pipeline inferred from general discourse.

## Recommendations
- 🔧 Anchor each elaborated claim back to a traceable source fragment or flag it explicitly as pipeline inference to preserve attribution integrity
- 🔧 Introduce at least one steelmanned counterposition to avoid advocacy drift masquerading as analysis
- 🔧 Specify the mechanism by which static analysis fails on AI output rather than asserting insufficiency
- 🔧 Distinguish Stroustrup's documented positions from plausible extensions inferred from his broader published views
- 🔧 Add a confidence-weighted tier to the taxonomy terms indicating which concepts are source-direct versus pipeline-derived

## Quick Reference
```python
from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class RiskTier(Enum):
    """Classifies the consequence domain of the software under evaluation."""
    CONSUMER = "consumer"                  # crashes annoy users
    BUSINESS_CRITICAL = "business_critical"  # outages cost money or data
    CRITICAL_INFRASTRUCTURE = "critical_infrastructure"  # failures harm people


class Disposition(Enum):
    """Final risk disposition after assessment."""
    ACCEPT_STANDARD = "accept_standard_review"
    ACCEPT_AUGMENTED = "accept_augmented_validation"
    REJECT_PENDING_REWRITE = "reject_pending_human_rewrite"


@dataclass
class CodeSurface:
    """
    Represents a single unit of AI-contributed code under evaluation.
    A 'surface' is any function, module, or dependency where LLM output
    was used directly or lightly edited.
    """
    name: str
    handles_untrusted_input: bool = False
    handles_authentication: bool = False
    handles_memory: bool = False
    handles_network: bool = False
    prompt_version_controlled: bool = False   # is the generating prompt tracked?
    has_institutional_owner: bool = True       # does a senior engineer still own this?
    happy_path_only_tested: bool = False       # test suite covers only nominal cases?
    uses_external_apis: bool = False           # risk of hallucinated API contracts


@dataclass
class AssessmentResult:
    """Structured output of a single surface assessment."""
    surface_name: str
    risk_score: float          # 0.0 (low) – 1.0 (critical)
    flags: list[str] = field(default_factory=list)
    disposition: Optional[Disposition] = None
    rationale: str = ""


def score_surface(surface: CodeSurface, tier: RiskTier) -> AssessmentResult:
    """
    Evaluate a single AI-contributed code surface and return a risk score
    with an explicit disposition recommendation.

    Scoring logic encodes the key patterns from the skill:
      - Plausibility-Correctness Gap (happy path testing masks AI failure modes)
      - Training Corpus Security Debt (untrusted input + auth paths)
      - Oversight Degradation (missing institutional owner + no prompt versioning)
      - Prompt History as Hidden Dependency (prompt not version-controlled)
    """
    score: float = 0.0
    flags: list[str] = []

    # --- Security-sensitive code paths (Training Corpus Security Debt) ---
    if surface.handles_untrusted_input:
        score += 0.25
        flags.append(
            "Handles untrusted input: AI models statistically reinforce "
            "pre-remediation vulnerability patterns from training corpus."
        )
    if surface.handles_authentication:
        score += 0.25
        flags.append(
            "Authentication path: hallucinated invariants here are "
            "categorically dangerous; requires adversarial testing."
        )
    if surface.handles_memory:
        score += 0.20
        flags.append(
            "Memory management: LLMs reproduce patterns from unsafe historical "
            "code; buffer and use-after-free risks elevated."
        )
    if surface.handles_network:
        score += 0.15
        flags.append(
            "Network-facing: injection and deserialization attack surfaces "
            "are underrepresented in AI happy-path generation."
        )

    # --- Plausibility-Correctness Gap (test suite calibration) ---
    if surface.happy_path_only_tested:
        score += 0.20
        flags.append(
            "Test suite covers nominal cases only: AI bugs cluster at edge cases "
            "and adversarial inputs, which existing suites likely miss."
        )

    # --- Prompt History as Hidden Dependency ---
    if not surface.prompt_version_controlled:
        score += 0.10
        flags.append(
            "Prompt not version-controlled: codebase carries an invisible "
            "dependency; reproducibility and auditability are compromised."
        )

    # --- Hallucinated API contracts ---
    if surface.uses_external_apis:
        score += 0.15
        flags.append(
            "External API usage: risk of hallucinated contracts—syntactically "
            "valid calls to nonexistent or misused library interfaces."
        )

    # --- Oversight Degradation Under Scale ---
    if not surface.has_institutional_owner:
        score += 0.15
        flags.append(
            "No institutional owner: senior engineer knowledge gap means "
            "tacit correctness requirements are not held by anyone on the team."
        )

    # --- Tier multiplier: critical infrastructure amplifies every defect ---
    tier_multiplier = {
        RiskTier.CONSUMER: 1.0,
        RiskTier.BUSINESS_CRITICAL: 1.3,
        RiskTier.CRITICAL_INFRASTRUCTURE: 1.7,
    }[tier]

    final_score = min(score * tier_multiplier, 1.0)

    # --- Disposition mapping ---
    if final_score < 0.35:
        disposition = Disposition.ACCEPT_STANDARD
        rationale = "Risk level is manageable with standard peer review."
    elif final_score < 0.65:
        disposition = Disposition.ACCEPT_AUGMENTED
        rationale = (
            "Elevated risk requires augmented validation: adversarial testing, "
            "expert audit of flagged paths, or formal verification where feasible."
        )
    else:
        disposition = Disposition.REJECT_PENDING_REWRITE
        rationale = (
            "Risk score exceeds acceptable threshold for this tier. "
            "Human rewrite of flagged components is required before deployment."
        )

    return AssessmentResult(
        surface_name=surface.name,
        risk_score=round(final_score, 3),
        flags=flags,
        disposition=disposition,
        rationale=rationale,
    )


def assess_codebase(
    surfaces: list[CodeSurface],
    tier: RiskTier,
) -> dict:
    """
    Run a full codebase assessment across all AI-contributed surfaces.
    Returns a structured report with per-surface results and an aggregate
    disposition for the overall merge/deployment decision.
    """
    results = [score_surface(s, tier) for s in surfaces]

    # Aggregate: the codebase disposition is driven by the highest-risk surface.
    max_score = max(r.risk_score for r in results) if results else 0.0
    reject_count = sum(
        1 for r in results if r.disposition == Disposition.REJECT_PENDING_REWRITE
    )

    if reject_count > 0:
        aggregate_disposition = Disposition.REJECT_PENDING_REWRITE
    elif max_score >= 0.35:
        aggregate_disposition = Disposition.ACCEPT_AUGMENTED
    else:
        aggregate_disposition = Disposition.ACCEPT_STANDARD

    return {
        "tier": tier.value,
        "surface_count": len(surfaces),
        "aggregate_risk_score": round(max_score, 3),
        "aggregate_disposition": aggregate_disposition.value,
        "surfaces": [
            {
                "name": r.surface_name,
                "risk_score": r.risk_score,
                "disposition": r.disposition.value if r.disposition else None,
                "rationale": r.rationale,
                "flags": r.flags,
            }
            for r in results
        ],
    }


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    surfaces = [
        CodeSurface(
            name="auth_token_validator",
            handles_authentication=True,
            handles_untrusted_input=True,
            prompt_version_controlled=False,
            has_institutional_owner=False,
            happy_path_only_tested=True,
        ),
        CodeSurface(
            name="report_pdf_renderer",
            handles_untrusted_input=False,
            prompt_version_controlled=True,
            has_institutional_owner=True,
            happy_path_only_tested=False,
        ),
        CodeSurface(
            name="payment_gateway_client",
            handles_network=True,
            uses_external_apis=True,
            handles_untrusted_input=True,
            prompt_version_controlled=False,
            has_institutional_owner=True,
            happy_path_only_tested=True,
        ),
    ]

    report = assess_codebase(surfaces, RiskTier.BUSINESS_CRITICAL)
    print(json.dumps(report, indent=2))
```

---
_Generated by Philosopher's Stone v4 — EchoSeed_
