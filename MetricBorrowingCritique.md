# Metric Borrowing Critique

> Activate when institutional systems import AI evaluation frameworks—throughput scores, accuracy benchmarks, latency thresholds—to assess human beings, triggering ontological mismatch between computational measurement categories and persons whose value, dignity, and agency are irreducible to performance proxies.

## Core Thesis
When institutional systems begin assessing humans using output metrics borrowed from AI evaluation frameworks, a fundamental ontological mismatch occurs: categories designed to measure computational throughput are applied to beings whose value is irreducible to performance proxies, committing a categorical error that enacts epistemic violence against human dignity. This metric borrowing enacts a normative transfer whereby the instrumental rationality embedded in algorithmic governance migrates into human contexts, carrying with it quantification bias and efficiency fetishism that systematically devalue what cannot be measured. Goodhart's Law operates with particular force here, as the moment human worth is indexed to a proxy measure, individuals optimize for the metric rather than the underlying quality, producing behavioral homogenization and hollowing out genuine capability. The process constitutes a form of technocratic isomorphism in which human institutions reshape themselves structurally to resemble computational systems, not because such resemblance improves outcomes but because it confers institutional legitimacy within neoliberal managerial cultures. Datafication of human performance is never neutral: it encodes the values of surveillance capitalism and labor commodification into the fabric of organizational life, making exploitation appear as objective measurement. Agentic reduction follows inevitably, as persons are collapsed into their measurable outputs, stripping away the complexity, relationality, and self-determination that constitute genuine human agency and producing a slow, systemic dehumanization. Framework colonization proceeds quietly, with AI benchmark validity standards—precision-recall tradeoffs, latency thresholds, scalability targets—imported wholesale into performance reviews, credentialing systems, and welfare assessments designed for fundamentally different purposes. Automation bias amplifies the damage, as evaluators defer to algorithmic accountability metrics even when those metrics contradict richer, contextual human judgment, embedding panoptic surveillance into routine institutional life. Commensurability is falsely assumed: the move to make human qualities comparable and rankable destroys the very distinctions that make those qualities meaningful, a reductionism that severs people from the social and historical contexts that give their actions significance. Alienation deepens as workers and students internalize the machine-legible self-image imposed on them, experiencing their own capacities as external, quantified properties owned and adjudicated by institutional systems rather than as expressions of selfhood. Resisting this trajectory requires construct validity scrutiny, epistemic humility about what metrics can and cannot capture, robust value alignment processes that center human dignity, and a willingness to defend the irreducible difference between evaluating a system and honoring a person.

## Overview

This skill equips an agent to recognize, diagnose, and resist the migration of AI evaluation logic into human institutional contexts. It operates at the intersection of philosophy of mind, organizational theory, critical data studies, and ethics, providing conceptual vocabulary and analytical procedures for identifying when measurement frameworks have been borrowed from the wrong domain. The skill is not anti-measurement: it distinguishes between metrics designed with awareness of human ontology and metrics transplanted wholesale from systems whose entire architecture assumes a fundamentally different kind of entity.

The skill matters because the borrowing process is structurally invisible. When an institution adopts precision-recall language for employee reviews, or applies throughput benchmarks to student credentialing, the numbers arrive looking authoritative. Goodhart's Law then accelerates the damage quietly: individuals reshape themselves around the proxy, institutions read that reshaping as success, and the original human qualities the metrics were meant to track dissolve. By the time the dehumanization is legible, it has been normalized into policy. This skill creates the early-detection capacity that prevents that normalization.

Reach for it whenever you encounter evaluation language that feels borrowed—when performance reviews speak of "response latency," when welfare assessments produce "risk scores," when educational systems rank students by "output velocity," or when any institution begins treating human beings as systems to be benchmarked rather than persons to be understood. The skill provides both a diagnostic for identifying the mismatch and a constructive alternative: assessment traditions grounded in construct validity, relational context, and the categorical protection of human dignity.

## When to Use
- An institution proposes adopting AI benchmark criteria (accuracy rates, throughput scores, precision-recall tradeoffs) for evaluating human employees, students, or welfare recipients
- A performance review, credentialing process, or welfare assessment reduces persons to ranked numerical outputs with no contextual, relational, or developmental dimension
- Evaluators exhibit automation bias, deferring to algorithmic scores even when those scores contradict richer human judgment available in the same context
- A policy debate frames human worth, productivity, or potential as commensurable and rankable across individuals using a single metric axis
- An organization's assessment infrastructure is being redesigned and there is opportunity to scrutinize what is being imported from computational governance models
- A person or community reports feeling "erased" or "reduced" by an institutional evaluation process, suggesting that dehumanization through datafication may already be underway

## Core Workflow
1. **Identify the metric's origin domain** — trace the evaluation criteria back to their source and ask whether they were designed for entities with intentions, relationships, development trajectories, and irreducible dignity, or for systems defined solely by inputs, outputs, and parameters
2. **Run an ontological mismatch audit** — systematically compare what the metric assumes about the entity being evaluated against what is actually true of the persons being assessed, cataloguing everything the metric cannot see: suffering, aspiration, care, growth, context, moral complexity
3. **Apply Goodhart's Law projection** — model what behavioral changes will occur once individuals learn they are being assessed by this metric, and ask whether those changes improve or hollow out the underlying quality the metric was meant to track
4. **Assess construct validity** — determine whether the metric has been validated for the human context in which it is being used, not merely for the AI evaluation context from which it was borrowed, demanding evidence that the construct measured corresponds to the human quality claimed
5. **Identify displaced traditions** — name the prior human assessment frameworks (pedagogical, jurisprudential, clinical, ethical) that the borrowed metric is overwriting, and evaluate whether those traditions carried knowledge that is now being discarded
6. **Propose dignity-centered alternatives** — articulate assessment approaches that preserve relational context, developmental time, qualitative judgment, and categorical protection of human worth, distinguishing them clearly from the borrowed framework
7. **Document the normative transfer** — make explicit what values are smuggled in with the metric: efficiency fetishism, productivity theology, surveillance capitalism assumptions, labor commodification logic, so that institutional decision-makers can evaluate those values deliberately rather than inheriting them invisibly

## Key Patterns

### The Invisible Smuggling Pattern
Borrowed metrics do not arrive as ideology; they arrive as spreadsheets. The values embedded in AI evaluation frameworks — that correct answers exist, that throughput is the primary good, that non-productive states are waste — are invisible precisely because they are encoded in the structure of the measurement instrument rather than stated as claims. An agent must learn to read the implicit ontology of a metric the way a careful reader reads the implicit assumptions of a text: not what it says, but what it takes for granted in order to say anything at all.

```python
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class MetricOntologyAudit:
    """
    Audits a borrowed metric for hidden ontological assumptions
    before it is applied to human assessment contexts.
    """
    metric_name: str
    origin_domain: str  # e.g. "LLM evaluation", "search ranking"
    proposed_human_context: str  # e.g. "employee performance review"

    # Assumptions the metric requires to be valid
    assumes_ground_truth_exists: bool = False
    assumes_entity_is_output_producing: bool = False
    assumes_states_are_comparable: bool = False
    assumes_non_productive_states_are_waste: bool = False

    # What the metric cannot see
    invisible_to_metric: list[str] = field(default_factory=list)

    def ontological_mismatch_score(self) -> int:
        """
        Returns count of active mismatch flags.
        Higher score = greater risk of categorical error when applied to humans.
        """
        flags = [
            self.assumes_ground_truth_exists,
            self.assumes_entity_is_output_producing,
            self.assumes_states_are_comparable,
            self.assumes_non_productive_states_are_waste,
        ]
        return sum(flags)

    def report(self) -> str:
        score = self.ontological_mismatch_score()
        invisible = ", ".join(self.invisible_to_metric) or "none catalogued"
        return (
            f"Metric: {self.metric_name}\n"
            f"Origin: {self.origin_domain} → Proposed use: {self.proposed_human_context}\n"
            f"Ontological mismatch flags: {score}/4\n"
            f"What this metric cannot see: {invisible}\n"
            f"Recommendation: {'HIGH RISK — do not import without construct revalidation' if score >= 2 else 'Review with caution'}"
        )


# Example usage: auditing a throughput metric proposed for student assessment
audit = MetricOntologyAudit(
    metric_name="tokens_per_second",
    origin_domain="LLM inference benchmarking",
    proposed_human_context="student writing assessment",
    assumes_ground_truth_exists=True,
    assumes_entity_is_output_producing=True,
    assumes_states_are_comparable=True,
    assumes_non_productive_states_are_waste=True,
    invisible_to_metric=[
        "revision and reflection time",
        "emotional state during writing",
        "learning trajectory",
        "meaning and argument quality",
        "developmental context",
    ],
)

print(audit.report())
```

### The Goodhart Cascade Pattern
Goodhart's Law in human-metric contexts is not a single event but a cascade. First, individuals discover the metric and begin optimizing for it. Second, the metric score rises, and institutions read this as improvement. Third, the underlying quality — the thing the metric was meant to track — degrades because optimization effort has been redirected. Fourth, the metric is reinforced because it is producing "results." Understanding this as a temporal cascade rather than a static distortion allows an agent to intervene at the earliest stage, before the cascade self-reinforces.

```python
from enum import Enum

class CascadeStage(Enum):
    METRIC_INTRODUCED = 1      # Borrowed metric enters institutional use
    INDIVIDUAL_OPTIMIZATION = 2  # People learn to game the metric
    METRIC_SCORE_RISES = 3     # Institution reads gaming as success
    UNDERLYING_QUALITY_DEGRADES = 4  # Real capability hollows out
    METRIC_REINFORCED = 5      # Institution doubles down; cycle locks in

@dataclass
class GoodhartCascadeTracker:
    """
    Tracks progression of Goodhart's Law cascade
    when a borrowed metric enters a human assessment context.
    """
    metric_name: str
    underlying_quality: str  # What the metric was meant to track
    current_stage: CascadeStage = CascadeStage.METRIC_INTRODUCED
    observable_signals: dict[CascadeStage, list[str]] = field(default_factory=dict)

    def advance_stage(self, evidence: str) -> str:
        """
        Advance the cascade tracker when new evidence is observed.
        Returns a warning with intervention recommendation.
        """
        if self.current_stage.value < len(CascadeStage):
            self.current_stage = CascadeStage(self.current_stage.value + 1)
            if self.current_stage not in self.observable_signals:
                self.observable_signals[self.current_stage] = []
            self.observable_signals[self.current_stage].append(evidence)

        interventions = {
            CascadeStage.INDIVIDUAL_OPTIMIZATION: "Communicate that gaming will be detected and that the underlying quality matters more than the score.",
            CascadeStage.METRIC_SCORE_RISES: "Audit whether score rise correlates with actual quality improvement using independent qualitative assessment.",
            CascadeStage.UNDERLYING_QUALITY_DEGRADES: "Suspend the metric immediately; convene construct validity review.",
            CascadeStage.METRIC_REINFORCED: "Escalate: institutional lock-in is occurring. Require external review before metric continues.",
        }
        rec = interventions.get(self.current_stage, "Monitor.")
        return f"[Stage {self.current_stage.value}: {self.current_stage.name}] Evidence: {evidence}\nIntervention: {rec}"


# Example: tracking a cascade after throughput metric enters a care worker assessment
tracker = GoodhartCascadeTracker(
    metric_name="cases_closed_per_week",
    underlying_quality="quality of care and client wellbeing",
)
print(tracker.advance_stage("Care workers begin closing cases prematurely to hit weekly targets."))
print(tracker.advance_stage("Closure rate rises 40%; management reports productivity improvement."))
```

### The Displaced Tradition Pattern
Every human assessment domain — pedagogy, clinical evaluation, jurisprudence, performance review — evolved alongside ethical and relational frameworks precisely because evaluating persons was understood to be morally fraught. When AI metrics colonize these domains, they displace not just different numbers but accumulated institutional wisdom about what it means to assess a person well. Making that displacement explicit — naming what is being lost — is one of the most effective interventions available, because it converts an invisible substitution into a legible choice.

```python
@dataclass
class DisplacedTraditionRecord:
    """
    Documents what prior human assessment tradition is overwritten
    when an AI-derived metric is imported into a human context.
    """
    domain: str  # e.g. "clinical social work", "higher education"
    prior_tradition: str  # e.g. "holistic case formulation"
    imported_metric: str  # e.g. "case resolution latency score"

    # Knowledge carried by the prior tradition that the metric cannot encode
    lost_knowledge: list[str] = field(default_factory=list)

    # Ethical commitments embedded in the prior tradition
    displaced_ethical_commitments: list[str] = field(default_factory=list)

    def displacement_summary(self) -> str:
        lost = "\n  - ".join(self.lost_knowledge) or "not yet catalogued"
        ethics = "\n  - ".join(self.displaced_ethical_commitments) or "not yet catalogued"
        return (
            f"Domain: {self.domain}\n"
            f"Prior tradition displaced: {self.prior_tradition}\n"
            f"Replacing metric: {self.imported_metric}\n"
            f"Knowledge lost:\n  - {lost}\n"
            f"Ethical commitments displaced:\n  - {ethics}"
        )


record = DisplacedTraditionRecord(
    domain="secondary education",
    prior_tradition="portfolio-based developmental assessment with teacher narrative",
    imported_metric="standardized output score percentile rank",
    lost_knowledge=[
        "student's growth trajectory over time",
        "relationship between effort and outcome given starting conditions",
        "creative and relational capacities not captured by test formats",
        "contextual factors shaping performance on any given day",
    ],
    displaced_ethical_commitments=[
        "the student is a developing person, not a finished product",
        "assessment should serve learning, not sort for labor market allocation",
        "the evaluator has a duty of care toward the evaluated",
    ],
)

print(record.displacement_summary())
```

## Triple-Mode Insights

### Dehumanization
**🎯 Decision:** An agent applies this when systems strip away contextual, relational, and subjective dimensions of persons, reducing them to measurable outputs. Triggered when evaluation ignores suffering, aspiration
**🎭 Analogy:** A portrait reduced to a histogram of pixel values — technically derived from the original, yet the face, the gaze, the person, has vanished entirely.
**💡 Insight:** Dehumanization through metrics is more insidious than overt contempt because it masquerades as objectivity. The evaluator feels neutral while the evaluated feels erased, creating asymmetric violence w

### Human Dignity
**🎯 Decision:** Invoked when an agent must resist frameworks that treat persons as means rather than ends. Applied when evaluation systems ignore irreducible worth — moral status not contingent on productivity, effic
**🎭 Analogy:** A constitution that cannot be amended by majority vote — some values are placed deliberately beyond the reach of optimization, protected precisely bec
**💡 Insight:** Dignity functions as a categorical constraint, not a variable to be traded off. When institutions borrow AI metrics, they implicitly treat dignity as negotiable, revealing that the deeper violation is

### Metric Borrowing
**🎯 Decision:** Occurs when a domain lacking native measurement tools adopts metrics designed for a structurally different domain. Applied when institutions import AI evaluation criteria — throughput, accuracy, laten
**🎭 Analogy:** Using a thermometer to measure sound pressure — the instrument is real, the reading is real, but it measures nothing relevant to the phenomenon at han
**💡 Insight:** Borrowed metrics carry hidden assumptions about what counts as success. When AI output metrics migrate to humans, they silently import the assumption that humans, like models, exist solely to produce 

### Goodhart's Law
**🎯 Decision:** Applied when a proxy measure becomes the optimization target and ceases to track the underlying value it represented. Triggered whenever institutional incentives cause agents to maximize the metric ra
**🎭 Analogy:** A doctor who improves patient satisfaction scores by prescribing whatever patients request — the metric rises while health, the original goal, quietly
**💡 Insight:** When humans are assessed by AI-derived metrics, Goodhart's Law operates at two levels simultaneously: individuals game the new measures, and institutions mistake the gaming for success. The original h

### AI Evaluation Frameworks
**🎯 Decision:** Relevant when an agent must assess model capabilities, safety, or alignment using structured benchmarks. Designed for systems whose ontology is defined by inputs, outputs, and parameters — not intenti
**🎭 Analogy:** A standardized test designed for sorting machines — useful for comparing processors, catastrophic when applied to beings whose most important qualitie
**💡 Insight:** AI evaluation frameworks are built on the assumption that ground truth exists and can be checked. Applying them to humans smuggles in that assumption, implying human behavior has correct answers — a p

### Ontological Mismatch
**🎯 Decision:** Diagnosed when the categories of an evaluation system do not correspond to the kind of being being evaluated. Applied when the metadata of a framework — its variables, assumptions, scoring logic — pre
**🎭 Analogy:** Judging a river by traffic engineering standards — asking about lane capacity, stopping distance, signal compliance. The river keeps flowing; the repo
**💡 Insight:** Ontological mismatch is self-concealing: the framework generates numbers regardless, so institutions rarely notice the misfit. The damage accumulates in what gets excluded — moral complexity, relation

### Instrumental Rationality
**🎯 Decision:** Invoked when an agent selects the most efficient means to a given end without interrogating whether the end is appropriate. Applied when institutions optimize human assessment processes for efficiency
**🎭 Analogy:** A navigator who sails perfectly toward the wrong destination — the competence is real, the effort is real, the arrival is a failure.
**💡 Insight:** Instrumental rationality applied to human assessment produces technically sophisticated injustice. The tools work exactly as designed; the problem is that the design question — what is a human life fo

### Output Metrics
**🎯 Decision:** Applied when evaluation focuses on measurable products of a process — tokens generated, tasks completed, accuracy percentages — rather than on process quality, intent, or context. Useful for systems w
**🎭 Analogy:** Judging a conversation by word count — the measure is real, reproducible, and entirely misses whether anything meaningful was communicated or received
**💡 Insight:** Output metrics create invisible penalties for non-productive states: rest, reflection, care, grief, learning, play. When applied to humans, they encode a theology of productivity — the idea that worth

### Agentic Reduction
**🎯 Decision:** Applied when a complex, multidimensional entity is reduced to its role as an agent that produces actions or outputs. Triggered when institutional frameworks strip away the passive, receptive, relation
**🎭 Analogy:** Describing a musician only by notes played per minute — the music is acknowledged, the musician as listener, feeler, interpreter, meaning-maker, is er
**💡 Insight:** Agentic reduction is particularly seductive in institutional contexts because it aligns with legal and managerial categories that already treat persons as responsible actors. Borrowing AI metrics acce

### Framework Colonization
**🎯 Decision:** Diagnosed when a framework designed for one domain expands to govern another, displacing indigenous categories, values, and modes of understanding. Applied when AI evaluation logic overwrites prior hu
**🎭 Analogy:** A foreign legal code imposed on a community whose entire moral vocabulary — obligations, relationships, sacred prohibitions — has no translation in th
**💡 Insight:** Framework colonization is not always intentional. Institutions adopt AI metrics because they are available, precise-seeming, and prestigious. The displacement of richer human assessment traditions hap

### Human Assessment
**🎯 Decision:** Invoked when an agent must evaluate persons — their performance, development, potential, or conduct. Requires sensitivity to context, relationships, intentions, growth trajectories, and values that ca
**🎭 Analogy:** A wise mentor whose evaluation takes years — not because they are slow, but because the thing they are assessing, a developing person, unfolds in time
**💡 Insight:** Human assessment traditions evolved alongside ethics, pedagogy, and jurisprudence precisely because evaluating persons is morally fraught. Replacing these traditions with AI metrics discards centuries

### Datafication
**🎯 Decision:** Applied when qualitative, relational, or experiential phenomena are converted into data structures for analysis and storage. Triggered when human behaviors, relationships, or states are rendered as da
**🎭 Analogy:** Pressing a living flower into a book — preserved, portable, analyzable — but no longer blooming, no longer responsive, no longer alive in the relevant
**💡 Insight:** Datafication creates an archive that feels like understanding. Institutions mistake the accumulation of data points for knowledge of the person. But the conversion always loses something — the texture

## Edge Cases & Warnings

Some legitimate human assessment tools do borrow quantitative logic from adjacent scientific domains without committing the categorical errors described here. The distinction lies in construct validation: a validated psychological instrument that produces a numerical score is not automatically an instance of metric borrowing in the harmful sense, provided it was designed and normed for the human context in which it is used, and provided its designers acknowledged what it cannot measure. The skill should be applied with proportionality — the alarm is for unreflective importation of AI-specific ontological assumptions, not for quantification per se.

Be cautious about overcorrecting into pure anti-quantification. The argument here is not that numbers are always wrong in human contexts; it is that numbers borrowed from systems designed to assess computational throughput carry hidden assumptions that do not transfer. Institutions can and should measure what they can measure, provided they remain epistemically humble about what they cannot, and provided they treat the unmeasured dimensions as equally real and equally important.

Automation bias can operate in the analyst as well as the evaluator. When using this skill's diagnostic tools, be alert to the risk of treating the ontological mismatch audit score as itself a ground truth. The audit is a heuristic for directing attention, not a replacement for contextual judgment. A score of 2/4 on the mismatch audit does not automatically condemn a metric; it flags the metric for deeper scrutiny.

The skill may encounter institutional resistance framed as pragmatism: "we know the metrics are imperfect but they're the best we have." This framing is worth taking seriously rather than dismissing, because the alternative to imperfect measurement is not always richer assessment — sometimes it is no accountability at all, which can also harm persons. The response is to push for better measurement rather than no measurement, specifically measurement designed with human ontology in mind from the start.

Framework colonization is not always initiated by bad actors. Many institutions adopt AI metrics because they are under pressure to demonstrate accountability, face resource constraints that make rich qualitative assessment impractical, or operate in policy environments that reward quantifiable outcomes. Diagnosing the structural conditions that make metric borrowing attractive is part of a complete application of this skill, because structural diagnosis opens pathways for structural intervention.

## Emergence Assessment

The deepest emergent property of this skill is the recognition that metric borrowing is not primarily a technical error but a political one. The choice to assess humans using AI-derived frameworks is never merely a choice about measurement instruments; it is a choice about what kind of being a human is, what institutions owe to the people they evaluate, and whose interests the evaluation serves. By forcing the ontological audit — by asking what kind of entity the metric assumes it is measuring — the skill makes this political dimension visible, and visibility is the precondition for democratic deliberation about whether the choice should be made at all.

A second emergent insight is the

## Concept Reference
| Concept | Technical | Plain | Importance |
|---------|-----------|-------|------------|
| **Institutional Systems** | Formal organizational structures with codified rules, hierarchies, and processes | Organizations like governments, schools, or companies that have official rules a | 85% |
| **Output Metrics** | Quantifiable performance indicators used to measure and evaluate the productive  | Numbers or scores used to measure how much someone or something produces or acco | 92% |
| **AI Evaluation Frameworks** | Structured methodologies and benchmark suites used to assess artificial intellig | The tools and tests used to grade how well an AI system works at its assigned ta | 93% |
| **Human Assessment** | The systematic appraisal of individual cognitive, behavioral, or productive capa | How organizations judge and grade people's work, abilities, or contributions. | 91% |
| **Metric Borrowing** | The transfer of evaluative instruments from one domain to another without full e | Using measurement tools designed for one thing to judge something fundamentally  | 94% |
| **Algorithmic Governance** | The use of computational decision-making procedures to regulate, allocate resour | When automated or rule-based systems make decisions that control how people are  | 88% |
| **Quantification Bias** | A systematic epistemic distortion arising when measurable proxies displace holis | The tendency to overvalue things that can be counted and ignore important qualit | 90% |
| **Commensurability** | The property of two or more entities or systems being reducible to a common scal | Whether two very different things can fairly be measured and compared using the  | 87% |
| **Technocratic Isomorphism** | The institutional process whereby organizations adopt structurally similar pract | When organizations copy tech-world practices just to seem modern, even if those  | 89% |
| **Performance Proxy** | A measurable surrogate variable substituted for a target construct that resists  | A stand-in number used to represent something real but hard to measure directly, | 88% |
| **Ontological Mismatch** | A fundamental incompatibility between the categorical assumptions embedded in an | When the basic assumptions behind a measuring system simply don't match the real | 93% |
| **Throughput Optimization** | The maximization of task completion rate or output volume per unit time, a core  | Focusing on doing as many tasks as possible as fast as possible, a goal borrowed | 85% |
| **Human Dignity** | The intrinsic, non-instrumental worth attributed to persons, resisting reduction | The idea that people have inherent worth beyond what they produce or accomplish. | 95% |
| **Reductionism** | The epistemic strategy of explaining complex phenomena by decomposing them into  | Oversimplifying something complex by breaking it down so far that important qual | 87% |
| **Benchmark Validity** | The degree to which a standardized test accurately and completely represents the | Whether a test actually measures what it claims to measure in a meaningful way. | 89% |
| **Surveillance Capitalism** | An economic logic in which behavioral data is extracted and commodified as a pre | A business model where personal behavior data is harvested and sold to predict o | 86% |
| **Dehumanization** | The cognitive and institutional process of stripping persons of their attributed | Treating people as if they were objects or machines rather than complex human be | 96% |
| **Goodhart's Law** | The principle that once a measure becomes a target, it ceases to be a good measu | When people start gaming a scoring system, the score stops meaning what it was s | 94% |
| **Epistemic Violence** | The delegitimization or erasure of certain forms of knowledge, experience, or be | Harm caused by forcing one way of knowing onto people whose reality it doesn't r | 88% |
| **Neoliberal Managerialism** | An organizational ideology applying market-rationality principles—efficiency, co | Running organizations like businesses obsessed with efficiency and measurable re | 87% |
| **Labor Commodification** | The transformation of human labor power into an exchangeable commodity evaluated | Treating people's work and skills as just another product to be bought, sold, an | 89% |
| **Latency** | The delay between a stimulus and response in a computational system, used as a k | How long it takes a system to respond — a machine metric now sometimes applied t | 80% |
| **Construct Validity** | The extent to which a measurement instrument actually captures the theoretical c | Whether your measuring tool genuinely captures the real thing you care about. | 88% |
| **Datafication** | The transformation of social actions, behaviors, and human qualities into quanti | Converting human behavior and qualities into data that computers can track and a | 91% |
| **Agentic Reduction** | The theoretical collapse of the complex, intentional, and reflexive properties o | Shrinking the full complexity of human decision-making and freedom down to simpl | 92% |
| **Normative Transfer** | The migration of values and standards embedded in one domain's practices into an | When the values and rules from one world quietly get imported into another, chan | 90% |
| **Institutional Legitimacy** | The socially constructed perception that an organization's actions and structure | How much trust and authority people grant to an organization based on whether it | 84% |
| **Scalability** | The capacity of a system to handle increasing workloads by proportional resource | How well something grows to handle more demand — originally a tech concept now a | 82% |
| **Automation Bias** | The cognitive tendency to over-rely on automated decision-support systems, disco | Trusting machines or algorithms more than human judgment, even when the humans m | 87% |
| **Panopticism** | A mode of disciplinary power in which the possibility of constant observation in | The way constant monitoring — or just the threat of it — makes people police the | 86% |
| **Efficiency Fetishism** | An ideological overvaluation of productive efficiency as a terminal good, subord | Being so obsessed with efficiency that it crowds out other important values like | 89% |
| **Categorical Error** | A logical fallacy in which entities of fundamentally different kinds are treated | A mistake in thinking where two very different kinds of things are treated as if | 91% |
| **Sociotechnical System** | An integrated configuration of social structures, human actors, and technologica | The combined system of people, organizations, and technology that can't be under | 85% |
| **Accountability Metrics** | Quantified indicators deployed within governance frameworks to assign responsibi | Numbers used to track who is responsible for results within an organization or s | 83% |
| **Alienation** | The estrangement of persons from their productive activity, its products, fellow | Feeling disconnected from your work, its results, or your own humanity — often c | 88% |
| **Framework Colonization** | The process by which the conceptual infrastructure of a dominant paradigm displa | When one field's way of thinking takes over and replaces other ways of understan | 92% |
| **Precision-Recall Tradeoff** | In information retrieval and ML evaluation, the inverse relationship between the | The balance between being accurate and being thorough — finding the right things | 78% |
| **Behavioral Homogenization** | The convergence of diverse individual behaviors toward a narrower repertoire dri | When everyone starts acting the same way because they're all being judged by the | 90% |
| **Epistemic Humility** | The metacognitive recognition of the limits of one's knowledge and the fallibili | Knowing that your way of measuring or understanding things might be wrong or inc | 86% |
| **Value Alignment** | The problem of ensuring that an AI system's objective function and behavioral ou | Making sure AI systems actually pursue what humans truly care about, not just wh | 91% |
| **Instrumental Rationality** | A mode of reasoning focused exclusively on optimizing means for given ends, with | Focusing purely on how to achieve a goal efficiently, without asking whether the | 93% |

## Glossary
| Term | Definition | Concept IDs |
|------|------------|-------------|
| **Metric Borrowing** | The practice of transplanting quantitative measures developed for AI system evaluation directly into | 5, 26, 3 |
| **Ontological Mismatch** | The foundational incompatibility between the kind of entity an AI system is—a computational process  | 11, 32, 8 |
| **Categorical Error** | The logical mistake of applying concepts, evaluative standards, or measurement schemes to an entity  | 32, 11, 14 |
| **Normative Transfer** | The migration of embedded values, priorities, and assumptions from one evaluative domain into anothe | 26, 5, 9 |
| **Goodhart's Law** | The empirical principle that once a measure becomes a target, it ceases to be a good measure, becaus | 18, 10, 38 |
| **Quantification Bias** | The systematic tendency within institutions to privilege information that can be expressed numerical | 7, 14, 31 |
| **Technocratic Isomorphism** | The structural convergence of human institutions toward forms that resemble computational or enginee | 9, 1, 20 |
| **Algorithmic Governance** | The use of automated, rule-bound, or data-driven systems to make or inform decisions about the alloc | 6, 33, 34 |
| **Datafication** | The process of translating aspects of human life, behavior, and identity into discrete data points t | 24, 16, 21 |
| **Agentic Reduction** | The collapse of a person's multidimensional agency—including autonomy, creativity, moral reasoning,  | 25, 14, 17 |
| **Dehumanization** | The institutional and discursive process by which persons are treated as objects, instruments, or da | 17, 13, 35 |
| **Human Dignity** | The intrinsic, non-negotiable worth of persons that is not contingent on productivity, performance,  | 13, 17, 19 |
| **Performance Proxy** | A measurable indicator used as a stand-in for a complex underlying quality, whose validity depends o | 10, 23, 15 |
| **Throughput Optimization** | The engineering goal of maximizing the volume of outputs produced per unit of time or resource, whic | 12, 31, 21 |
| **Epistemic Violence** | The harm inflicted when dominant knowledge frameworks invalidate, silence, or render unintelligible  | 19, 36, 17 |
| **Framework Colonization** | The aggressive expansion of an evaluative framework developed in one domain into contexts where it i | 36, 26, 9 |
| **Behavioral Homogenization** | The narrowing of human conduct toward a small range of metric-legible behaviors as individuals adapt | 38, 18, 31 |
| **Reductionism** | The methodological and ontological stance that complex wholes—including persons—can be fully underst | 14, 11, 25 |
| **Benchmark Validity** | The degree to which a standardized test or evaluation protocol actually measures the construct it cl | 15, 23, 3 |
| **Construct Validity** | The psychometric property of a measure that reflects whether the operationalization of a concept gen | 23, 15, 7 |
| **Instrumental Rationality** | A mode of reasoning that evaluates actions and entities solely by their efficiency in achieving pred | 41, 31, 20 |
| **Efficiency Fetishism** | The ideological elevation of efficiency to an overriding institutional value, such that arrangements | 31, 41, 20 |
| **Neoliberal Managerialism** | An administrative ideology that applies market logic and quantitative performance management to all  | 20, 21, 1 |
| **Labor Commodification** | The treatment of human work and workers as fungible market commodities whose value is exhaustively d | 21, 35, 20 |
| **Alienation** | The estrangement of persons from their own labor, capacities, and self-determination that occurs whe | 35, 21, 25 |
| **Surveillance Capitalism** | An economic logic in which the behavioral data generated by persons is extracted, analyzed, and mone | 16, 30, 24 |
| **Panopticism** | A form of social control in which the possibility of being observed at any time induces individuals  | 30, 16, 38 |
| **Automation Bias** | The documented tendency of human decision-makers to defer to automated system outputs even when cont | 29, 6, 33 |
| **AI Evaluation Frameworks** | The formal methodologies, benchmark suites, and scoring systems developed to assess the capabilities | 3, 15, 5 |
| **Commensurability** | The property of two or more things being measurable on a common scale such that they can be meaningf | 8, 11, 32 |
| **Output Metrics** | Quantitative measures that index the volume, speed, or accuracy of discrete products or results, use | 2, 10, 12 |
| **Scalability** | The capacity of a system or process to expand in throughput without proportional increases in cost o | 28, 12, 9 |
| **Institutional Legitimacy** | The socially constructed acceptance of an institution's authority and methods as appropriate and val | 27, 1, 9 |
| **Institutional Systems** | The organized, rule-governed arrangements through which societies allocate resources, confer credent | 1, 6, 33 |
| **Value Alignment** | The challenge of ensuring that the objectives encoded in an evaluation or optimization system genuin | 40, 26, 41 |
| **Epistemic Humility** | The intellectual disposition to recognize the limits of one's knowledge and methods, particularly im | 39, 23, 19 |
| **Precision-Recall Tradeoff** | A technical property of classification systems describing the inverse relationship between the rate  | 37, 3, 8 |

---
_Generated by Philosopher's Stone v4 — EchoSeed_
