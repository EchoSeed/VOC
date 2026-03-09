import { useState, useCallback, useRef, useEffect } from "react";

// ═══════════════════════════════════════════════════════════════
//  PHILOSOPHER'S STONE v4 — AI Agent Markdown Forge
//  Dual-Register · Triple-Mode · Semantic Taxonomy
//  Compress/Expand · Meta-Analysis · Skill Forge
//  EchoSeed v4.2
// ═══════════════════════════════════════════════════════════════

const DEPTH_PRIMES = {
  shallow: { concepts: 41, terms: 29, clusters: 13, glossary: 37, thesis: 11, label: "Deep", color: "#f472b6" },
};

const STAGES = [
  { id: "dual_extract", name: "Dual-Register Extraction", icon: "⚗️", deps: [], color: "#f59e0b", desc: "Technical + plain language concept pairs" },
  { id: "triple_mode", name: "Triple-Mode Elaboration", icon: "🔬", deps: ["dual_extract"], color: "#22d3ee", desc: "Decision logic · Analogy · Insight per concept" },
  { id: "semantic_index", name: "Semantic Taxonomy", icon: "🗂️", deps: ["dual_extract"], color: "#a78bfa", desc: "Terms, clusters, density mapping" },
  { id: "compress_expand", name: "Compress / Expand", icon: "💎", deps: ["dual_extract"], color: "#f472b6", desc: "Core thesis + glossary generation" },
  { id: "meta_analysis", name: "Meta-Analysis", icon: "📡", deps: [], color: "#34d399", desc: "Complexity profiling & blind spots" },
  { id: "skill_forge", name: "Skill Forge", icon: "🛠️", deps: ["dual_extract", "semantic_index", "compress_expand"], color: "#fb923c", desc: "Transmute pipeline output into SKILL.md" },
];

const STAGE_MAP = Object.fromEntries(STAGES.map(s => [s.id, s]));

function resolveDeps(id, visited = new Set()) {
  if (visited.has(id)) return visited;
  visited.add(id);
  (STAGE_MAP[id]?.deps || []).forEach(d => resolveDeps(d, visited));
  return visited;
}

function resolveDependents(id) {
  const out = new Set();
  STAGES.forEach(s => {
    if (s.deps.includes(id)) { out.add(s.id); resolveDependents(s.id).forEach(d => out.add(d)); }
  });
  return out;
}

function repairJSON(raw) {
  let s = raw.replace(/```json\s*/g, "").replace(/```\s*/g, "").trim();
  try { return JSON.parse(s); } catch {}
  // Fix unescaped newlines inside strings
  s = s.replace(/(?<=":[ ]?"[^"]*)\n(?=[^"]*")/g, "\\n");
  s = s.replace(/\t/g, "\\t").replace(/,\s*([\]}])/g, "$1");
  try { return JSON.parse(s); } catch {}
  // Try fixing all newlines between quotes more aggressively
  let inStr = false, escaped = false, cleaned = "";
  for (let i = 0; i < s.length; i++) {
    const ch = s[i];
    if (escaped) { cleaned += ch; escaped = false; continue; }
    if (ch === "\\") { cleaned += ch; escaped = true; continue; }
    if (ch === '"') { inStr = !inStr; cleaned += ch; continue; }
    if (inStr && ch === "\n") { cleaned += "\\n"; continue; }
    if (inStr && ch === "\r") { cleaned += ""; continue; }
    cleaned += ch;
  }
  try { return JSON.parse(cleaned); } catch {}
  const arrM = cleaned.match(/\[[\s\S]*\]/), objM = cleaned.match(/\{[\s\S]*\}/);
  const m = arrM && arrM[0].length > (objM ? objM[0].length : 0) ? arrM : objM;
  if (m) try { return JSON.parse(m[0]); } catch {}
  return { raw, _parse_failed: true };
}

async function callAPI(system, user, retries = 2, maxTok = 6000, raw = false) {
  for (let i = 0; i <= retries; i++) {
    try {
      const r = await fetch("https://api.anthropic.com/v1/messages", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model: "claude-sonnet-4-20250514", max_tokens: maxTok, messages: [{ role: "user", content: user }], system }),
      });
      if (!r.ok) throw new Error(`API ${r.status}`);
      const d = await r.json();
      const txt = d.content?.map(c => c.text || "").join("\n") || "";
      if (raw) return txt;
      return repairJSON(txt);
    } catch (e) {
      if (i === retries) throw e;
      await new Promise(r => setTimeout(r, 1000 * (i + 1)));
    }
  }
}

// ═══════════════════════════════════════════════════════════════
//  PROMPTS
// ═══════════════════════════════════════════════════════════════

const PROMPTS = {
  dual_extract: (text, depth) => ({
    system: `Extract exactly ${DEPTH_PRIMES[depth].concepts} key concepts using dual-register analysis. Output ONLY valid JSON.`,
    user: `Analyze this text and extract ${DEPTH_PRIMES[depth].concepts} concepts. For each, provide a technical definition AND a plain-language explanation.\n\nTEXT:\n${text}\n\nReturn: [{"id":1,"concept":"...","technical":"...","plain":"...","importance":0.9}]`
  }),
  triple_mode: (text, concepts) => ({
    system: "For each concept output a JSON array. Each element: {concept_id, concept, decision, analogy, insight}. decision/analogy/insight max 100 words each. Output ONLY a raw JSON array, no prose, no fences.",
    user: `CONCEPTS:\n${JSON.stringify(concepts.map(c => ({id: c.id, concept: c.concept})))}\n\nSOURCE:\n${text.slice(0, 2000)}\n\nFor each concept provide:\n- decision: when/why an agent applies this (max 100 words)\n- analogy: metaphor for pattern recognition (max 80 words)\n- insight: non-obvious implication (max 100 words)\n\nReturn ONLY: [{"concept_id":1,"concept":"name","decision":"...","analogy":"...","insight":"..."}]`
  }),
  semantic_index: (text, concepts, depth) => ({
    system: `Build a semantic taxonomy with exactly ${DEPTH_PRIMES[depth].terms} index terms and ${DEPTH_PRIMES[depth].clusters} clusters. Output ONLY valid JSON.`,
    user: `Create a semantic taxonomy for these concepts:\n${JSON.stringify(concepts)}\n\nSOURCE:\n${text.slice(0, 3000)}\n\nReturn: {"terms":[{"term":"...","concept_ids":[1,2]}],"clusters":[{"name":"...","concept_ids":[1,2,3],"description":"..."}],"density_map":{"highest":"...","lowest":"...","avg_connections":2.5}}`
  }),
  compress_expand: (text, concepts, depth) => ({
    system: `Compress knowledge into a ${DEPTH_PRIMES[depth].thesis}-sentence thesis and ${DEPTH_PRIMES[depth].glossary}-entry glossary. Output ONLY valid JSON.`,
    user: `CONCEPTS:\n${JSON.stringify(concepts)}\n\nSOURCE:\n${text.slice(0, 3000)}\n\nCompress into:\n1. A ${DEPTH_PRIMES[depth].thesis}-sentence core thesis\n2. A ${DEPTH_PRIMES[depth].glossary}-entry glossary linking back to concept IDs\n\nReturn: {"thesis":"...","glossary":[{"term":"...","definition":"...","concept_ids":[1]}]}`
  }),
  meta_analysis: (text, allResults, depth) => ({
    system: "Analyze the pipeline outputs for completeness, complexity, and blind spots. Output ONLY valid JSON.",
    user: `Analyze pipeline outputs.\nPIPELINE (depth=${depth}):\n${JSON.stringify(allResults).slice(0, 3000)}\n\nTEXT:\n${text.slice(0, 1000)}\n\nReturn: {"complexity_profile":{"conceptual_density":0.8,"technical_depth":0.7,"abstraction_level":0.6,"interconnectedness":0.9},"emergence_assessment":"...","blind_spots":["..."],"coverage_score":0.85,"recommendations":["..."]}`
  }),
  skill_forge: (text, allResults, depth) => {
    const dp = DEPTH_PRIMES[depth];
    const concepts = (allResults.concepts || []).slice(0, dp.concepts).map(c => ({ id: c.id, concept: c.concept, technical: (c.technical || "").slice(0, 150), plain: (c.plain || "").slice(0, 150), importance: c.importance }));
    const terms = (allResults.taxonomy?.terms || []).slice(0, dp.terms).map(t => ({ term: t.term, concept_ids: t.concept_ids }));
    const clusters = (allResults.taxonomy?.clusters || []).slice(0, dp.clusters).map(cl => ({ name: cl.name, concept_ids: cl.concept_ids, description: (cl.description || "").slice(0, 100) }));
    const thesis = allResults.compression?.thesis || "";
    const glossary = (allResults.compression?.glossary || []).slice(0, dp.glossary).map(g => ({ term: g.term, definition: (g.definition || "").slice(0, 120), concept_ids: g.concept_ids }));
    const elaborations = (allResults.elaborations || []).slice(0, 10).map(e => ({
      concept_id: e.concept_id,
      concept: e.concept || "",
      decision: (e.decision || "").slice(0, 200),
      analogy: (e.analogy || "").slice(0, 150),
      insight: (e.insight || "").slice(0, 200),
    }));
    const emergence = allResults.meta?.emergence_assessment || "";
    const recommendations = (allResults.meta?.recommendations || []);
    const blindSpots = (allResults.meta?.blind_spots || []);
    return {
      system: `You are a Skill File generator following progressive disclosure architecture with prime-number constraints. Extraction used ${dp.concepts} concepts, ${dp.terms} index terms, ${dp.clusters} clusters, ${dp.glossary} glossary entries, and a ${dp.thesis}-sentence thesis. Synthesize ALL pipeline outputs into a single coherent SKILL.md. Output ONLY raw markdown. You MUST include fenced Python code blocks and CSV tables where relevant. Start directly with the # heading.`,
      user: `Synthesize this analyzed knowledge into a SKILL.md instruction file for AI agents.

DEPTH: ${depth} | PRIMES: concepts=${dp.concepts}, terms=${dp.terms}, clusters=${dp.clusters}, glossary=${dp.glossary}, thesis=${dp.thesis}

SOURCE:
${text.slice(0, 3000)}

CONCEPTS (${concepts.length}):
${JSON.stringify(concepts)}

TAXONOMY:
Terms: ${JSON.stringify(terms)}
Clusters: ${JSON.stringify(clusters)}

COMPRESSION:
Thesis: ${thesis}
Glossary: ${JSON.stringify(glossary)}

TRIPLE-MODE ELABORATIONS (top ${elaborations.length}):
${JSON.stringify(elaborations)}

META-ANALYSIS:
Emergence: ${emergence}
Blind Spots: ${JSON.stringify(blindSpots)}
Recommendations: ${JSON.stringify(recommendations)}

Write the SKILL.md now using this structure:
# [2-4 word skill name]

> [description: when to trigger this skill, ~100 tokens]

## Core Thesis
[paste the thesis verbatim from COMPRESSION above — the distilled essence of the source material]

## Overview
[what this skill does and why]

## When to Use
- [trigger condition 1]
- [trigger condition 2]

## Core Workflow
1. [step 1]
2. [step 2]
3. [step 3]

## Key Patterns
### [pattern name]
[actionable insight]

\`\`\`python
# Concrete Python implementation of the pattern
[runnable code snippet, typed, with inline comments]
\`\`\`

## Triple-Mode Insights
[For each top concept from TRIPLE-MODE ELABORATIONS, render a sub-section:]

### [concept name]
**🎯 Decision:** [decision — when/why an agent applies this]
**🎭 Analogy:** [analogy — pattern recognition aid]
**💡 Insight:** [insight — non-obvious implication or edge]

## Concept Reference
| Concept | Technical | Plain | Importance |
|---------|-----------|-------|------------|
[one row per top concept, pulled from CONCEPTS above]

## Glossary
| Term | Definition | Concept IDs |
|------|------------|-------------|
[one row per glossary entry]

## Edge Cases & Warnings
- ⚠️ [warning 1 from blind_spots]
- ⚠️ [warning 2 from blind_spots]

## Emergence Assessment
[paste the emergence field verbatim — this is the meta-analysis observation of what patterns surfaced beyond the explicit concepts]

## Recommendations
- 🔧 [recommendation 1 from recommendations — concrete fix or improvement]
- 🔧 [recommendation 2]

## Quick Reference
\`\`\`python
# Minimal runnable cheat-sheet snippet
[distilled usage example]
\`\`\`

---
_Generated by Philosopher's Stone v4 — EchoSeed_`
    };
  },
};

// ═══════════════════════════════════════════════════════════════
//  SKILL.MD ASSEMBLY
// ═══════════════════════════════════════════════════════════════

// ═══════════════════════════════════════════════════════════════
//  STAGE REPORT RENDERERS
// ═══════════════════════════════════════════════════════════════

const Badge = ({ children, color }) => (
  <span style={{ fontSize: 9, fontWeight: 700, padding: "2px 7px", borderRadius: 10, background: `${color}20`, color, border: `1px solid ${color}44` }}>{children}</span>
);

const Bar = ({ value, color, label }) => (
  <div style={{ marginBottom: 8 }}>
    <div style={{ display: "flex", justifyContent: "space-between", fontSize: 10, marginBottom: 3 }}>
      <span style={{ color: "#94a3b8" }}>{label}</span>
      <span style={{ color, fontWeight: 600 }}>{Math.round((value || 0) * 100)}%</span>
    </div>
    <div style={{ height: 5, borderRadius: 3, background: "rgba(255,255,255,0.06)" }}>
      <div style={{ height: "100%", borderRadius: 3, width: `${(value || 0) * 100}%`, background: `linear-gradient(90deg, ${color}, ${color}aa)`, transition: "width 0.5s" }} />
    </div>
  </div>
);

function ReportDualExtract({ data, color }) {
  const items = Array.isArray(data) ? data : [];
  if (!items.length) return <div style={{ color: "#555", fontSize: 11 }}>No concepts extracted</div>;
  return (
    <div>
      <div style={{ fontSize: 11, fontWeight: 700, color, marginBottom: 8 }}>⚗️ {items.length} Concepts Extracted</div>
      {items.map((c, i) => (
        <div key={i} style={{ marginBottom: 10, padding: "8px 10px", borderRadius: 6, background: "rgba(255,255,255,0.02)", borderLeft: `3px solid ${color}` }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 4 }}>
            <span style={{ fontSize: 12, fontWeight: 700, color: "#e2e8f0" }}>{c.concept || `Concept ${c.id || i + 1}`}</span>
            {c.importance != null && <Badge color={c.importance > 0.8 ? "#f59e0b" : c.importance > 0.5 ? "#a78bfa" : "#64748b"}>{Math.round(c.importance * 100)}%</Badge>}
          </div>
          <div style={{ fontSize: 10, color: "#22d3ee", marginBottom: 3, lineHeight: 1.4 }}><span style={{ fontWeight: 600, color: "#22d3ee99" }}>Technical:</span> {c.technical}</div>
          <div style={{ fontSize: 10, color: "#94a3b8", lineHeight: 1.4 }}><span style={{ fontWeight: 600, color: "#94a3b899" }}>Plain:</span> {c.plain}</div>
        </div>
      ))}
    </div>
  );
}

function ReportTripleMode({ data, color }) {
  const items = Array.isArray(data) ? data : [];
  if (!items.length) return <div style={{ color: "#555", fontSize: 11 }}>No elaborations</div>;
  return (
    <div>
      <div style={{ fontSize: 11, fontWeight: 700, color, marginBottom: 8 }}>🔬 {items.length} Triple-Mode Elaborations</div>
      {items.map((el, i) => (
        <div key={i} style={{ marginBottom: 10, padding: "8px 10px", borderRadius: 6, background: "rgba(255,255,255,0.02)", borderLeft: `3px solid ${color}` }}>
          <div style={{ fontSize: 10, fontWeight: 700, color: "#e2e8f0", marginBottom: 6 }}>#{el.concept_id || i + 1}{el.concept ? ` — ${el.concept}` : ""}</div>
          {el.decision && (
            <div style={{ marginBottom: 6 }}>
              <div style={{ fontSize: 9, fontWeight: 700, color: "#34d399", marginBottom: 2 }}>🎯 DECISION LOGIC</div>
              <div style={{ fontSize: 10, color: "#a7f3d0", lineHeight: 1.4 }}>{el.decision}</div>
            </div>
          )}
          {el.analogy && (
            <div style={{ marginBottom: 6 }}>
              <div style={{ fontSize: 9, fontWeight: 700, color: "#a78bfa", marginBottom: 2 }}>🎭 ANALOGY</div>
              <div style={{ fontSize: 10, color: "#c4b5fd", lineHeight: 1.4, fontStyle: "italic" }}>{el.analogy}</div>
            </div>
          )}
          {el.insight && (
            <div>
              <div style={{ fontSize: 9, fontWeight: 700, color: "#f59e0b", marginBottom: 2 }}>💡 INSIGHT</div>
              <div style={{ fontSize: 10, color: "#fcd34d", lineHeight: 1.4 }}>{el.insight}</div>
            </div>
          )}
        </div>
      ))}
    </div>
  );
}

function ReportSemanticIndex({ data, color }) {
  if (!data || data._parse_failed) return <div style={{ color: "#555", fontSize: 11 }}>No taxonomy data</div>;
  return (
    <div>
      <div style={{ fontSize: 11, fontWeight: 700, color, marginBottom: 8 }}>🗂️ Semantic Taxonomy</div>
      {data.terms?.length > 0 && (
        <div style={{ marginBottom: 12 }}>
          <div style={{ fontSize: 10, fontWeight: 700, color: "#94a3b8", marginBottom: 6 }}>INDEX TERMS ({data.terms.length})</div>
          <div style={{ display: "flex", flexWrap: "wrap", gap: 4 }}>
            {data.terms.map((t, i) => (
              <span key={i} style={{ fontSize: 9, padding: "3px 8px", borderRadius: 12, background: `${color}15`, border: `1px solid ${color}33`, color }}>
                {t.term} <span style={{ opacity: 0.5 }}>→ {t.concept_ids?.length || 0}</span>
              </span>
            ))}
          </div>
        </div>
      )}
      {data.clusters?.length > 0 && (
        <div style={{ marginBottom: 12 }}>
          <div style={{ fontSize: 10, fontWeight: 700, color: "#94a3b8", marginBottom: 6 }}>CLUSTERS ({data.clusters.length})</div>
          {data.clusters.map((cl, i) => (
            <div key={i} style={{ marginBottom: 8, padding: "6px 10px", borderRadius: 6, background: "rgba(255,255,255,0.02)", borderLeft: `3px solid ${color}` }}>
              <div style={{ fontSize: 11, fontWeight: 700, color: "#e2e8f0" }}>{cl.name}</div>
              <div style={{ fontSize: 10, color: "#94a3b8", marginTop: 2, lineHeight: 1.4 }}>{cl.description}</div>
              <div style={{ fontSize: 9, color: "#64748b", marginTop: 3 }}>Concepts: [{cl.concept_ids?.join(", ")}]</div>
            </div>
          ))}
        </div>
      )}
      {data.density_map && (
        <div style={{ padding: "8px 10px", borderRadius: 6, background: "rgba(167,139,250,0.06)", border: "1px solid rgba(167,139,250,0.15)" }}>
          <div style={{ fontSize: 10, fontWeight: 700, color: "#94a3b8", marginBottom: 4 }}>DENSITY MAP</div>
          <div style={{ fontSize: 10, color: "#c4b5fd" }}>Highest: {data.density_map.highest}</div>
          <div style={{ fontSize: 10, color: "#94a3b8" }}>Lowest: {data.density_map.lowest}</div>
          <div style={{ fontSize: 10, color: "#64748b" }}>Avg connections: {data.density_map.avg_connections}</div>
        </div>
      )}
    </div>
  );
}

function ReportCompressExpand({ data, color }) {
  if (!data || data._parse_failed) return <div style={{ color: "#555", fontSize: 11 }}>No compression data</div>;
  return (
    <div>
      <div style={{ fontSize: 11, fontWeight: 700, color, marginBottom: 8 }}>💎 Compress / Expand</div>
      {data.thesis && (
        <div style={{ marginBottom: 12, padding: "10px 12px", borderRadius: 8, background: "rgba(244,114,182,0.06)", border: "1px solid rgba(244,114,182,0.2)" }}>
          <div style={{ fontSize: 10, fontWeight: 700, color: "#f472b6", marginBottom: 6 }}>CORE THESIS</div>
          <div style={{ fontSize: 12, color: "#fda4af", lineHeight: 1.6, fontStyle: "italic" }}>"{data.thesis}"</div>
        </div>
      )}
      {data.glossary?.length > 0 && (
        <div>
          <div style={{ fontSize: 10, fontWeight: 700, color: "#94a3b8", marginBottom: 6 }}>GLOSSARY ({data.glossary.length} entries)</div>
          {data.glossary.map((g, i) => (
            <div key={i} style={{ marginBottom: 6, padding: "6px 10px", borderRadius: 6, background: "rgba(255,255,255,0.02)", borderLeft: `3px solid ${color}` }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                <span style={{ fontSize: 11, fontWeight: 700, color: "#e2e8f0" }}>{g.term}</span>
                <span style={{ fontSize: 8, color: "#64748b" }}>→ [{g.concept_ids?.join(", ")}]</span>
              </div>
              <div style={{ fontSize: 10, color: "#94a3b8", marginTop: 2, lineHeight: 1.4 }}>{g.definition}</div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function ReportMetaAnalysis({ data, color }) {
  if (!data || data._parse_failed) return <div style={{ color: "#555", fontSize: 11 }}>No meta-analysis data</div>;
  return (
    <div>
      <div style={{ fontSize: 11, fontWeight: 700, color, marginBottom: 8 }}>📡 Meta-Analysis</div>
      {data.complexity_profile && typeof data.complexity_profile === "object" && (
        <div style={{ marginBottom: 12 }}>
          <div style={{ fontSize: 10, fontWeight: 700, color: "#94a3b8", marginBottom: 6 }}>COMPLEXITY PROFILE</div>
          {Object.entries(data.complexity_profile).map(([k, v]) => {
            const num = typeof v === "number" ? v : parseFloat(v);
            return isNaN(num) ? null : <Bar key={k} value={num} color={color} label={k.replace(/_/g, " ")} />;
          })}
        </div>
      )}
      {data.coverage_score != null && !isNaN(parseFloat(data.coverage_score)) && (
        <div style={{ marginBottom: 12, padding: 10, borderRadius: 8, background: `${color}0d`, border: `1px solid ${color}25`, textAlign: "center" }}>
          <div style={{ fontSize: 9, color: "#64748b" }}>Coverage</div>
          <div style={{ fontSize: 26, fontWeight: 700, color }}>{Math.round(parseFloat(data.coverage_score) * 100)}%</div>
        </div>
      )}
      {data.emergence_assessment && (
        <div style={{ marginBottom: 12 }}>
          <div style={{ fontSize: 10, fontWeight: 700, color: "#94a3b8", marginBottom: 4 }}>EMERGENCE</div>
          <div style={{ fontSize: 11, color: "#94a3b8", lineHeight: 1.5 }}>{String(data.emergence_assessment)}</div>
        </div>
      )}
      {Array.isArray(data.blind_spots) && data.blind_spots.length > 0 && (
        <div style={{ marginBottom: 12 }}>
          <div style={{ fontSize: 10, fontWeight: 700, color: "#f472b6", marginBottom: 4 }}>BLIND SPOTS</div>
          {data.blind_spots.map((b, i) => <div key={i} style={{ fontSize: 10, color: "#fda4af", padding: "3px 0" }}>⚠️ {b}</div>)}
        </div>
      )}
      {Array.isArray(data.recommendations) && data.recommendations.length > 0 && (
        <div>
          <div style={{ fontSize: 10, fontWeight: 700, color: "#22d3ee", marginBottom: 4 }}>RECOMMENDATIONS</div>
          {data.recommendations.map((r, i) => <div key={i} style={{ fontSize: 10, color: "#67e8f9", padding: "3px 0" }}>→ {String(r)}</div>)}
        </div>
      )}
    </div>
  );
}

function ReportSkillForge({ data, color }) {
  if (!data) return <div style={{ color: "#555", fontSize: 11 }}>No skill data</div>;
  const md = typeof data === "string" ? data : "";
  const name = (md.match(/^#\s+(.+)/m) || [])[1] || "Skill File";
  const lineCount = md.split("\n").length;
  return (
    <div>
      <div style={{ fontSize: 11, fontWeight: 700, color, marginBottom: 4 }}>🛠️ Skill Forge Output</div>
      <div style={{ padding: "8px 10px", borderRadius: 6, background: "rgba(251,146,60,0.06)", border: "1px solid rgba(251,146,60,0.2)", marginBottom: 8 }}>
        <div style={{ fontSize: 14, fontWeight: 700, color: "#fb923c" }}>{name}</div>
        <div style={{ fontSize: 10, color: "#94a3b8", marginTop: 4 }}>{lineCount} lines — see right panel for full output</div>
      </div>
    </div>
  );
}

const REPORT_RENDERERS = {
  dual_extract: ReportDualExtract,
  triple_mode: ReportTripleMode,
  semantic_index: ReportSemanticIndex,
  compress_expand: ReportCompressExpand,
  meta_analysis: ReportMetaAnalysis,
  skill_forge: ReportSkillForge,
};

function RenderJSON({ obj, depth = 0 }) {
  if (obj === null || obj === undefined) return <span style={{ color: "#555" }}>null</span>;
  if (typeof obj === "boolean") return <span style={{ color: "#f472b6" }}>{String(obj)}</span>;
  if (typeof obj === "number") return <span style={{ color: "#22d3ee" }}>{obj}</span>;
  if (typeof obj === "string") {
    const display = obj.length > 140 ? obj.slice(0, 140) + "…" : obj;
    return <span style={{ color: "#a78bfa" }}>&quot;{display}&quot;</span>;
  }
  if (Array.isArray(obj)) {
    if (!obj.length) return <span style={{ color: "#555" }}>{"[]"}</span>;
    return (
      <div style={{ paddingLeft: depth > 0 ? 12 : 0 }}>
        {obj.map((item, i) => (
          <div key={i} style={{ borderLeft: "2px solid rgba(255,255,255,0.06)", paddingLeft: 8, marginBottom: 4 }}>
            <span style={{ color: "#555", fontSize: 9 }}>[{i}] </span>
            <RenderJSON obj={item} depth={depth + 1} />
          </div>
        ))}
      </div>
    );
  }
  if (typeof obj === "object") {
    return (
      <div style={{ paddingLeft: depth > 0 ? 12 : 0 }}>
        {Object.entries(obj).map(([k, v]) => (
          <div key={k} style={{ marginBottom: 3 }}>
            <span style={{ color: "#f59e0b", fontWeight: 600, fontSize: 10 }}>{k}: </span>
            {(typeof v === "object" && v !== null) ? <RenderJSON obj={v} depth={depth + 1} /> : <RenderJSON obj={v} depth={depth} />}
          </div>
        ))}
      </div>
    );
  }
  return <span>{String(obj)}</span>;
}

function StageReport({ data, stageId, color }) {
  const [showRaw, setShowRaw] = useState(false);
  const Renderer = REPORT_RENDERERS[stageId];
  if (!Renderer) return null;
  return (
    <div>
      <Renderer data={data} color={color} />
      <div style={{ marginTop: 10, borderTop: "1px solid rgba(255,255,255,0.06)", paddingTop: 8 }}>
        <div
          onClick={() => setShowRaw(prev => !prev)}
          style={{ fontSize: 9, fontWeight: 700, color: "#64748b", cursor: "pointer", display: "flex", alignItems: "center", gap: 4, userSelect: "none" }}
        >
          <span style={{ display: "inline-block", transition: "transform 0.2s", transform: showRaw ? "rotate(90deg)" : "rotate(0deg)" }}>▶</span>
          RAW JSON
        </div>
        {showRaw && (
          <div style={{ marginTop: 6, padding: 8, background: "rgba(0,0,0,0.4)", borderRadius: 6, fontSize: 10, fontFamily: "'Courier New', monospace", maxHeight: 250, overflow: "auto" }}>
            <pre style={{ margin: 0, whiteSpace: "pre-wrap", wordBreak: "break-word", color: "#94a3b8", fontSize: 9, lineHeight: 1.5 }}>
              {JSON.stringify(data, null, 2)}
            </pre>
          </div>
        )}
      </div>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════
//  STYLES
// ═══════════════════════════════════════════════════════════════

const S = {
  root: { minHeight: "100vh", background: "linear-gradient(135deg, #0a0a1a 0%, #0d1117 50%, #0a0a1a 100%)", color: "#e2e8f0", fontFamily: "-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif", fontSize: 13 },
  header: { padding: "12px 24px 10px", borderBottom: "1px solid rgba(255,255,255,0.06)", background: "linear-gradient(180deg, rgba(251,146,60,0.08) 0%, transparent 100%)" },
  title: { margin: 0, fontSize: 24, fontWeight: 700, background: "linear-gradient(135deg, #f59e0b, #fb923c, #f472b6)", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent", letterSpacing: "-0.02em" },
  subtitle: { margin: "4px 0 0", fontSize: 11, color: "#64748b", letterSpacing: "0.08em", textTransform: "uppercase" },
  layout: { display: "flex", height: "calc(100vh - 56px)", overflow: "hidden" },
  left: { width: 380, minWidth: 380, borderRight: "1px solid rgba(255,255,255,0.06)", display: "flex", flexDirection: "column", overflowY: "auto", height: "100%" },
  center: { flex: 1, display: "flex", flexDirection: "column", overflowY: "auto", height: "100%" },
  right: { width: 520, minWidth: 520, borderLeft: "1px solid rgba(255,255,255,0.06)", display: "flex", flexDirection: "column", overflowY: "auto", height: "100%" },
  sec: { padding: "14px 18px", borderBottom: "1px solid rgba(255,255,255,0.04)" },
  label: { fontSize: 11, fontWeight: 600, letterSpacing: "0.1em", textTransform: "uppercase", marginBottom: 8 },
  ta: { width: "100%", minHeight: 260, background: "rgba(255,255,255,0.03)", border: "1px solid rgba(255,255,255,0.08)", borderRadius: 8, color: "#e2e8f0", padding: "10px 12px", fontSize: 12, fontFamily: "inherit", resize: "vertical", outline: "none", boxSizing: "border-box" },
  btn: (c, off) => ({ padding: "10px 0", width: "100%", border: "none", borderRadius: 8, fontWeight: 700, fontSize: 13, cursor: off ? "default" : "pointer", color: "#000", background: off ? "#333" : `linear-gradient(135deg, ${c}, ${c}dd)`, opacity: off ? 0.4 : 1, transition: "all 0.2s" }),
  pill: (c, on) => ({ padding: "4px 10px", borderRadius: 20, fontSize: 10, fontWeight: 600, border: `1.5px solid ${on ? c : "rgba(255,255,255,0.1)"}`, background: on ? `${c}18` : "transparent", color: on ? c : "#555", cursor: "pointer", transition: "all 0.2s" }),
  card: (c, st) => ({ padding: "10px 14px", borderRadius: 8, border: `1px solid ${st === "running" ? c : st === "complete" ? c + "66" : "rgba(255,255,255,0.06)"}`, background: st === "running" ? `${c}0d` : st === "complete" ? `${c}08` : "rgba(255,255,255,0.02)", transition: "all 0.3s", marginBottom: 6, cursor: st === "complete" ? "pointer" : "default" }),
};

// ═══════════════════════════════════════════════════════════════
//  MAIN COMPONENT
// ═══════════════════════════════════════════════════════════════

export default function PhilosophersStone() {
  const [input, setInput] = useState("");
  const depth = "shallow";
  const [selected, setSelected] = useState(new Set(STAGES.map(s => s.id)));
  const [status, setStatus] = useState({});
  const [results, setResults] = useState({});
  const [running, setRunning] = useState(false);
  const [error, setError] = useState(null);
  const [log, setLog] = useState([]);
  const [expanded, setExpanded] = useState(null);
  const [copied, setCopied] = useState(false);
  const logRef = useRef(null);
  const startTime = useRef(null);

  const addLog = useCallback((msg, type = "info") => {
    setLog(p => [...p, { t: new Date().toLocaleTimeString(), msg, type }]);
  }, []);

  useEffect(() => { if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight; }, [log]);

  const toggle = (id) => {
    if (running) return;
    setSelected(prev => {
      const next = new Set(prev);
      if (next.has(id)) { next.delete(id); resolveDependents(id).forEach(d => next.delete(d)); }
      else { resolveDeps(id).forEach(d => next.add(d)); }
      return next;
    });
  };

  const runPipeline = async () => {
    if (!input.trim() || running) return;
    setRunning(true); setError(null); setResults({}); setLog([]); setExpanded(null); setCopied(false);
    startTime.current = Date.now();
    const st = {};
    STAGES.forEach(s => { st[s.id] = selected.has(s.id) ? "pending" : "skipped"; });
    setStatus(st);
    addLog(`Pipeline initiated — depth: ${depth} (${DEPTH_PRIMES[depth].concepts} concepts)`);

    const all = {};
    try {
      // Stage 1: Dual Extract
      if (selected.has("dual_extract")) {
        setStatus(p => ({ ...p, dual_extract: "running" }));
        addLog("⚗️ Extracting dual-register concepts...");
        const pr = PROMPTS.dual_extract(input, depth);
        const r = await callAPI(pr.system, pr.user);
        all.concepts = Array.isArray(r) ? r : r.concepts || r;
        setResults(p => ({ ...p, dual_extract: all.concepts }));
        setStatus(p => ({ ...p, dual_extract: "complete" }));
        addLog(`⚗️ Extracted ${Array.isArray(all.concepts) ? all.concepts.length : "?"} concepts`, "success");
      }

      // Stage 2 & 3: Triple Mode + Semantic Index (parallel)
      const tripleP = selected.has("triple_mode") && all.concepts ? (async () => {
        setStatus(p => ({ ...p, triple_mode: "running" }));
        addLog("🔬 Elaborating triple modes...");
        try {
          const topConcepts = [...(all.concepts || [])].sort((a, b) => (b.importance || 0) - (a.importance || 0)).slice(0, 17);
          const pr = PROMPTS.triple_mode(input, topConcepts);
          const rawTxt = await callAPI(pr.system, pr.user, 2, 10000, true);
          // Strip fences, find the outermost JSON array
          const stripped = rawTxt.replace(/```json\s*/gi, "").replace(/```\s*/g, "").trim();
          const arrStart = stripped.indexOf("[");
          const arrEnd = stripped.lastIndexOf("]");
          let eArr = [];
          if (arrStart !== -1 && arrEnd > arrStart) {
            try { eArr = JSON.parse(stripped.slice(arrStart, arrEnd + 1)); } catch {
              // Parse object-by-object between { }
              const objRx = /\{[^{}]*"concept_id"[^{}]*\}/gs;
              const matches = stripped.match(objRx) || [];
              eArr = matches.reduce((acc, m) => {
                try { acc.push(JSON.parse(m)); } catch {}
                return acc;
              }, []);
            }
          }
          all.elaborations = Array.isArray(eArr) ? eArr : [];
          setResults(p => ({ ...p, triple_mode: all.elaborations }));
          setStatus(p => ({ ...p, triple_mode: "complete" }));
          addLog(`🔬 Triple-mode complete: ${all.elaborations.length} elaborations`, "success");
        } catch(e) {
          addLog(`🔬 Triple-mode error: ${e.message}`, "error");
          setStatus(p => ({ ...p, triple_mode: "complete" }));
        }
      })() : Promise.resolve();

      const semP = selected.has("semantic_index") && all.concepts ? (async () => {
        setStatus(p => ({ ...p, semantic_index: "running" }));
        addLog("🗂️ Building semantic taxonomy...");
        const pr = PROMPTS.semantic_index(input, all.concepts, depth);
        const r = await callAPI(pr.system, pr.user);
        all.taxonomy = r;
        setResults(p => ({ ...p, semantic_index: r }));
        setStatus(p => ({ ...p, semantic_index: "complete" }));
        addLog(`🗂️ Taxonomy: ${r.terms?.length || "?"} terms, ${r.clusters?.length || "?"} clusters`, "success");
      })() : Promise.resolve();

      await Promise.all([tripleP, semP]);

      // Stage 4: Compress / Expand
      if (selected.has("compress_expand") && all.concepts) {
        setStatus(p => ({ ...p, compress_expand: "running" }));
        addLog("💎 Compressing and expanding...");
        const pr = PROMPTS.compress_expand(input, all.concepts, depth);
        const r = await callAPI(pr.system, pr.user);
        all.compression = r;
        setResults(p => ({ ...p, compress_expand: r }));
        setStatus(p => ({ ...p, compress_expand: "complete" }));
        addLog(`💎 Thesis compressed, ${r.glossary?.length || "?"} glossary entries`, "success");
      }

      // Stage 5 & 6: Meta-Analysis + Skill Forge (parallel)
      const metaP = selected.has("meta_analysis") ? (async () => {
        setStatus(p => ({ ...p, meta_analysis: "running" }));
        addLog("📡 Running meta-analysis...");
        const pr = PROMPTS.meta_analysis(input, all, depth);
        const r = await callAPI(pr.system, pr.user);
        all.meta = r;
        setResults(p => ({ ...p, meta_analysis: r }));
        setStatus(p => ({ ...p, meta_analysis: "complete" }));
        addLog(`📡 Coverage: ${Math.round((r.coverage_score || 0) * 100)}%`, "success");
      })() : Promise.resolve();

      const forgeP = selected.has("skill_forge") && all.concepts ? (async () => {
        setStatus(p => ({ ...p, skill_forge: "running" }));
        const dp = DEPTH_PRIMES[depth];
        addLog(`🛠️ Forging SKILL.md — c=${dp.concepts} t=${dp.terms} cl=${dp.clusters} g=${dp.glossary} th=${dp.thesis}`);
        try {
          const pr = PROMPTS.skill_forge(input, all, depth);
          const md = await callAPI(pr.system, pr.user, 2, 16000, true);
          const lines = md.split("\n");
          const firstLine = lines[0].trim();
          const lastLine = lines[lines.length - 1].trim();
          const cleaned = (firstLine.startsWith("```") && lastLine === "```")
            ? lines.slice(1, -1).join("\n").trim()
            : md.replace(/^```(?:markdown)?\n/, "").replace(/\n```$/, "").trim();
          all.skill = cleaned;
          setResults(p => ({ ...p, skill_forge: cleaned }));
          setStatus(p => ({ ...p, skill_forge: "complete" }));
          const name = (cleaned.match(/^#\s+(.+)/m) || [])[1] || "unnamed";
          addLog(`🛠️ Skill forged: ${name}`, "success");
        } catch (e) {
          addLog(`🛠️ Skill Forge error: ${e.message}`, "error");
          setStatus(p => ({ ...p, skill_forge: "complete" }));
        }
      })() : Promise.resolve();

      await Promise.all([metaP, forgeP]);

      const elapsed = ((Date.now() - startTime.current) / 1000).toFixed(1);
      addLog(`✅ Pipeline complete in ${elapsed}s`, "success");
    } catch (e) {
      setError(e.message);
      addLog(`❌ Error: ${e.message}`, "error");
    }
    setRunning(false);
  };

  const copySkill = () => {
    const md = typeof results.skill_forge === "string" ? results.skill_forge : "";
    if (md) { navigator.clipboard.writeText(md); setCopied(true); setTimeout(() => setCopied(false), 2000); }
  };

  useEffect(() => {
    const handler = (e) => {
      if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
        e.preventDefault();
        if (!running && input.trim()) runPipeline();
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [running, input, runPipeline]);

  const completedCount = Object.values(status).filter(s => s === "complete").length;
  const totalSelected = [...selected].length;

  return (
    <div style={S.root}>
      {/* Header */}
      <div style={S.header}>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
          <div>
            <h1 style={S.title}>⚗️ Philosopher's Stone</h1>
            <p style={S.subtitle}>v4 — AI Agent Markdown Forge · EchoSeed</p>
          </div>
          {running && (
            <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
              <div style={{ width: 8, height: 8, borderRadius: "50%", background: "#f59e0b", animation: "pulse 1s infinite alternate" }} />
              <span style={{ fontSize: 11, color: "#f59e0b" }}>Processing...</span>
            </div>
          )}
          {completedCount > 0 && !running && <span style={{ fontSize: 11, color: "#34d399" }}>✅ {completedCount}/{totalSelected} stages complete</span>}
        </div>
      </div>

      <style>{`
        @keyframes pulse { from { opacity: 0.4; } to { opacity: 1; } }
        @keyframes shimmer { from { background-position: -200% 0; } to { background-position: 200% 0; } }
        ::-webkit-scrollbar { width: 7px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.15); border-radius: 4px; }
        ::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.25); }
        textarea:focus { border-color: rgba(245,158,11,0.5) !important; box-shadow: 0 0 0 2px rgba(245,158,11,0.1); }
        textarea { transition: border-color 0.2s, box-shadow 0.2s; }
        .stage-row:hover { background: rgba(255,255,255,0.04) !important; }
        .stage-row.active:hover { filter: brightness(1.15); }
        .run-btn:hover:not(:disabled) { filter: brightness(1.2); transform: translateY(-1px); box-shadow: 0 4px 16px rgba(245,158,11,0.3); }
        .run-btn:active:not(:disabled) { transform: translateY(0); }
        .copy-btn:hover { background: #64748b !important; }
        .card-done:hover { filter: brightness(1.08); }
        * { box-sizing: border-box; }
      `}</style>

      <div style={S.layout}>
        {/* ═══ LEFT PANEL ═══ */}
        <div style={S.left}>
          <div style={S.sec}>
            <div style={{ ...S.label, color: "#f59e0b" }}>Source Material</div>
            <textarea style={S.ta} placeholder={"Paste any text — research, docs, specs, conversations, code…\n\nThe Stone transmutes raw knowledge into structured SKILL.md files for AI agents."} value={input} onChange={e => setInput(e.target.value)} disabled={running} />
            <div style={{ display: "flex", justifyContent: "space-between", marginTop: 6 }}>
              <span style={{ fontSize: 10, color: "#475569" }}>{input.length > 0 ? `${input.length.toLocaleString()} chars` : ""}</span>
              {input.length > 0 && !running && <span style={{ fontSize: 10, color: "#64748b", cursor: "pointer" }} onClick={() => setInput("")}>Clear</span>}
            </div>
          </div>

          <div style={S.sec}>
            <div style={{ ...S.label, color: "#22d3ee" }}>Pipeline Stages</div>
            {STAGES.map(stage => {
              const active = selected.has(stage.id);
              return (
                <div key={stage.id} onClick={() => toggle(stage.id)} className={`stage-row${active ? " active" : ""}`} style={{ display: "flex", alignItems: "center", gap: 10, padding: "9px 12px", borderRadius: 6, marginBottom: 4, cursor: running ? "default" : "pointer", background: active ? `${stage.color}0a` : "transparent", border: `1px solid ${active ? stage.color + "33" : "transparent"}`, transition: "all 0.2s" }}>
                  <div style={{ width: 16, height: 16, borderRadius: 4, display: "flex", alignItems: "center", justifyContent: "center", border: `2px solid ${active ? stage.color : "#333"}`, background: active ? stage.color : "transparent", fontSize: 10, color: "#000", fontWeight: 700, transition: "all 0.2s" }}>{active && "✓"}</div>
                  <span style={{ fontSize: 14 }}>{stage.icon}</span>
                  <div style={{ flex: 1 }}>
                    <div style={{ fontSize: 11, fontWeight: 600, color: active ? "#e2e8f0" : "#555" }}>{stage.name}</div>
                    <div style={{ fontSize: 9, color: "#475569", marginTop: 1 }}>{stage.desc}</div>
                  </div>
                  {stage.deps.length > 0 && <div style={{ fontSize: 8, color: "#475569", background: "rgba(255,255,255,0.03)", padding: "2px 5px", borderRadius: 3 }}>needs {stage.deps.length}</div>}
                </div>
              );
            })}
          </div>

          <div style={{ padding: "14px 18px" }}>
            <button className="run-btn" onClick={runPipeline} disabled={!input.trim() || running} style={S.btn("#f59e0b", !input.trim() || running)}>
              {running ? "⚗️ Transmuting..." : "⚗️ Run Pipeline"}
            </button>
            {error && <div style={{ fontSize: 10, color: "#ef4444", marginTop: 6 }}>Error: {error}</div>}
            <div style={{ fontSize: 9, color: "#333", textAlign: "center", marginTop: 6 }}>Ctrl+Enter to run</div>
          </div>
        </div>

        {/* ═══ CENTER PANEL ═══ */}
        <div style={S.center}>
          <div style={S.sec}>
            <div style={{ ...S.label, color: "#34d399" }}>Pipeline Progress & Reports</div>
            {STAGES.filter(s => selected.has(s.id)).map(stage => {
              const st = status[stage.id] || "idle";
              const isRunning = st === "running";
              const isDone = st === "complete";
              const isExp = expanded === stage.id;
              return (
                <div key={stage.id}>
                  <div style={S.card(stage.color, st)} onClick={() => isDone && setExpanded(isExp ? null : stage.id)}>
                    <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
                      <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                        <span style={{ fontSize: 18 }}>{stage.icon}</span>
                        <div>
                          <div style={{ fontSize: 12, fontWeight: 600, color: isRunning ? stage.color : isDone ? "#e2e8f0" : "#555" }}>{stage.name}</div>
                          <div style={{ fontSize: 9, color: "#475569" }}>{stage.desc}</div>
                        </div>
                      </div>
                      <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                        {isRunning && (
                          <div style={{ width: 50, height: 3, borderRadius: 2, overflow: "hidden", background: "rgba(255,255,255,0.06)" }}>
                            <div style={{ height: "100%", borderRadius: 2, background: `linear-gradient(90deg, transparent, ${stage.color}, transparent)`, backgroundSize: "200% 100%", animation: "shimmer 1.5s infinite" }} />
                          </div>
                        )}
                        <span style={{ fontSize: 9, fontWeight: 600, padding: "3px 8px", borderRadius: 10, background: isRunning ? `${stage.color}22` : isDone ? "#34d39922" : "rgba(255,255,255,0.03)", color: isRunning ? stage.color : isDone ? "#34d399" : "#555" }}>
                          {isRunning ? "RUNNING" : isDone ? "DONE" : st === "pending" ? "QUEUED" : "—"}
                        </span>
                        {isDone && <span style={{ fontSize: 10, color: "#475569", transition: "transform 0.2s", transform: isExp ? "rotate(90deg)" : "rotate(0deg)", display: "inline-block" }}>▶</span>}
                      </div>
                    </div>
                  </div>
                  {isExp && results[stage.id] && (
                    <div style={{ padding: "12px 14px", marginBottom: 6, borderRadius: "0 0 8px 8px", marginTop: -6, background: "rgba(0,0,0,0.3)", border: `1px solid ${stage.color}22`, borderTop: "none", maxHeight: 450, overflow: "auto" }}>
                      <StageReport data={results[stage.id]} stageId={stage.id} color={stage.color} />
                    </div>
                  )}
                </div>
              );
            })}
            {Object.keys(status).length === 0 && (
              <div style={{ padding: 40, textAlign: "center", color: "#333" }}>
                <div style={{ fontSize: 40, marginBottom: 12 }}>⚗️</div>
                <div style={{ fontSize: 12, color: "#475569" }}>Paste source material and run the pipeline</div>
                <div style={{ fontSize: 10, marginTop: 4, color: "#333" }}>The Stone awaits transmutation</div>
              </div>
            )}
          </div>

          <div style={{ ...S.sec, flex: 1 }}>
            <div style={{ ...S.label, color: "#64748b" }}>Pipeline Log</div>
            <div ref={logRef} style={{ maxHeight: 180, overflow: "auto", fontFamily: "'Courier New', monospace" }}>
              {log.length === 0 && <div style={{ fontSize: 10, color: "#333" }}>Awaiting pipeline execution...</div>}
              {log.map((l, i) => (
                <div key={i} style={{ fontSize: 11, fontFamily: "'Courier New', monospace", padding: "3px 0", color: l.type === "error" ? "#ef4444" : l.type === "success" ? "#34d399" : "#64748b", borderBottom: "1px solid rgba(255,255,255,0.02)" }}>
                  <span style={{ color: "#333", marginRight: 8 }}>{l.t}</span>{l.msg}
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* ═══ RIGHT PANEL — SKILL.MD OUTPUT ═══ */}
        <div style={S.right}>
          <div style={{ padding: "12px 18px", borderBottom: "1px solid rgba(255,255,255,0.06)", display: "flex", alignItems: "center", justifyContent: "space-between" }}>
            <div style={{ fontSize: 11, fontWeight: 700, color: "#fb923c", letterSpacing: "0.06em" }}>🛠️ SKILL.MD OUTPUT</div>
            {typeof results.skill_forge === "string" && results.skill_forge.length > 0 && (
              <button className="copy-btn" onClick={copySkill} style={{ padding: "5px 14px", border: "none", borderRadius: 5, fontWeight: 700, fontSize: 11, cursor: "pointer", color: "#fff", background: copied ? "#16a34a" : "#475569", transition: "all 0.2s" }}>
                {copied ? "✓ Copied!" : "📋 Copy"}
              </button>
            )}
          </div>
          <div style={{ flex: 1, overflow: "auto", padding: "16px 20px", background: "#f8fafc" }}>
            {typeof results.skill_forge === "string" && results.skill_forge.length > 0 ? (
              <pre style={{ whiteSpace: "pre-wrap", wordBreak: "break-word", fontSize: 12, lineHeight: 1.75, color: "#0f172a", fontFamily: "'Courier New', monospace", margin: 0 }}>
                {results.skill_forge}
              </pre>
            ) : (
              <div style={{ padding: 40, textAlign: "center" }}>
                <div style={{ fontSize: 40, marginBottom: 12 }}>⚗️</div>
                <div style={{ fontSize: 12, color: "#475569" }}>SKILL.md output will render here</div>
                <div style={{ fontSize: 10, marginTop: 4, color: "#333" }}>Run pipeline with Skill Forge enabled</div>
              </div>
            )}
          </div>
          <div style={{ padding: "10px 18px", borderTop: "1px solid rgba(255,255,255,0.04)", fontSize: 9, color: "#333", textAlign: "center" }}>
            Philosopher's Stone v4 × Skill Forge × EchoSeed
          </div>
        </div>
      </div>
    </div>
  );
}
