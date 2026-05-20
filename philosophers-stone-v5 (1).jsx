import { useState, useCallback, useRef, useEffect } from "react";

// ═══════════════════════════════════════════════════════════════
//  PHILOSOPHER'S STONE v5 — AI Agent Markdown Forge
//  Dual-Register · Triple-Mode · Semantic Taxonomy
//  Compress/Expand · Meta-Analysis · Skill Forge
//  EchoSeed v5
// ═══════════════════════════════════════════════════════════════

// Source-proportional prime scaling — prevents concept inflation on short sources
function computeDP(text) {
  const words = text.trim().split(/\s+/).length;
  // ~1 concept per 8 words, capped at 41, floored at 7 (next prime above 5)
  const rawConcepts = Math.round(words / 8);
  const PRIMES = [7, 11, 13, 17, 19, 23, 29, 31, 37, 41];
  const concepts = PRIMES.find(p => p >= rawConcepts) || 41;
  // Scale other fields proportionally
  const ratio = concepts / 41;
  const snap = (arr, val) => arr.find(p => p >= Math.round(val)) || arr[arr.length - 1];
  return {
    concepts,
    terms:    snap(PRIMES, 29 * ratio),
    clusters: snap([5,7,11,13], 13 * ratio),
    glossary: snap(PRIMES, 37 * ratio),
    thesis:   snap([3,5,7,11], 11 * ratio),
  };
}
const DP_MAX = { concepts: 41, terms: 29, clusters: 13, glossary: 37, thesis: 11 };

const STAGES = [
  { id: "dual_extract",    name: "Dual-Register Extraction", icon: "⚗️",  deps: [],                                                   color: "#f59e0b", desc: "Technical + plain language concept pairs" },
  { id: "triple_mode",     name: "Triple-Mode Elaboration",  icon: "🔬",  deps: ["dual_extract"],                                     color: "#22d3ee", desc: "Decision logic · Analogy · Insight per concept" },
  { id: "semantic_index",  name: "Semantic Taxonomy",        icon: "🗂️", deps: ["dual_extract"],                                     color: "#a78bfa", desc: "Terms, clusters, density mapping" },
  { id: "compress_expand", name: "Compress / Expand",        icon: "💎",  deps: ["dual_extract"],                                     color: "#f472b6", desc: "Core thesis + glossary generation" },
  { id: "meta_analysis",   name: "Meta-Analysis",            icon: "📡",  deps: [],                                                   color: "#34d399", desc: "Complexity profiling & blind spots" },
  { id: "skill_forge",     name: "Skill Forge",              icon: "🛠️", deps: ["dual_extract", "semantic_index", "compress_expand"], color: "#fb923c", desc: "Transmute pipeline output into SKILL.md" },
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

// ─── Parsing helpers ─────────────────────────────────────────

function stripFences(txt) {
  return (txt || "").replace(/```[\w]*\n?/g, "").trim();
}

function extractArray(txt) {
  const s = stripFences(txt);
  const start = s.indexOf("[");
  const end = s.lastIndexOf("]");
  if (start === -1 || end <= start) return [];
  try { const r = JSON.parse(s.slice(start, end + 1)); return Array.isArray(r) ? r : []; } catch {}
  // object-by-object fallback
  const matches = s.match(/\{[\s\S]*?\}/g) || [];
  return matches.reduce((acc, m) => { try { acc.push(JSON.parse(m)); } catch {} return acc; }, []);
}

function extractObject(txt) {
  const s = stripFences(txt);
  const start = s.indexOf("{");
  const end = s.lastIndexOf("}");
  if (start === -1 || end <= start) return null;
  // Attempt 1: direct parse
  try { return JSON.parse(s.slice(start, end + 1)); } catch {}
  // Attempt 2: repair unescaped control chars inside strings
  let inStr = false, esc = false, out = "";
  for (const ch of s) {
    if (esc) { out += ch; esc = false; continue; }
    if (ch === "\\") { out += ch; esc = true; continue; }
    if (ch === '"') { inStr = !inStr; out += ch; continue; }
    if (inStr && ch === "\n") { out += "\\n"; continue; }
    if (inStr && ch === "\r") continue;
    if (inStr && ch === "\t") { out += "\\t"; continue; }
    out += ch;
  }
  out = out.replace(/,\s*([\]}])/g, "$1");
  try { return JSON.parse(out.slice(out.indexOf("{"), out.lastIndexOf("}") + 1)); } catch {}
  // Attempt 3: regex field-by-field extraction (handles broken code fields)
  const grab = (key) => {
    const rx = new RegExp(`"${key}"\\s*:\\s*"((?:[^"\\\\]|\\\\.)*)"`, "s");
    const m = s.match(rx);
    return m ? m[1].replace(/\\n/g, "\n").replace(/\\t/g, "\t") : null;
  };
  const grabArr = (key) => {
    const rx = new RegExp(`"${key}"\\s*:\\s*\\[([^\\]]*?)\\]`, "s");
    const m = s.match(rx);
    if (!m) return null;
    return [...m[1].matchAll(/"((?:[^"\\\\]|\\\\.)*)"/gs)].map(x => x[1]);
  };
  const grabObjArr = (key) => {
    const rx = new RegExp(`"${key}"\\s*:\\s*\\[([\\s\\S]*?)\\](?=\\s*[,}])`, "s");
    const m = s.match(rx);
    if (!m) return null;
    return [...m[1].matchAll(/\{([\s\S]*?)\}/gs)].map(match => {
      const chunk = match[1];
      const nameM   = chunk.match(/"name"\s*:\s*"((?:[^"\\]|\\.)*)"/s);
      const insightM = chunk.match(/"insight"\s*:\s*"((?:[^"\\]|\\.)*)"/s);
      return { name: nameM?.[1] || "", insight: insightM?.[1] || "" };
    });
  };
  const result = {
    skill_name:          grab("skill_name"),
    trigger:             grab("trigger"),
    overview:            grab("overview"),
    when_to_use:         grabArr("when_to_use"),
    workflow:            grabArr("workflow"),
    key_patterns:        grabObjArr("key_patterns"),
    implementation_code: null,
    cheat_sheet_code:    null,
  };
  if (Object.values(result).some(v => v !== null)) return result;
  return null;
}

function toArr(v) {
  if (Array.isArray(v)) return v;
  if (v && typeof v === "object") {
    const found = Object.values(v).find(x => Array.isArray(x));
    if (found) return found;
  }
  return [];
}

function toStr(v) {
  if (!v) return "";
  if (typeof v === "string") return v;
  if (typeof v === "object") return v.text || v.summary || v.assessment || v.value || JSON.stringify(v);
  return String(v);
}

// ─── API ─────────────────────────────────────────────────────

async function callAPI(system, user, maxTok = 4000, retries = 1) {
  for (let i = 0; i <= retries; i++) {
    try {
      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), 40000);
      const r = await fetch("https://api.anthropic.com/v1/messages", {
        signal: controller.signal,
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "x-api-key": "",
          "anthropic-version": "2023-06-01",
          "anthropic-dangerous-direct-browser-access": "true",
        },
        body: JSON.stringify({
          model: "claude-sonnet-4-6",
          max_tokens: maxTok,
          system,
          messages: [{ role: "user", content: user }],
        }),
      });
      clearTimeout(timeout);
      if (!r.ok) {
        const errText = await r.text().catch(() => r.status);
        throw new Error(`API ${r.status}: ${errText}`);
      }
      const d = await r.json();
      return d.content?.map(c => c.text || "").join("\n") || "";
    } catch (e) {
      if (i === retries) throw e;
      await new Promise(res => setTimeout(res, 1500));
    }
  }
}

// ─── Prompts ─────────────────────────────────────────────────

const PROMPTS = {
  dual_extract: (text, DP) => ({
    system: `Extract exactly ${DP.concepts} key concepts using dual-register analysis. Output ONLY a raw JSON array, no prose, no fences.`,
    user: `Analyze this text and extract exactly ${DP.concepts} concepts. For each: technical definition (STRICT MAX 25 words), plain explanation (STRICT MAX 20 words), importance 0.0-1.0. Exceeding word limits is not allowed.

Extraction rules:
- Distinguish ACTIVE behaviors (intentional-seeming substitution, redefinition) from PASSIVE states (drift, vulnerability) — do not collapse them into one concept
- Preserve temporal/progressive markers (e.g. "left unchecked", "over time", "entrenchment") as distinct concepts, not adjectives folded into another concept
- PROVENANCE: prefix technical field with "extracted:" if the concept uses source vocabulary or clear direct implication; prefix with "inferred:" if it introduces domain theory, terminology, or framing not present in the source
- Treat mechanistically distinct behaviors as separate concepts even if superficially related
- Do NOT introduce domain terminology (reward hacking, principal-agent, tractability bias, instrumental convergence) unless the source uses or clearly implies it — flag all such imports as "inferred:"
- Scale concept count to source density: a short source (~100 words) should yield ${DP.concepts} concepts MAX with no padding

TEXT:
${text.slice(0, 4000)}

Return ONLY: [{"id":1,"concept":"...","technical":"max 25 words","plain":"max 20 words","importance":0.9}]`,
  }),

  triple_mode: (text, concepts) => ({
    system: "Output ONLY a raw JSON array. Each element: {concept_id, concept, provenance, decision, analogy, insight}. Max 100 words per field. No prose, no fences.",
    user: `CONCEPTS:
${JSON.stringify(concepts.map(c => ({ id: c.id, concept: c.concept })))}

SOURCE:
${text.slice(0, 2000)}

For each concept provide:
- provenance: MUST be "extracted" (uses source vocabulary or direct implication) or "inferred" (introduces external framing not in source)
- decision: when/why this applies — stay within source warrant; describe trajectory only as stated in source (max 100 words)
- analogy: metaphor for pattern recognition (max 80 words)
- insight: non-obvious implication strictly within source scope — flag as inferred if extending into domain theory (max 100 words)

Return ONLY: [{"concept_id":1,"concept":"name","provenance":"extracted|inferred","decision":"...","analogy":"...","insight":"..."}]`,
  }),

  semantic_index: (text, concepts, DP) => ({
    system: `Build a semantic taxonomy with exactly ${DP.terms} index terms and ${DP.clusters} clusters. Output ONLY a raw JSON object, no fences.`,
    user: `CONCEPTS:
${JSON.stringify(concepts.map(c => ({ id: c.id, concept: c.concept })))}

SOURCE:
${text.slice(0, 3000)}

Clustering rules:
- If concepts include a competence-gradient or task-selection-by-solvability pattern, give it its own dedicated cluster
- If concepts include observability failure, oversight gaps, or hierarchy-induced blindspots, give those a dedicated cluster separate from general vulnerability clusters
- Do not merge mechanistically distinct failure modes into a single cluster
- Inferred concepts (prefixed "inferred:") must be clustered separately from extracted concepts — never mix provenance in one cluster
- Score taxonomy TERMS only against language or clear implication present in the source text — do not include terms the source cannot support
- Flag any term that imports external domain vocabulary (e.g. reward hacking, principal-agent) with an "~" prefix to denote low source fidelity

Return ONLY: {"terms":[{"term":"...","concept_ids":[1,2]}],"clusters":[{"name":"...","concept_ids":[1,2],"description":"..."}],"density_map":{"highest":"...","lowest":"...","avg_connections":2.5}}`,
  }),

  compress_expand: (text, concepts, DP) => ({
    system: `Compress knowledge into a ${DP.thesis}-sentence thesis and ${DP.glossary}-entry glossary. Output ONLY a raw JSON object, no fences.`,
    user: `CONCEPTS:
${JSON.stringify(concepts.map(c => ({ id: c.id, concept: c.concept, importance: c.importance })))}

SOURCE:
${text.slice(0, 3000)}

THESIS RULES — strictly enforced:
- Every clause in the thesis must be traceable to a direct phrase or unambiguous implication in SOURCE
- Do NOT introduce entrenchment, compounding, resistance-to-correction, or any progressive-damage framing unless SOURCE explicitly states it
- A sparse governance signal (e.g. "left unchecked") licenses ONLY the observation that oversight is absent — do not elaborate it into a framework
- If a concept is prefixed "inferred:" it may appear in glossary but NOT in the thesis
- Thesis must be falsifiable against the source: if removing a clause would require adding new source evidence, remove it

GLOSSARY RULES:
- Nominalizations of source phrases are acceptable (e.g. "metric redefinition" from "redefine the optimization metric")
- Terms carrying external domain connotations not present in source (tractability bias, entrenchment, reward hacking) must have definition prefixed with "~inferred:"

Return ONLY: {"thesis":"...","glossary":[{"term":"...","definition":"...","concept_ids":[1]}]}`,
  }),

  meta_analysis: (text, all, DP) => ({
    system: "You are a rigorous pipeline auditor with strict audit fidelity. Score HIGH when the pipeline captured the source thoroughly. Penalize conflation of inferred theoretical extensions with directly extracted content. Output ONLY a raw JSON object, no fences. emergence_assessment must be a plain string.",
    user: `SOURCE (~${text.trim().split(/\s+/).length} words):\n${text.slice(0, 2000)}\n\nCONCEPTS EXTRACTED: ${toArr(all.concepts).length} (target was ${DP.concepts})\nINFERRED CONCEPTS (by provenance field): ${toArr(all.elaborations).filter(e => e.provenance === "inferred").length} elaborations | ${toArr(all.concepts).filter(c => toStr(c.technical).startsWith("inferred:")).length} concepts\nTAXONOMY TERMS: ${all.taxonomy?.terms?.length || 0} | CLUSTERS: ${all.taxonomy?.clusters?.length || 0}\nTHESIS: ${toStr(all.compression?.thesis)}\nELABORATIONS: ${toArr(all.elaborations).length}\n\nSource-fidelity audit rules:\n- Penalize concepts that import domain terminology not in source (reward hacking, principal-agent, tractability bias, etc.)\n- Penalize taxonomy terms that cannot be traced to source language or clear implication\n- Penalize over-expansion of sparse governance signals (e.g. "left unchecked") into full frameworks\n- Reward accurate provenance tagging (extracted: vs inferred:)\n- coverage_score reflects source fidelity, not concept volume — fewer accurate concepts scores higher than many inflated ones\n\nReturn ONLY: {"complexity_profile":{"conceptual_density":0.0,"technical_depth":0.0,"abstraction_level":0.0,"interconnectedness":0.0},"emergence_assessment":"...","blind_spots":["..."],"coverage_score":0.0,"recommendations":["..."]}`,
  }),

  skill_forge: (text, all, DP) => {
    const elaborations = toArr(all.elaborations).slice(0, 12).map(e => ({
      concept_id: e.concept_id, concept: e.concept || "",
      decision: (e.decision || "").slice(0, 200),
      analogy: (e.analogy || "").slice(0, 200),
      insight: (e.insight || "").slice(0, 200),
    }));
    const thesis = toStr(all.compression?.thesis);
    const emergence = toStr(all.meta?.emergence_assessment);
    const blindSpots = toArr(all.meta?.blind_spots);
    const recommendations = toArr(all.meta?.recommendations);

    return {
      system: "You are a Skill File compiler with strict source fidelity. Output two parts: first a raw JSON object for metadata (no fences), then Python code blocks wrapped in exact <<<SENTINEL>>> tags. No markdown fences. No extra prose. Do not present inferred theoretical extensions as extracted facts — mark them as implications.",
      user: `Analyze and synthesize the following pipeline outputs into creative skill primitives.

SOURCE:
${text.slice(0, 2000)}

THESIS: ${thesis}
EMERGENCE: ${emergence}
BLIND SPOTS: ${JSON.stringify(blindSpots)}
RECOMMENDATIONS: ${JSON.stringify(recommendations)}
ELABORATIONS: ${JSON.stringify(elaborations)}

Return your response in two parts:

PART 1 — output this JSON object (no fences):
{
  "skill_name": "2-4 word skill name",
  "trigger": "~80 token description of when to trigger this skill",
  "overview": "2-3 paragraph narrative: what this skill does, why it matters, when to reach for it",
  "when_to_use": ["specific trigger condition 1", "specific trigger condition 2", "specific trigger condition 3"],
  "workflow": ["step 1", "step 2", "step 3", "step 4"],
  "key_patterns": [{"name": "pattern name", "insight": "2-4 sentence insight"}]
}

PART 2 — output BOTH code blocks using these exact sentinels (no backticks, no fences):

<<<IMPLEMENTATION>>>
# typed, runnable Python implementing the key concepts — full function with type hints and comments
<<<END_IMPLEMENTATION>>>

<<<CHEATSHEET>>>
# minimal self-contained Python cheat-sheet, 5-15 lines, copy-paste ready
<<<END_CHEATSHEET>>>`,
    };
  },
};

// ═══════════════════════════════════════════════════════════════
//  RENDERERS
// ═══════════════════════════════════════════════════════════════

const Badge = ({ children, color }) => (
  <span style={{ fontSize: 9, fontWeight: 700, padding: "2px 7px", borderRadius: 10, background: `${color}20`, color, border: `1px solid ${color}44` }}>{children}</span>
);

const Bar = ({ value, color, label }) => {
  const num = typeof value === "number" ? value : parseFloat(value);
  if (isNaN(num)) return null;
  return (
    <div style={{ marginBottom: 8 }}>
      <div style={{ display: "flex", justifyContent: "space-between", fontSize: 10, marginBottom: 3 }}>
        <span style={{ color: "#94a3b8" }}>{label}</span>
        <span style={{ color, fontWeight: 600 }}>{Math.round(num * 100)}%</span>
      </div>
      <div style={{ height: 5, borderRadius: 3, background: "rgba(255,255,255,0.06)" }}>
        <div style={{ height: "100%", borderRadius: 3, width: `${Math.min(num * 100, 100)}%`, background: `linear-gradient(90deg, ${color}, ${color}aa)`, transition: "width 0.5s" }} />
      </div>
    </div>
  );
};

function ReportDualExtract({ data, color }) {
  const items = toArr(data);
  if (!items.length) return <div style={{ color: "#555", fontSize: 11 }}>No concepts extracted</div>;
  return (
    <div>
      <div style={{ fontSize: 11, fontWeight: 700, color, marginBottom: 8 }}>⚗️ {items.length} Concepts Extracted</div>
      {items.map((c, i) => (
        <div key={i} style={{ marginBottom: 10, padding: "8px 10px", borderRadius: 6, background: "rgba(255,255,255,0.02)", borderLeft: `3px solid ${color}` }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 4 }}>
            <span style={{ fontSize: 12, fontWeight: 700, color: "#e2e8f0" }}>{c.concept || `Concept ${c.id || i + 1}`}</span>
            {c.importance != null && <Badge color={parseFloat(c.importance) > 0.8 ? "#f59e0b" : parseFloat(c.importance) > 0.5 ? "#a78bfa" : "#64748b"}>{Math.round(parseFloat(c.importance) * 100)}%</Badge>}
          </div>
          <div style={{ fontSize: 10, color: "#22d3ee", marginBottom: 3, lineHeight: 1.4 }}>
            <span style={{ fontWeight: 600, color: "#22d3ee99" }}>Technical:</span> {toStr(c.technical)}
            {toStr(c.technical).startsWith("inferred:") && <Badge color="#f472b6" style={{marginLeft:4}}>inferred</Badge>}
            {toStr(c.technical).startsWith("extracted:") && <Badge color="#34d399" style={{marginLeft:4}}>extracted</Badge>}
          </div>
          <div style={{ fontSize: 10, color: "#94a3b8", lineHeight: 1.4 }}><span style={{ fontWeight: 600, color: "#94a3b899" }}>Plain:</span> {toStr(c.plain)}</div>
        </div>
      ))}
    </div>
  );
}

function ReportTripleMode({ data, color }) {
  const items = toArr(data);
  if (!items.length) return <div style={{ color: "#555", fontSize: 11 }}>No elaborations</div>;
  return (
    <div>
      <div style={{ fontSize: 11, fontWeight: 700, color, marginBottom: 8 }}>🔬 {items.length} Triple-Mode Elaborations</div>
      {items.map((el, i) => (
        <div key={i} style={{ marginBottom: 10, padding: "8px 10px", borderRadius: 6, background: "rgba(255,255,255,0.02)", borderLeft: `3px solid ${color}` }}>
          <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 6 }}>
            <div style={{ fontSize: 10, fontWeight: 700, color: "#e2e8f0" }}>#{el.concept_id || i + 1}{el.concept ? ` — ${el.concept}` : ""}</div>
            {el.provenance && <Badge color={el.provenance === "extracted" ? "#34d399" : "#f472b6"}>{el.provenance}</Badge>}
          </div>
          {el.decision && <div style={{ marginBottom: 6 }}><div style={{ fontSize: 9, fontWeight: 700, color: "#34d399", marginBottom: 2 }}>🎯 DECISION LOGIC</div><div style={{ fontSize: 10, color: "#a7f3d0", lineHeight: 1.4 }}>{toStr(el.decision)}</div></div>}
          {el.analogy  && <div style={{ marginBottom: 6 }}><div style={{ fontSize: 9, fontWeight: 700, color: "#a78bfa", marginBottom: 2 }}>🎭 ANALOGY</div><div style={{ fontSize: 10, color: "#c4b5fd", lineHeight: 1.4, fontStyle: "italic" }}>{toStr(el.analogy)}</div></div>}
          {el.insight  && <div>                           <div style={{ fontSize: 9, fontWeight: 700, color: "#f59e0b", marginBottom: 2 }}>💡 INSIGHT</div><div style={{ fontSize: 10, color: "#fcd34d", lineHeight: 1.4 }}>{toStr(el.insight)}</div></div>}
        </div>
      ))}
    </div>
  );
}

function ReportSemanticIndex({ data, color }) {
  if (!data || (!data.terms && !data.clusters)) return <div style={{ color: "#555", fontSize: 11 }}>No taxonomy data</div>;
  return (
    <div>
      <div style={{ fontSize: 11, fontWeight: 700, color, marginBottom: 8 }}>🗂️ Semantic Taxonomy</div>
      {Array.isArray(data.terms) && data.terms.length > 0 && (
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
      {Array.isArray(data.clusters) && data.clusters.length > 0 && (
        <div style={{ marginBottom: 12 }}>
          <div style={{ fontSize: 10, fontWeight: 700, color: "#94a3b8", marginBottom: 6 }}>CLUSTERS ({data.clusters.length})</div>
          {data.clusters.map((cl, i) => (
            <div key={i} style={{ marginBottom: 8, padding: "6px 10px", borderRadius: 6, background: "rgba(255,255,255,0.02)", borderLeft: `3px solid ${color}` }}>
              <div style={{ fontSize: 11, fontWeight: 700, color: "#e2e8f0" }}>{cl.name}</div>
              <div style={{ fontSize: 10, color: "#94a3b8", marginTop: 2, lineHeight: 1.4 }}>{toStr(cl.description)}</div>
              <div style={{ fontSize: 9, color: "#64748b", marginTop: 3 }}>Concepts: [{(cl.concept_ids || []).join(", ")}]</div>
            </div>
          ))}
        </div>
      )}
      {data.density_map && (
        <div style={{ padding: "8px 10px", borderRadius: 6, background: "rgba(167,139,250,0.06)", border: "1px solid rgba(167,139,250,0.15)" }}>
          <div style={{ fontSize: 10, fontWeight: 700, color: "#94a3b8", marginBottom: 4 }}>DENSITY MAP</div>
          <div style={{ fontSize: 10, color: "#c4b5fd" }}>Highest: {toStr(data.density_map.highest)}</div>
          <div style={{ fontSize: 10, color: "#94a3b8" }}>Lowest: {toStr(data.density_map.lowest)}</div>
          <div style={{ fontSize: 10, color: "#64748b" }}>Avg connections: {data.density_map.avg_connections}</div>
        </div>
      )}
    </div>
  );
}

function ReportCompressExpand({ data, color }) {
  if (!data || (!data.thesis && !data.glossary)) return <div style={{ color: "#555", fontSize: 11 }}>No compression data</div>;
  return (
    <div>
      <div style={{ fontSize: 11, fontWeight: 700, color, marginBottom: 8 }}>💎 Compress / Expand</div>
      {data.thesis && (
        <div style={{ marginBottom: 12, padding: "10px 12px", borderRadius: 8, background: "rgba(244,114,182,0.06)", border: "1px solid rgba(244,114,182,0.2)" }}>
          <div style={{ fontSize: 10, fontWeight: 700, color: "#f472b6", marginBottom: 6 }}>CORE THESIS</div>
          <div style={{ fontSize: 12, color: "#fda4af", lineHeight: 1.6, fontStyle: "italic" }}>"{toStr(data.thesis)}"</div>
        </div>
      )}
      {Array.isArray(data.glossary) && data.glossary.length > 0 && (
        <div>
          <div style={{ fontSize: 10, fontWeight: 700, color: "#94a3b8", marginBottom: 6 }}>GLOSSARY ({data.glossary.length} entries)</div>
          {data.glossary.map((g, i) => (
            <div key={i} style={{ marginBottom: 6, padding: "6px 10px", borderRadius: 6, background: "rgba(255,255,255,0.02)", borderLeft: `3px solid ${color}` }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                <span style={{ fontSize: 11, fontWeight: 700, color: "#e2e8f0" }}>{g.term}</span>
                <span style={{ fontSize: 8, color: "#64748b" }}>→ [{(g.concept_ids || []).join(", ")}]</span>
              </div>
              <div style={{ fontSize: 10, color: "#94a3b8", marginTop: 2, lineHeight: 1.4 }}>{toStr(g.definition)}</div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function ReportMetaAnalysis({ data, color }) {
  if (!data) return <div style={{ color: "#555", fontSize: 11 }}>No meta-analysis data</div>;
  return (
    <div>
      <div style={{ fontSize: 11, fontWeight: 700, color, marginBottom: 8 }}>📡 Meta-Analysis</div>
      {data.complexity_profile && typeof data.complexity_profile === "object" && (
        <div style={{ marginBottom: 12 }}>
          <div style={{ fontSize: 10, fontWeight: 700, color: "#94a3b8", marginBottom: 6 }}>COMPLEXITY PROFILE</div>
          {Object.entries(data.complexity_profile).map(([k, v]) => <Bar key={k} value={v} color={color} label={k.replace(/_/g, " ")} />)}
        </div>
      )}
      {data.coverage_score != null && (
        <div style={{ marginBottom: 12, padding: 10, borderRadius: 8, background: `${color}0d`, border: `1px solid ${color}25`, textAlign: "center" }}>
          <div style={{ fontSize: 9, color: "#64748b" }}>Coverage</div>
          <div style={{ fontSize: 26, fontWeight: 700, color }}>{Math.round(parseFloat(data.coverage_score) * 100)}%</div>
        </div>
      )}
      {data.emergence_assessment && (
        <div style={{ marginBottom: 12 }}>
          <div style={{ fontSize: 10, fontWeight: 700, color: "#94a3b8", marginBottom: 4 }}>EMERGENCE</div>
          <div style={{ fontSize: 11, color: "#94a3b8", lineHeight: 1.5 }}>{toStr(data.emergence_assessment)}</div>
        </div>
      )}
      {Array.isArray(data.blind_spots) && data.blind_spots.length > 0 && (
        <div style={{ marginBottom: 12 }}>
          <div style={{ fontSize: 10, fontWeight: 700, color: "#f472b6", marginBottom: 4 }}>BLIND SPOTS</div>
          {data.blind_spots.map((b, i) => <div key={i} style={{ fontSize: 10, color: "#fda4af", padding: "3px 0" }}>⚠️ {toStr(b)}</div>)}
        </div>
      )}
      {Array.isArray(data.recommendations) && data.recommendations.length > 0 && (
        <div>
          <div style={{ fontSize: 10, fontWeight: 700, color: "#22d3ee", marginBottom: 4 }}>RECOMMENDATIONS</div>
          {data.recommendations.map((r, i) => <div key={i} style={{ fontSize: 10, color: "#67e8f9", padding: "3px 0" }}>→ {toStr(r)}</div>)}
        </div>
      )}
    </div>
  );
}

function ReportSkillForge({ data, color }) {
  const md = typeof data === "string" ? data : "";
  if (!md) return <div style={{ color: "#555", fontSize: 11 }}>No skill data</div>;
  const name = (md.match(/^#\s+(.+)/m) || [])[1] || "Skill File";
  return (
    <div>
      <div style={{ fontSize: 11, fontWeight: 700, color, marginBottom: 4 }}>🛠️ Skill Forge Output</div>
      <div style={{ padding: "8px 10px", borderRadius: 6, background: "rgba(251,146,60,0.06)", border: "1px solid rgba(251,146,60,0.2)" }}>
        <div style={{ fontSize: 14, fontWeight: 700, color: "#fb923c" }}>{name}</div>
        <div style={{ fontSize: 10, color: "#94a3b8", marginTop: 4 }}>{md.split("\n").length} lines — see right panel</div>
      </div>
    </div>
  );
}

const RENDERERS = {
  dual_extract: ReportDualExtract,
  triple_mode: ReportTripleMode,
  semantic_index: ReportSemanticIndex,
  compress_expand: ReportCompressExpand,
  meta_analysis: ReportMetaAnalysis,
  skill_forge: ReportSkillForge,
};

function StageReport({ data, stageId, color }) {
  const [showRaw, setShowRaw] = useState(false);
  const Renderer = RENDERERS[stageId];
  if (!Renderer) return null;
  return (
    <div>
      <Renderer data={data} color={color} />
      <div style={{ marginTop: 10, borderTop: "1px solid rgba(255,255,255,0.06)", paddingTop: 8 }}>
        <div onClick={() => setShowRaw(p => !p)} style={{ fontSize: 9, fontWeight: 700, color: "#64748b", cursor: "pointer", display: "flex", alignItems: "center", gap: 4, userSelect: "none" }}>
          <span style={{ display: "inline-block", transition: "transform 0.2s", transform: showRaw ? "rotate(90deg)" : "rotate(0deg)" }}>▶</span>
          RAW JSON
        </div>
        {showRaw && (
          <div style={{ marginTop: 6, padding: 8, background: "rgba(0,0,0,0.4)", borderRadius: 6, maxHeight: 300, overflow: "auto" }}>
            <pre style={{ margin: 0, whiteSpace: "pre-wrap", wordBreak: "break-word", color: "#94a3b8", fontSize: 9, lineHeight: 1.5, fontFamily: "'Courier New', monospace" }}>
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
  root:   { minHeight: "100vh", background: "linear-gradient(135deg, #0a0a1a 0%, #0d1117 50%, #0a0a1a 100%)", color: "#e2e8f0", fontFamily: "-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif", fontSize: 13 },
  header: { padding: "12px 24px 10px", borderBottom: "1px solid rgba(255,255,255,0.06)", background: "linear-gradient(180deg, rgba(251,146,60,0.08) 0%, transparent 100%)" },
  title:  { margin: 0, fontSize: 24, fontWeight: 700, background: "linear-gradient(135deg, #f59e0b, #fb923c, #f472b6)", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent", letterSpacing: "-0.02em" },
  sub:    { margin: "4px 0 0", fontSize: 11, color: "#64748b", letterSpacing: "0.08em", textTransform: "uppercase" },
  layout: { display: "flex", height: "calc(100vh - 56px)", overflow: "hidden" },
  left:   { width: 380, minWidth: 380, borderRight: "1px solid rgba(255,255,255,0.06)", display: "flex", flexDirection: "column", overflowY: "auto" },
  center: { flex: 1, display: "flex", flexDirection: "column", overflowY: "auto" },
  right:  { width: 520, minWidth: 520, borderLeft: "1px solid rgba(255,255,255,0.06)", display: "flex", flexDirection: "column" },
  sec:    { padding: "14px 18px", borderBottom: "1px solid rgba(255,255,255,0.04)" },
  lbl:    { fontSize: 11, fontWeight: 600, letterSpacing: "0.1em", textTransform: "uppercase", marginBottom: 8 },
  ta:     { width: "100%", minHeight: 260, background: "rgba(255,255,255,0.03)", border: "1px solid rgba(255,255,255,0.08)", borderRadius: 8, color: "#e2e8f0", padding: "10px 12px", fontSize: 12, fontFamily: "inherit", resize: "vertical", outline: "none", boxSizing: "border-box" },
  btn:    (c, off) => ({ padding: "10px 0", width: "100%", border: "none", borderRadius: 8, fontWeight: 700, fontSize: 13, cursor: off ? "default" : "pointer", color: "#000", background: off ? "#333" : `linear-gradient(135deg, ${c}, ${c}dd)`, opacity: off ? 0.4 : 1, transition: "all 0.2s" }),
  card:   (c, st) => ({ padding: "10px 14px", borderRadius: 8, border: `1px solid ${st === "running" ? c : st === "complete" ? c + "66" : "rgba(255,255,255,0.06)"}`, background: st === "running" ? `${c}0d` : st === "complete" ? `${c}08` : "rgba(255,255,255,0.02)", transition: "all 0.3s", marginBottom: 6, cursor: st === "complete" ? "pointer" : "default" }),
};

// ═══════════════════════════════════════════════════════════════
//  MAIN COMPONENT
// ═══════════════════════════════════════════════════════════════

export default function PhilosophersStone() {
  const [input,    setInput]    = useState("");
  const [selected, setSelected] = useState(new Set(STAGES.map(s => s.id)));
  const [status,   setStatus]   = useState({});
  const [results,  setResults]  = useState({});
  const [running,  setRunning]  = useState(false);
  const [error,    setError]    = useState(null);
  const [log,      setLog]      = useState([]);
  const [expanded, setExpanded] = useState(null);
  const [copied,   setCopied]   = useState(false);
  const logRef     = useRef(null);
  const startTime  = useRef(null);

  const addLog = useCallback((msg, type = "info") => {
    setLog(p => [...p, { t: new Date().toLocaleTimeString(), msg, type }]);
  }, []);

  useEffect(() => {
    if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight;
  }, [log]);

  const toggle = (id) => {
    if (running) return;
    setSelected(prev => {
      const next = new Set(prev);
      if (next.has(id)) { next.delete(id); resolveDependents(id).forEach(d => next.delete(d)); }
      else { resolveDeps(id).forEach(d => next.add(d)); }
      return next;
    });
  };

  const runPipeline = useCallback(async () => {
    if (!input.trim() || running) return;
    setRunning(true); setError(null); setResults({}); setLog([]); setExpanded(null); setCopied(false);
    startTime.current = Date.now();
    const st = {};
    STAGES.forEach(s => { st[s.id] = selected.has(s.id) ? "pending" : "skipped"; });
    setStatus(st);
    addLog(`Pipeline initiated`);

    const all = {};
    const DP = computeDP(input);
    addLog(`📐 Source: ~${input.trim().split(/\s+/).length} words → ${DP.concepts} concepts · ${DP.terms} terms · ${DP.clusters} clusters`);

    try {
      // ── 1. Dual Extract ───────────────────────────────────
      if (selected.has("dual_extract")) {
        setStatus(p => ({ ...p, dual_extract: "running" }));
        addLog("⚗️ Extracting dual-register concepts...");
        try {
          const pr = PROMPTS.dual_extract(input, DP);
          const raw = await callAPI(pr.system, pr.user, 6000);
          all.concepts = extractArray(raw);
          setResults(p => ({ ...p, dual_extract: all.concepts }));
          addLog(`⚗️ Extracted ${all.concepts.length} concepts`, "success");
        } catch (e) {
          addLog(`⚗️ Dual Extract error: ${e.message}`, "error");
          all.concepts = [];
        }
        setStatus(p => ({ ...p, dual_extract: "complete" }));
      } else { all.concepts = []; }

      // ── 2+3+4 kick off (all parallel, awaited below) ─────
      const tripleP = selected.has("triple_mode") && all.concepts.length ? (async () => {
        setStatus(p => ({ ...p, triple_mode: "running" }));
        addLog("🔬 Elaborating triple modes...");
        try {
          const top = [...all.concepts].sort((a, b) => (b.importance || 0) - (a.importance || 0)).slice(0, 17);
          const pr = PROMPTS.triple_mode(input, top);
          const raw = await callAPI(pr.system, pr.user, 4000);
          all.elaborations = extractArray(raw);
          setResults(p => ({ ...p, triple_mode: all.elaborations }));
          addLog(`🔬 Triple-mode: ${all.elaborations.length} elaborations`, "success");
        } catch (e) {
          addLog(`🔬 Triple-mode error: ${e.message}`, "error");
          all.elaborations = [];
        }
        setStatus(p => ({ ...p, triple_mode: "complete" }));
      })() : Promise.resolve();

      const semP = selected.has("semantic_index") && all.concepts.length ? (async () => {
        setStatus(p => ({ ...p, semantic_index: "running" }));
        addLog("🗂️ Building semantic taxonomy...");
        try {
          const pr = PROMPTS.semantic_index(input, all.concepts, DP);
          const raw = await callAPI(pr.system, pr.user, 4000);
          all.taxonomy = extractObject(raw) || {};
          setResults(p => ({ ...p, semantic_index: all.taxonomy }));
          addLog(`🗂️ Taxonomy: ${all.taxonomy.terms?.length || 0} terms, ${all.taxonomy.clusters?.length || 0} clusters`, "success");
        } catch (e) {
          addLog(`🗂️ Semantic Index error: ${e.message}`, "error");
          all.taxonomy = {};
        }
        setStatus(p => ({ ...p, semantic_index: "complete" }));
      })() : Promise.resolve();

      // ── 2+3+4. Triple Mode + Semantic Index + Compress/Expand (parallel) ──
      const compressP = selected.has("compress_expand") && all.concepts.length ? (async () => {
        setStatus(p => ({ ...p, compress_expand: "running" }));
        addLog("💎 Compressing and expanding...");
        try {
          const pr = PROMPTS.compress_expand(input, all.concepts, DP);
          const raw = await callAPI(pr.system, pr.user, 4000);
          all.compression = extractObject(raw) || {};
          setResults(p => ({ ...p, compress_expand: all.compression }));
          addLog(`💎 Thesis compressed, ${all.compression.glossary?.length || 0} glossary entries`, "success");
        } catch (e) {
          addLog(`💎 Compress/Expand error: ${e.message}`, "error");
          all.compression = {};
        }
        setStatus(p => ({ ...p, compress_expand: "complete" }));
      })() : Promise.resolve();

      await Promise.all([tripleP, semP, compressP]);

      // ── 5+6. Meta-Analysis + Skill Forge (parallel) ───────
      const metaP = selected.has("meta_analysis") ? (async () => {
        setStatus(p => ({ ...p, meta_analysis: "running" }));
        addLog("📡 Running meta-analysis...");
        try {
          const pr = PROMPTS.meta_analysis(input, all, DP);
          const raw = await callAPI(pr.system, pr.user, 4000);
          all.meta = extractObject(raw) || {};
          setResults(p => ({ ...p, meta_analysis: all.meta }));
          const cov = all.meta.coverage_score != null ? `${Math.round(parseFloat(all.meta.coverage_score) * 100)}%` : "?";
          addLog(`📡 Coverage: ${cov}`, "success");
        } catch (e) {
          addLog(`📡 Meta-Analysis error: ${e.message}`, "error");
          all.meta = {};
        }
        setStatus(p => ({ ...p, meta_analysis: "complete" }));
      })() : Promise.resolve();

      const forgeP = selected.has("skill_forge") && all.concepts.length ? (async () => {
        setStatus(p => ({ ...p, skill_forge: "running" }));
        addLog("🛠️ Compressing and forging final SKILL.md layout...");
        try {
          const pr = PROMPTS.skill_forge(input, all, DP);
          const raw = await callAPI(pr.system, pr.user, 4000);

          // Extract code via sentinels — immune to quote/backslash corruption
          const grabSentinel = (tag, txt) => {
            const open = `<<<${tag}>>>`;
            const close = `<<<END_${tag}>>>`;
            const s = txt.indexOf(open);
            const e = txt.indexOf(close);
            return s !== -1 && e > s ? txt.slice(s + open.length, e).trim() : null;
          };
          const implCode  = grabSentinel("IMPLEMENTATION", raw);
          const cheatCode = grabSentinel("CHEATSHEET", raw);

          // Parse JSON from the part before the first sentinel
          const jsonPart = raw.slice(0, raw.indexOf("<<<") === -1 ? raw.length : raw.indexOf("<<<"));
          const forgeData = extractObject(jsonPart) || {};
          forgeData.implementation_code = implCode || forgeData.implementation_code || "# No implementation generated";
          forgeData.cheat_sheet_code    = cheatCode || implCode || forgeData.cheat_sheet_code || "# No cheat sheet generated";

          // ── Deterministic local assembly ──────────────────────
          const thesis    = toStr(all.compression?.thesis);
          const concepts  = toArr(all.concepts).slice(0, DP.concepts);
          const glossary = toArr(all.compression?.glossary).slice(0, DP.glossary);
          const elaborations = toArr(all.elaborations);

          const conceptRows = concepts.map(c => {
            const imp = c.importance != null ? `${Math.round(parseFloat(c.importance) * 100)}%` : "—";
            return `| ${toStr(c.concept)} | ${toStr(c.technical).replace(/\|/g, "&#124;").slice(0, 200)} | ${toStr(c.plain).replace(/\|/g, "&#124;").slice(0, 200)} | ${imp} |`;
          }).join("\n");

          const glossaryRows = glossary.map(g =>
            `| ${toStr(g.term)} | ${toStr(g.definition).replace(/\|/g, "&#124;").slice(0, 200)} | ${(g.concept_ids || []).join(", ")} |`
          ).join("\n");

          const tripleModeBlocks = elaborations.map(e =>
            `### ${toStr(e.concept)}\n**🎯 Decision:** ${toStr(e.decision)}\n**🎭 Analogy:** ${toStr(e.analogy)}\n**💡 Insight:** ${toStr(e.insight)}`
          ).join("\n\n");

          const keyPatterns = toArr(forgeData.key_patterns).map(p =>
            `### ${toStr(p.name)}\n${toStr(p.insight)}`
          ).join("\n\n");

          const blindSpots = toArr(all.meta?.blind_spots).map(b => `- ⚠️ ${toStr(b)}`).join("\n");
          const recs       = toArr(all.meta?.recommendations).map(r => `- 🔧 ${toStr(r)}`).join("\n");
          const emergence  = toStr(all.meta?.emergence_assessment);
          const workflow = toArr(forgeData.workflow).map((s, i) => `${i + 1}. ${s}`).join("\n");

          const finalMarkdown = `# ${forgeData.skill_name || "Untitled Skill"}

> ${forgeData.trigger || ""}

## Core Thesis
${thesis}

## Overview
${forgeData.overview || ""}

## When to Use
${toArr(forgeData.when_to_use).map(w => `- ${w}`).join("\n") || "- See Core Workflow below"}

## Core Workflow
${workflow}

## Key Patterns
${keyPatterns}

## Code Implementation
\`\`\`python
${forgeData.implementation_code || "# No implementation generated"}
\`\`\`

## Triple-Mode Insights
${tripleModeBlocks}

## Concept Reference
| Concept | Technical | Plain | Importance |
|---------|-----------|-------|------------|
${conceptRows}

## Glossary
| Term | Definition | Concept IDs |
|------|------------|-------------|
${glossaryRows}

## Edge Cases & Warnings
${blindSpots}

## Emergence Assessment
${emergence}

## Recommendations
${recs}

## Quick Reference
\`\`\`python
${forgeData.cheat_sheet_code || forgeData.implementation_code || "# No cheat sheet generated"}
\`\`\`

---
_Generated by Philosopher's Stone v5 — EchoSeed_`;

          all.skill = finalMarkdown;
          setResults(p => ({ ...p, skill_forge: finalMarkdown }));
          addLog(`🛠️ Skill forged instantly: ${forgeData.skill_name}`, "success");
        } catch (e) {
          addLog(`🛠️ Skill Forge error: ${e.message}`, "error");
        }
        setStatus(p => ({ ...p, skill_forge: "complete" }));
      })() : Promise.resolve();

      await Promise.all([metaP, forgeP]);

      const elapsed = ((Date.now() - startTime.current) / 1000).toFixed(1);
      addLog(`✅ Pipeline complete in ${elapsed}s`, "success");
    } catch (e) {
      setError(e.message);
      addLog(`❌ Error: ${e.message}`, "error");
    }
    setRunning(false);
  }, [input, selected, running, addLog]);

  useEffect(() => {
    const h = (e) => { if ((e.ctrlKey || e.metaKey) && e.key === "Enter") { e.preventDefault(); runPipeline(); } };
    window.addEventListener("keydown", h);
    return () => window.removeEventListener("keydown", h);
  }, [runPipeline]);

  const copySkill = () => {
    const md = typeof results.skill_forge === "string" ? results.skill_forge : "";
    if (md) { navigator.clipboard.writeText(md); setCopied(true); setTimeout(() => setCopied(false), 2000); }
  };

  const completedCount = Object.values(status).filter(s => s === "complete").length;
  const totalSelected  = [...selected].length;

  return (
    <div style={S.root}>
      {/* Header */}
      <div style={S.header}>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
          <div>
            <h1 style={S.title}>⚗️ Philosopher's Stone</h1>
            <p style={S.sub}>v5 — AI Agent Markdown Forge · EchoSeed</p>
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
        @keyframes pulse   { from { opacity: 0.4; } to { opacity: 1; } }
        @keyframes shimmer { from { background-position: -200% 0; } to { background-position: 200% 0; } }
        ::-webkit-scrollbar { width: 7px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.15); border-radius: 4px; }
        ::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.25); }
        textarea { transition: border-color 0.2s, box-shadow 0.2s; }
        textarea:focus { border-color: rgba(245,158,11,0.5) !important; box-shadow: 0 0 0 2px rgba(245,158,11,0.1); }
        .stage-row:hover { background: rgba(255,255,255,0.04) !important; }
        .stage-row.active:hover { filter: brightness(1.15); }
        .run-btn:hover:not(:disabled) { filter: brightness(1.2); transform: translateY(-1px); box-shadow: 0 4px 16px rgba(245,158,11,0.3); }
        .run-btn:active:not(:disabled) { transform: translateY(0); }
        .copy-btn:hover { background: #64748b !important; }
        * { box-sizing: border-box; }
      `}</style>

      <div style={S.layout}>
        {/* ═══ LEFT ═══ */}
        <div style={S.left}>
          <div style={S.sec}>
            <div style={{ ...S.lbl, color: "#f59e0b" }}>Source Material</div>
            <textarea
              style={S.ta}
              placeholder={"Paste any text — research, docs, specs, conversations, code…\n\nThe Stone transmutes raw knowledge into structured SKILL.md files for AI agents."}
              value={input}
              onChange={e => setInput(e.target.value)}
              disabled={running}
            />
            <div style={{ display: "flex", justifyContent: "space-between", marginTop: 6 }}>
              <span style={{ fontSize: 10, color: "#475569" }}>{input.length > 0 ? `${input.length.toLocaleString()} chars` : ""}</span>
              {input.length > 0 && !running && <span style={{ fontSize: 10, color: "#64748b", cursor: "pointer" }} onClick={() => setInput("")}>Clear</span>}
            </div>
          </div>

          <div style={S.sec}>
            <div style={{ ...S.lbl, color: "#22d3ee" }}>Pipeline Stages</div>
            {STAGES.map(stage => {
              const active = selected.has(stage.id);
              return (
                <div
                  key={stage.id}
                  onClick={() => toggle(stage.id)}
                  className={`stage-row${active ? " active" : ""}`}
                  style={{ display: "flex", alignItems: "center", gap: 10, padding: "9px 12px", borderRadius: 6, marginBottom: 4, cursor: running ? "default" : "pointer", background: active ? `${stage.color}0a` : "transparent", border: `1px solid ${active ? stage.color + "33" : "transparent"}`, transition: "all 0.2s" }}
                >
                  <div style={{ width: 16, height: 16, borderRadius: 4, display: "flex", alignItems: "center", justifyContent: "center", border: `2px solid ${active ? stage.color : "#333"}`, background: active ? stage.color : "transparent", fontSize: 10, color: "#000", fontWeight: 700, flexShrink: 0 }}>{active && "✓"}</div>
                  <span style={{ fontSize: 14 }}>{stage.icon}</span>
                  <div style={{ flex: 1 }}>
                    <div style={{ fontSize: 11, fontWeight: 600, color: active ? "#e2e8f0" : "#555" }}>{stage.name}</div>
                    <div style={{ fontSize: 9, color: "#475569", marginTop: 1 }}>{stage.desc}</div>
                  </div>
                  {stage.deps.length > 0 && <div style={{ fontSize: 8, color: "#475569", background: "rgba(255,255,255,0.03)", padding: "2px 5px", borderRadius: 3, flexShrink: 0 }}>needs {stage.deps.length}</div>}
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

        {/* ═══ CENTER ═══ */}
        <div style={S.center}>
          <div style={S.sec}>
            <div style={{ ...S.lbl, color: "#34d399" }}>Pipeline Progress & Reports</div>
            {STAGES.filter(s => selected.has(s.id)).map(stage => {
              const st    = status[stage.id] || "idle";
              const isRun = st === "running";
              const isDone = st === "complete";
              const isExp = expanded === stage.id;
              return (
                <div key={stage.id}>
                  <div style={S.card(stage.color, st)} onClick={() => isDone && setExpanded(isExp ? null : stage.id)}>
                    <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
                      <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                        <span style={{ fontSize: 18 }}>{stage.icon}</span>
                        <div>
                          <div style={{ fontSize: 12, fontWeight: 600, color: isRun ? stage.color : isDone ? "#e2e8f0" : "#555" }}>{stage.name}</div>
                          <div style={{ fontSize: 9, color: "#475569" }}>{stage.desc}</div>
                        </div>
                      </div>
                      <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                        {isRun && (
                          <div style={{ width: 50, height: 3, borderRadius: 2, overflow: "hidden", background: "rgba(255,255,255,0.06)" }}>
                            <div style={{ height: "100%", borderRadius: 2, background: `linear-gradient(90deg, transparent, ${stage.color}, transparent)`, backgroundSize: "200% 100%", animation: "shimmer 1.5s infinite" }} />
                          </div>
                        )}
                        <span style={{ fontSize: 9, fontWeight: 600, padding: "3px 8px", borderRadius: 10, background: isRun ? `${stage.color}22` : isDone ? "#34d39922" : "rgba(255,255,255,0.03)", color: isRun ? stage.color : isDone ? "#34d399" : "#555" }}>
                          {isRun ? "RUNNING" : isDone ? "DONE" : st === "pending" ? "QUEUED" : "—"}
                        </span>
                        {isDone && <span style={{ fontSize: 10, color: "#475569", transition: "transform 0.2s", transform: isExp ? "rotate(90deg)" : "rotate(0deg)", display: "inline-block" }}>▶</span>}
                      </div>
                    </div>
                  </div>
                  {isExp && results[stage.id] !== undefined && (
                    <div style={{ padding: "12px 14px", marginBottom: 6, borderRadius: "0 0 8px 8px", marginTop: -6, background: "rgba(0,0,0,0.3)", border: `1px solid ${stage.color}22`, borderTop: "none", maxHeight: 500, overflow: "auto" }}>
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
            <div style={{ ...S.lbl, color: "#64748b" }}>Pipeline Log</div>
            <div ref={logRef} style={{ maxHeight: 200, overflow: "auto" }}>
              {log.length === 0 && <div style={{ fontSize: 10, color: "#333" }}>Awaiting pipeline execution...</div>}
              {log.map((l, i) => (
                <div key={i} style={{ fontSize: 11, fontFamily: "'Courier New', monospace", padding: "3px 0", color: l.type === "error" ? "#ef4444" : l.type === "success" ? "#34d399" : "#64748b", borderBottom: "1px solid rgba(255,255,255,0.02)" }}>
                  <span style={{ color: "#333", marginRight: 8 }}>{l.t}</span>{l.msg}
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* ═══ RIGHT ═══ */}
        <div style={S.right}>
          <div style={{ padding: "12px 18px", borderBottom: "1px solid rgba(255,255,255,0.06)", display: "flex", alignItems: "center", justifyContent: "space-between", flexShrink: 0 }}>
            <div style={{ fontSize: 11, fontWeight: 700, color: "#fb923c", letterSpacing: "0.06em" }}>🛠️ SKILL.MD OUTPUT</div>
            {typeof results.skill_forge === "string" && results.skill_forge.length > 0 && (
              <button className="copy-btn" onClick={copySkill} style={{ padding: "5px 14px", border: "none", borderRadius: 5, fontWeight: 700, fontSize: 11, cursor: "pointer", color: "#fff", background: copied ? "#16a34a" : "#475569", transition: "all 0.2s" }}>
                {copied ? "✓ Copied!" : "📋 Copy"}
              </button>
            )}
          </div>
          <div style={{ flex: 1, overflow: "auto", padding: "16px 20px", background: "#f8fafc" }}>
            {typeof results.skill_forge === "string" && results.skill_forge.length > 0 ? (
              <pre style={{ whiteSpace: "pre-wrap", wordBreak: "break-word", fontSize: 13, lineHeight: 1.8, color: "#0f172a", fontFamily: "'Courier New', monospace", margin: 0 }}>
                {results.skill_forge}
              </pre>
            ) : (
              <div style={{ padding: 40, textAlign: "center" }}>
                <div style={{ fontSize: 40, marginBottom: 12 }}>⚗️</div>
                <div style={{ fontSize: 12, color: "#475569" }}>SKILL.md output will render here</div>
                <div style={{ fontSize: 10, marginTop: 4, color: "#94a3b8" }}>Run pipeline with Skill Forge enabled</div>
              </div>
            )}
          </div>
          <div style={{ padding: "10px 18px", borderTop: "1px solid rgba(255,255,255,0.04)", fontSize: 9, color: "#333", textAlign: "center", flexShrink: 0 }}>
            Philosopher's Stone v5 × Skill Forge × EchoSeed
          </div>
        </div>
      </div>
    </div>
  );
}
