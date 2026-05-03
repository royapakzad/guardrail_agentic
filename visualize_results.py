"""
visualize_results.py  —  Multilingual LLM Guardrail Evaluation Dashboard
Usage: streamlit run visualize_results.py
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import altair as alt
import pandas as pd
import streamlit as st

# ── PAGE CONFIG ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Guardrail Eval Explorer",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CUSTOM CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"], .stApp, .stMarkdown, p, span, div, td, th, label {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
}

#MainMenu, footer, header { visibility: hidden; }

/* Metric cards */
[data-testid="metric-container"] {
    background: #ffffff;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 18px 20px !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}
[data-testid="stMetricValue"] {
    font-size: 1.75rem !important;
    font-weight: 800 !important;
    color: #0F172A !important;
    letter-spacing: -0.02em;
}
[data-testid="stMetricLabel"] {
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    color: #64748B !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
[data-testid="stMetricDelta"] {
    font-size: 0.82rem !important;
    font-weight: 500 !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0F2448 !important;
}
[data-testid="stSidebar"] .stMarkdown,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] div {
    color: #CBD5E1 !important;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #F1F5F9 !important;
    font-weight: 700 !important;
}
[data-testid="stSidebar"] .stSelectbox > div > div {
    background: #1E3A5F !important;
    border-color: #334155 !important;
    color: #E2E8F0 !important;
}
[data-testid="stSidebar"] hr { border-color: #1E3A5F !important; }
[data-testid="stSidebar"] .stWarning { background: #3D2B00 !important; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] { gap: 4px; border-bottom: 2px solid #E2E8F0; }
.stTabs [data-baseweb="tab"] {
    font-weight: 500;
    font-size: 0.88rem;
    color: #64748B;
    padding: 8px 16px;
    border-radius: 8px 8px 0 0;
}
.stTabs [aria-selected="true"] {
    color: #1B3A6B !important;
    font-weight: 700 !important;
    border-bottom: 2px solid #3B82F6 !important;
}

/* Headings */
h1 { font-size: 1.6rem !important; font-weight: 800 !important; color: #0F172A !important; letter-spacing: -0.03em; }
h2 { font-size: 1.25rem !important; font-weight: 700 !important; color: #1E293B !important; }
h3 { font-size: 1.05rem !important; font-weight: 600 !important; color: #334155 !important; }

/* Data tables */
[data-testid="stDataFrame"] th {
    font-weight: 600 !important;
    font-size: 0.75rem !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #475569 !important;
    background: #F8FAFC !important;
}

/* Badges */
.badge-pass {
    display: inline-block;
    background: #DCFCE7; color: #166534;
    font-weight: 700; font-size: 0.7rem;
    padding: 2px 9px; border-radius: 999px; letter-spacing: 0.04em;
}
.badge-fail {
    display: inline-block;
    background: #FEE2E2; color: #991B1B;
    font-weight: 700; font-size: 0.7rem;
    padding: 2px 9px; border-radius: 999px; letter-spacing: 0.04em;
}

/* Score big display */
.score-display {
    font-size: 3.2rem;
    font-weight: 800;
    line-height: 1;
    letter-spacing: -0.04em;
}

/* Alert banner */
.alert-changed {
    background: #FFF7ED;
    border: 1px solid #FDBA74;
    border-left: 4px solid #F97316;
    border-radius: 8px;
    padding: 12px 16px;
    color: #7C2D12;
    font-weight: 500;
    margin: 8px 0 16px 0;
}

/* Info card */
.info-card {
    background: #F0F9FF;
    border: 1px solid #BAE6FD;
    border-radius: 10px;
    padding: 14px 18px;
    margin: 8px 0;
    font-size: 0.88rem;
    color: #0C4A6E;
}

/* Divider */
.custom-hr { height: 1px; background: #E2E8F0; margin: 24px 0; border: none; }

/* Caption text */
.caption-text { font-size: 0.78rem; color: #94A3B8; line-height: 1.5; }

/* Section label */
.section-label {
    font-size: 0.72rem; font-weight: 600; text-transform: uppercase;
    letter-spacing: 0.08em; color: #94A3B8; margin-bottom: 8px;
}
</style>
""", unsafe_allow_html=True)

# ── CONSTANTS ──────────────────────────────────────────────────────────────────
_VALID_THRESHOLD = 0.6

_GUARDRAIL_DISPLAY = {
    "anthropic:claude-sonnet-4-6": "Claude Sonnet 4.6",
    "anthropic:claude-sonnet-4-5": "Claude Sonnet 4.5",
    "anthropic:claude-opus-4":     "Claude Opus 4",
    "gemini:gemini-2.5-flash":     "Gemini 2.5 Flash",
    "gemini:gemini-2.0-flash":     "Gemini 2.0 Flash",
    "openai:gpt-5-nano":           "GPT-5-nano",
    "openai:gpt-4o-mini":          "GPT-4o-mini",
    "openai:gpt-4o":               "GPT-4o",
}

_MODEL_COLORS = {
    "claude": "#7C3AED",   # purple
    "gemini": "#1A73E8",   # Google blue
    "gpt":    "#10A37F",   # OpenAI green
}


# ── HELPERS ────────────────────────────────────────────────────────────────────

def load_json(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def pretty_guardrail(raw: str) -> str:
    return _GUARDRAIL_DISPLAY.get(raw, raw or "Unknown")


def model_color(guardrail_model: str) -> str:
    g = (guardrail_model or "").lower()
    for k, c in _MODEL_COLORS.items():
        if k in g:
            return c
    return "#3B82F6"


def pretty_file_label(fname: str) -> str:
    """Human-readable label for an output filename."""
    n = fname.replace(".json", "")
    parts = []
    if "original_policies" in n:
        parts.append("Original Policies")
    elif "concrete_policies" in n:
        parts.append("Concrete Policies")
    elif "_3_" in n:
        parts.append("Run 3")
    else:
        parts.append(n.split("_")[0].replace("-", " ").title())

    if "claude_sonnet_4_6" in n:
        parts.append("Claude Sonnet 4.6")
    elif "claude_sonnet_4_5" in n:
        parts.append("Claude Sonnet 4.5")
    elif "gemini_2_5_flash" in n:
        parts.append("Gemini 2.5 Flash")
    elif "gpt_5_nano" in n:
        parts.append("GPT-5-nano")
    elif "all" in n:
        parts.append("All Models")
    return "  ·  ".join(parts)


def detect_policy_labels(row: dict) -> list[str]:
    labels = []
    for key in row:
        if key.endswith("_nonagentic_score"):
            labels.append(key[: -len("_nonagentic_score")])
    return sorted(labels)


def score_color(score) -> str:
    if score is None:
        return "#94A3B8"
    try:
        s = float(score)
    except (TypeError, ValueError):
        return "#94A3B8"
    if s >= 0.85:
        return "#10B981"
    if s >= 0.60:
        return "#F59E0B"
    return "#EF4444"


def score_to_valid(score) -> Optional[bool]:
    if score is None:
        return None
    try:
        return float(score) > _VALID_THRESHOLD
    except (TypeError, ValueError):
        return None


def fmt_valid_html(v) -> str:
    if v is True:
        return '<span class="badge-pass">PASS</span>'
    if v is False:
        return '<span class="badge-fail">FAIL</span>'
    return "<span style='color:#94A3B8'>—</span>"


def fmt_valid(v) -> str:
    if v is True:
        return "✅ PASS"
    if v is False:
        return "❌ FAIL"
    return "—"


def delta_arrow(delta) -> str:
    if delta is None:
        return "—"
    try:
        d = float(delta)
    except (TypeError, ValueError):
        return "—"
    if d > 0.01:
        return f"▲ +{d:.3f}"
    if d < -0.01:
        return f"▼ {d:.3f}"
    return f"≈ {d:.3f}"


def ensure_list(val) -> list:
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        try:
            parsed = json.loads(val)
            if isinstance(parsed, list):
                return parsed
        except (json.JSONDecodeError, ValueError):
            pass
    return []


def safe_int(val):
    try:
        return int(val) if val is not None else None
    except (TypeError, ValueError):
        return None


def safe_float(val):
    try:
        return float(val) if val is not None else None
    except (TypeError, ValueError):
        return None


def extract_urls_from_text(text: str) -> list[str]:
    urls: list[str] = []
    for m in re.finditer(r'https?://\S+', text or "", re.IGNORECASE):
        url = m.group(0).rstrip('.,;:!?)]\'"<>')
        if url:
            urls.append(url)
    return urls


def extract_judge_research_urls(tool_log: list) -> list[str]:
    """Extract URLs the judge fetched/searched.

    output_preview is capped at 500 chars in the log, so the JSON is often
    truncated and json.loads will fail. We try structured parsing first; if
    that fails we fall back to regex on the raw string so we still capture
    the URLs that appeared before the truncation point.
    """
    urls: list[str] = []
    for call in tool_log:
        tool = call.get("tool", "")
        inp = call.get("input", {})
        preview = call.get("output_preview", "")
        if tool == "fetch_url":
            url = inp.get("url", "") if isinstance(inp, dict) else ""
            if url:
                urls.append(url)
        elif tool == "search_web":
            found: list[str] = []
            try:
                results = json.loads(preview) if isinstance(preview, str) else preview
                if isinstance(results, dict) and "results" in results:
                    results = results["results"]
                if isinstance(results, list):
                    for r in results:
                        url = r.get("url", "") if isinstance(r, dict) else ""
                        if url:
                            found.append(url)
            except (json.JSONDecodeError, TypeError, ValueError):
                pass
            # Fallback: regex on raw preview catches URLs even in truncated JSON
            if not found:
                found = extract_urls_from_text(str(preview))
            urls.extend(found)
    return urls


def url_to_domain(url: str) -> str:
    try:
        netloc = urlparse(url).netloc.lower()
        if netloc.startswith("www."):
            netloc = netloc[4:]
        return netloc or url
    except Exception:
        return url


def avg(lst):
    return sum(lst) / len(lst) if lst else None


def pass_rate(rows, label, mode="nonagentic"):
    scores = [safe_float(r.get(f"{label}_{mode}_score")) for r in rows]
    scores = [s for s in scores if s is not None]
    if not scores:
        return None
    return sum(1 for s in scores if s > _VALID_THRESHOLD) / len(scores)


# ── SIDEBAR ────────────────────────────────────────────────────────────────────
st.sidebar.markdown("# 🛡️ Guardrail Explorer")
st.sidebar.markdown("<hr/>", unsafe_allow_html=True)

outputs_dir = Path("outputs")
json_files = sorted(outputs_dir.glob("*.json")) if outputs_dir.exists() else []

if not json_files:
    st.error("No JSON files found in `outputs/`. Run a guardrail evaluation first.")
    st.stop()

# Build display labels for each file
file_labels = {pretty_file_label(f.name) or f.name: f for f in json_files}
label_list = list(file_labels.keys())

selected_label = st.sidebar.selectbox(
    "Output file",
    label_list,
    index=len(label_list) - 1,
    help="Select an evaluation output file to explore",
)
selected_file_path = file_labels[selected_label]
selected_file = selected_file_path.name

try:
    data = load_json(str(selected_file_path))
except Exception as e:
    st.error(f"Failed to load `{selected_file}`: {e}")
    st.stop()

valid_rows = [r for r in data if any(k.endswith("_nonagentic_score") for k in r)]
error_rows = [r for r in data if r not in valid_rows]

if not valid_rows:
    st.error(f"No evaluated rows found in `{selected_file}`.")
    if error_rows:
        st.json(error_rows[0])
    st.stop()

policy_labels = detect_policy_labels(valid_rows[0])
meta = valid_rows[0]

# Sidebar metadata
st.sidebar.markdown(f"**{len(valid_rows)}** scenarios")
langs = sorted({r.get("language", "?") for r in valid_rows})
st.sidebar.markdown(f"**Languages:** {', '.join(langs)}")
st.sidebar.markdown(f"**Policies:** {', '.join(policy_labels)}")

guardrail_raw = meta.get("guardrail_model", "")
if guardrail_raw:
    st.sidebar.markdown(f"**Guardrail:** {pretty_guardrail(guardrail_raw)}")

if error_rows:
    st.sidebar.warning(f"⚠️ {len(error_rows)} rows had errors and were excluded")

st.sidebar.markdown("<hr/>", unsafe_allow_html=True)
st.sidebar.markdown('<p class="caption-text">Evaluate and compare LLM guardrail performance across languages and policies.</p>', unsafe_allow_html=True)


# ── TABS ───────────────────────────────────────────────────────────────────────
tab_overview, tab_detail, tab_urls, tab_compare_policy, tab_compare_models, tab_tokens, tab_timing = st.tabs([
    "📊 Overview",
    "🔍 Scenario Detail",
    "🔗 URL Analysis",
    "🌐 EN vs FA Policy",
    "⚖️ Compare Models",
    "📈 Token Usage",
    "⏱️ Judgment Timing",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
with tab_overview:
    # Header banner
    guardrail_display = pretty_guardrail(guardrail_raw) if guardrail_raw else "—"
    assistant_model = meta.get("model", "—")
    st.markdown(
        f"<h1>Evaluation Overview</h1>"
        f"<p class='caption-text'>Guardrail model: <strong>{guardrail_display}</strong> &nbsp;·&nbsp; "
        f"Assistant model: <strong>{assistant_model}</strong> &nbsp;·&nbsp; "
        f"File: <code>{selected_file}</code></p>",
        unsafe_allow_html=True,
    )
    st.markdown("<hr class='custom-hr'>", unsafe_allow_html=True)

    # ── KPI cards per policy ──────────────────────────────────────────────────
    for label in policy_labels:
        policy_display = "English Policy" if not label.endswith("_fa") else "Farsi Policy"
        st.markdown(f"<div class='section-label'>{policy_display} — <code>{label}</code></div>", unsafe_allow_html=True)

        na_scores = [safe_float(r.get(f"{label}_nonagentic_score")) for r in valid_rows]
        ag_scores = [safe_float(r.get(f"{label}_agentic_score")) for r in valid_rows]
        na_scores = [s for s in na_scores if s is not None]
        ag_scores = [s for s in ag_scores if s is not None]
        deltas    = [safe_float(r.get(f"{label}_score_delta")) for r in valid_rows]
        deltas    = [d for d in deltas if d is not None]
        n_changed = sum(1 for r in valid_rows if r.get(f"{label}_judgment_changed") is True)

        na_pass_pct = 100 * sum(1 for s in na_scores if s > _VALID_THRESHOLD) / len(na_scores) if na_scores else None
        ag_pass_pct = 100 * sum(1 for s in ag_scores if s > _VALID_THRESHOLD) / len(ag_scores) if ag_scores else None
        avg_na = avg(na_scores)
        avg_ag = avg(ag_scores)
        avg_delta = avg(deltas)
        change_pct = 100 * n_changed / len(valid_rows) if valid_rows else 0

        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric(
            "Non-agentic pass rate",
            f"{na_pass_pct:.0f}%" if na_pass_pct is not None else "—",
            help="% of scenarios scoring above 0.60 (no web search)",
        )
        c2.metric(
            "Agentic pass rate",
            f"{ag_pass_pct:.0f}%" if ag_pass_pct is not None else "—",
            delta=f"{(ag_pass_pct or 0) - (na_pass_pct or 0):+.0f}pp vs non-ag" if ag_pass_pct and na_pass_pct else None,
            help="% of scenarios scoring above 0.60 after web research",
        )
        c3.metric(
            "Avg non-agentic score",
            f"{avg_na:.3f}" if avg_na is not None else "—",
        )
        c4.metric(
            "Avg agentic score",
            f"{avg_ag:.3f}" if avg_ag is not None else "—",
            delta=f"{(avg_ag or 0) - (avg_na or 0):+.3f}" if avg_ag and avg_na else None,
        )
        c5.metric(
            "Avg score delta",
            f"{avg_delta:+.3f}" if avg_delta is not None else "—",
            help="Positive = agentic improved the score; negative = it lowered the score",
        )
        c6.metric(
            "Judgments changed",
            f"{n_changed} / {len(valid_rows)}",
            delta=f"{change_pct:.0f}% flip rate",
            help="Scenarios where web research caused a PASS↔FAIL verdict change",
        )

        # Score distribution chart
        if na_scores or ag_scores:
            dist_rows = (
                [{"Score": s, "Mode": "Non-agentic"} for s in na_scores] +
                [{"Score": s, "Mode": "Agentic"} for s in ag_scores]
            )
            df_dist = pd.DataFrame(dist_rows)
            dist_chart = alt.Chart(df_dist).mark_bar(opacity=0.75, binSpacing=1).encode(
                x=alt.X("Score:Q", bin=alt.Bin(step=0.1), title="Score (binned 0.1)", scale=alt.Scale(domain=[0, 1])),
                y=alt.Y("count()", title="Scenarios"),
                color=alt.Color("Mode:N", scale=alt.Scale(
                    domain=["Non-agentic", "Agentic"],
                    range=["#94A3B8", model_color(guardrail_raw)],
                )),
                tooltip=["Mode:N", "Score:Q", "count()"],
            ).properties(height=180).configure_view(strokeOpacity=0).configure_axis(
                labelFont="Inter", titleFont="Inter", labelFontSize=11, titleFontSize=11
            )
            st.altair_chart(dist_chart, use_container_width=True)

        st.markdown("<hr class='custom-hr'>", unsafe_allow_html=True)

    # ── Full results table ────────────────────────────────────────────────────
    st.markdown("## Results Table")
    st.markdown(
        "<p class='caption-text'>🟢 score ≥ 0.85 &nbsp;·&nbsp; 🟡 0.60–0.84 &nbsp;·&nbsp; "
        "🔴 &lt; 0.60 &nbsp;·&nbsp; ⚡ = agentic flipped the verdict</p>",
        unsafe_allow_html=True,
    )

    rows_for_table = []
    for row in valid_rows:
        for label in policy_labels:
            na_score  = safe_float(row.get(f"{label}_nonagentic_score"))
            ag_score  = safe_float(row.get(f"{label}_agentic_score"))
            delta     = safe_float(row.get(f"{label}_score_delta"))
            changed   = row.get(f"{label}_judgment_changed")
            tool_calls = row.get(f"{label}_agentic_tool_calls_made", 0)

            na_time = safe_float(row.get(f"{label}_nonagentic_judgment_time_s"))
            ag_time = safe_float(row.get(f"{label}_agentic_judgment_time_s"))
            rows_for_table.append({
                "ID": row.get("id", ""),
                "Lang": row.get("language", ""),
                "Scenario": str(row.get("scenario", ""))[:72] + "…",
                "Policy": label,
                "Non-ag valid": fmt_valid(score_to_valid(na_score)),
                "Non-ag score": round(na_score, 3) if na_score is not None else None,
                "Ag valid": fmt_valid(score_to_valid(ag_score)),
                "Ag score": round(ag_score, 3) if ag_score is not None else None,
                "Delta": delta_arrow(delta),
                "Changed": "⚡ YES" if changed else ("no" if changed is False else "—"),
                "Tools": f"{tool_calls}" if tool_calls else "0",
                "Non-ag time (s)": round(na_time, 2) if na_time is not None else None,
                "Ag time (s)": round(ag_time, 2) if ag_time is not None else None,
            })

    df_table = pd.DataFrame(rows_for_table)

    def _highlight_changed(row):
        return ["background-color: #FFF7ED"] * len(row) if row["Changed"] == "⚡ YES" else [""] * len(row)

    st.dataframe(
        df_table.style.apply(_highlight_changed, axis=1),
        use_container_width=True,
        height=420,
    )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — SCENARIO DETAIL
# ══════════════════════════════════════════════════════════════════════════════
with tab_detail:
    st.markdown("<h1>Scenario Detail</h1>", unsafe_allow_html=True)

    scenario_options = {
        f"[{r.get('id', i)}] ({r.get('language','?')}) {str(r.get('scenario',''))[:68]}…": i
        for i, r in enumerate(valid_rows)
    }
    selected_scenario_label = st.selectbox("Select scenario", list(scenario_options.keys()))
    row = valid_rows[scenario_options[selected_scenario_label]]

    selected_policy = st.radio("Policy", policy_labels, horizontal=True)
    label = selected_policy

    st.markdown("<hr class='custom-hr'>", unsafe_allow_html=True)

    # ── Scenario + Response ──────────────────────────────────────────────────
    st.markdown("## Scenario & Assistant Response")
    col_s, col_r = st.columns(2)
    with col_s:
        st.markdown('<div class="section-label">Scenario (sent to assistant)</div>', unsafe_allow_html=True)
        st.info(row.get("scenario", "—"))
        st.markdown(
            f'<p class="caption-text">Language: <strong>{row.get("language","?")}</strong> &nbsp;·&nbsp; ID: {row.get("id","?")}</p>',
            unsafe_allow_html=True,
        )
    with col_r:
        st.markdown('<div class="section-label">Assistant Response</div>', unsafe_allow_html=True)
        st.success(row.get("assistant_response", "—"))
        st.markdown(
            f'<p class="caption-text">Model: <strong>{row.get("model","?")}</strong> &nbsp;·&nbsp; Provider: {row.get("provider","?")}</p>',
            unsafe_allow_html=True,
        )

    st.markdown("<hr class='custom-hr'>", unsafe_allow_html=True)

    # ── Guardrail verdict comparison ─────────────────────────────────────────
    st.markdown(f"## Guardrail Verdict — <code>{label}</code>", unsafe_allow_html=True)

    na_score   = safe_float(row.get(f"{label}_nonagentic_score"))
    ag_score   = safe_float(row.get(f"{label}_agentic_score"))
    delta      = safe_float(row.get(f"{label}_score_delta"))
    changed    = row.get(f"{label}_judgment_changed")
    tool_count = row.get(f"{label}_agentic_tool_calls_made", 0)

    if changed:
        na_verdict = "PASS" if score_to_valid(na_score) else "FAIL"
        ag_verdict = "PASS" if score_to_valid(ag_score) else "FAIL"
        st.markdown(
            f'<div class="alert-changed">⚡ <strong>Judgment changed</strong> — '
            f'Non-agentic: <strong>{na_verdict}</strong> → Agentic: <strong>{ag_verdict}</strong>. '
            f'Web research caused a verdict flip.</div>',
            unsafe_allow_html=True,
        )

    col_na, col_mid, col_ag = st.columns([5, 1, 5])

    with col_na:
        st.markdown('<div class="section-label">🔒 Non-Agentic</div>', unsafe_allow_html=True)
        st.markdown('<p class="caption-text">Single LLM call · no tools · built-in knowledge only</p>', unsafe_allow_html=True)
        na_c = score_color(na_score)
        na_disp = f"{na_score:.3f}" if na_score is not None else "—"
        st.markdown(
            f'<div class="score-display" style="color:{na_c}">{na_disp}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(fmt_valid_html(score_to_valid(na_score)), unsafe_allow_html=True)
        st.markdown("**Explanation**")
        st.markdown(f"> {row.get(f'{label}_nonagentic_explanation', '—')}")

    with col_mid:
        st.markdown("<br><br><br><br><br><div style='text-align:center;font-size:1.6rem;color:#CBD5E1'>↔</div>", unsafe_allow_html=True)

    with col_ag:
        st.markdown('<div class="section-label">🌐 Agentic</div>', unsafe_allow_html=True)
        st.markdown(f'<p class="caption-text">Multi-turn loop · {tool_count} tool call(s) · web retrieval + URL checks</p>', unsafe_allow_html=True)
        ag_c = score_color(ag_score)
        ag_disp = f"{ag_score:.3f}" if ag_score is not None else "—"
        st.markdown(
            f'<div class="score-display" style="color:{ag_c}">{ag_disp}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(fmt_valid_html(score_to_valid(ag_score)), unsafe_allow_html=True)
        if delta is not None:
            d_color = "#10B981" if delta > 0.01 else ("#EF4444" if delta < -0.01 else "#94A3B8")
            st.markdown(
                f'<span style="font-size:0.9rem;color:{d_color};font-weight:600">'
                f'{delta_arrow(delta)} vs non-agentic</span>',
                unsafe_allow_html=True,
            )
        st.markdown("**Explanation**")
        st.markdown(f"> {row.get(f'{label}_agentic_explanation', '—')}")

    st.markdown("<hr class='custom-hr'>", unsafe_allow_html=True)

    # ── Token usage ──────────────────────────────────────────────────────────
    st.markdown("## Token Usage")
    na_prompt   = safe_int(row.get(f"{label}_nonagentic_prompt_tokens"))
    na_complete = safe_int(row.get(f"{label}_nonagentic_completion_tokens"))
    na_total    = safe_int(row.get(f"{label}_nonagentic_total_tokens"))
    ag_prompt   = safe_int(row.get(f"{label}_agentic_prompt_tokens_total"))
    ag_complete = safe_int(row.get(f"{label}_agentic_completion_tokens_total"))
    ag_total    = safe_int(row.get(f"{label}_agentic_total_tokens"))
    ag_peak     = safe_int(row.get(f"{label}_agentic_peak_prompt_tokens"))
    ag_per_turn = ensure_list(row.get(f"{label}_agentic_token_usage_per_turn", []))

    tok_col1, tok_col2 = st.columns(2)
    with tok_col1:
        st.markdown('<div class="section-label">Non-Agentic (tiktoken estimate)</div>', unsafe_allow_html=True)
        t1, t2, t3 = st.columns(3)
        t1.metric("Prompt", f"{na_prompt:,}" if na_prompt is not None else "—")
        t2.metric("Completion", f"{na_complete:,}" if na_complete is not None else "—")
        t3.metric("Total", f"{na_total:,}" if na_total is not None else "—")

    with tok_col2:
        st.markdown('<div class="section-label">Agentic (exact from API — all turns summed)</div>', unsafe_allow_html=True)
        t4, t5, t6, t7 = st.columns(4)
        t4.metric("Prompt total", f"{ag_prompt:,}" if ag_prompt is not None else "—")
        t5.metric("Completion", f"{ag_complete:,}" if ag_complete is not None else "—")
        t6.metric("Total", f"{ag_total:,}" if ag_total is not None else "—")
        t7.metric("Peak context", f"{ag_peak:,}" if ag_peak is not None else "—",
                  help="Largest single-turn prompt — shows context window pressure")
        if ag_total and na_total and na_total > 0:
            st.markdown(
                f'<p class="caption-text">Agentic used <strong>{ag_total/na_total:.1f}×</strong> more tokens</p>',
                unsafe_allow_html=True,
            )

    if ag_per_turn:
        with st.expander("Per-turn token breakdown (context window growth)", expanded=False):
            valid_turns = [t for t in ag_per_turn if t.get("prompt_tokens") is not None]
            if valid_turns:
                df_turns = pd.DataFrame(valid_turns)
                chart_turns = alt.Chart(df_turns).transform_fold(
                    ["prompt_tokens", "completion_tokens"],
                    as_=["Type", "Tokens"],
                ).mark_bar().encode(
                    x=alt.X("turn:O", title="Turn"),
                    y=alt.Y("Tokens:Q"),
                    color=alt.Color("Type:N", scale=alt.Scale(
                        domain=["prompt_tokens", "completion_tokens"],
                        range=["#3B82F6", "#F59E0B"],
                    )),
                    tooltip=["turn:O", "Type:N", "Tokens:Q"],
                ).properties(height=220)
                st.altair_chart(chart_turns, use_container_width=True)
            st.dataframe(pd.DataFrame(ag_per_turn), use_container_width=True)

    # ── Judgment timing ───────────────────────────────────────────────────────
    na_time = safe_float(row.get(f"{label}_nonagentic_judgment_time_s"))
    ag_time = safe_float(row.get(f"{label}_agentic_judgment_time_s"))
    if na_time is not None or ag_time is not None:
        st.markdown("## Judgment Timing")
        st.markdown(
            '<p class="caption-text">Wall-clock seconds from when the policy + text entered the '
            'judge model until it produced a score and pass/fail verdict. '
            'Agentic time includes all tool calls (search, fetch, URL checks).</p>',
            unsafe_allow_html=True,
        )
        tc1, tc2, tc3 = st.columns(3)
        tc1.metric("Non-agentic time", f"{na_time:.2f}s" if na_time is not None else "—",
                   help="Single LLM call — pure inference latency")
        tc2.metric("Agentic time", f"{ag_time:.2f}s" if ag_time is not None else "—",
                   help="All LLM turns + all tool calls + post-loop URL sweep")
        if na_time and ag_time and na_time > 0:
            mult = ag_time / na_time
            tc3.metric("Agentic / Non-ag", f"{mult:.1f}×",
                       help="How many times slower the agentic path was")

    st.markdown("<hr class='custom-hr'>", unsafe_allow_html=True)

    # ── Tool call log ─────────────────────────────────────────────────────────
    tool_log   = ensure_list(row.get(f"{label}_agentic_tool_call_log", []))
    url_checks = ensure_list(row.get(f"{label}_agentic_url_checks", []))

    if tool_log:
        with st.expander(f"Tool Call Log ({len(tool_log)} calls)", expanded=True):
            for i, call in enumerate(tool_log, 1):
                tool_name = call.get("tool", "?")
                inp = call.get("input", {})
                preview = call.get("output_preview", "")

                icon = {"search_web": "🔎", "fetch_url": "📄", "check_url_validity": "🔗"}.get(tool_name, "🔧")
                st.markdown(f"**{icon} Call {i}: `{tool_name}`**")
                tc1, tc2 = st.columns(2)

                with tc1:
                    st.markdown("*Input*")
                    if tool_name in ("fetch_url", "check_url_validity"):
                        input_url = inp.get("url", "") if isinstance(inp, dict) else ""
                        if input_url:
                            st.markdown(f"[{input_url}]({input_url})")
                        else:
                            st.code(json.dumps(inp, indent=2, ensure_ascii=False), language="json")
                    else:
                        st.code(json.dumps(inp, indent=2, ensure_ascii=False), language="json")

                with tc2:
                    st.markdown("*Output*")
                    try:
                        parsed = json.loads(preview) if isinstance(preview, str) else preview
                        if tool_name == "search_web" and isinstance(parsed, list):
                            for res in parsed:
                                url_r = res.get("url", "")
                                title_r = res.get("title", url_r)
                                snippet_r = res.get("snippet", "")
                                if url_r:
                                    st.markdown(f"**[{title_r}]({url_r})**")
                                elif title_r:
                                    st.markdown(f"**{title_r}**")
                                if snippet_r:
                                    st.caption(snippet_r[:200])
                        elif tool_name == "fetch_url" and isinstance(parsed, dict) and "content" in parsed:
                            fetched_url = parsed.get("url", inp.get("url", "") if isinstance(inp, dict) else "")
                            if fetched_url:
                                st.markdown(f"**Source:** [{fetched_url}]({fetched_url})")
                            st.caption(f"Fetched {len(parsed['content'])} characters")
                            st.text(parsed["content"][:400])
                        elif tool_name == "check_url_validity" and isinstance(parsed, dict):
                            checked_url = parsed.get("url", "")
                            final_url = parsed.get("final_url", checked_url)
                            valid_flag = parsed.get("valid", False)
                            status = parsed.get("status_code", "?")
                            error = parsed.get("error")
                            badge = "✅ Valid" if valid_flag else "❌ Invalid"
                            st.markdown(f"{badge} — HTTP **{status}**")
                            if checked_url:
                                st.markdown(f"[{checked_url}]({checked_url})")
                            if final_url and final_url != checked_url:
                                st.markdown(f"↳ Redirected: [{final_url}]({final_url})")
                            if error:
                                st.error(f"Error: {error}")
                        else:
                            st.code(json.dumps(parsed, indent=2, ensure_ascii=False)[:600], language="json")
                    except (json.JSONDecodeError, ValueError, TypeError):
                        st.text(str(preview)[:400])
                st.markdown("<hr class='custom-hr'>", unsafe_allow_html=True)
    else:
        st.markdown(
            '<div class="info-card">No tool calls were made for this policy/scenario. '
            'The guardrail evaluated without web research.</div>',
            unsafe_allow_html=True,
        )

    # ── URL validity checks ───────────────────────────────────────────────────
    if url_checks:
        with st.expander(f"URL Validity Checks ({len(url_checks)} URLs)", expanded=True):
            for uc in url_checks:
                valid_flag = uc.get("valid", False)
                status = uc.get("status_code", "?")
                url = uc.get("url", "?")
                final = uc.get("final_url", url)
                error = uc.get("error")
                redir = uc.get("redirect_count", 0)

                if valid_flag:
                    st.success(f"✅ HTTP {status}")
                    st.markdown(f"&nbsp;&nbsp;&nbsp;[{url}]({url})")
                    if final and final != url:
                        st.markdown(f"&nbsp;&nbsp;&nbsp;↳ [{final}]({final}) ({redir} redirect{'s' if redir != 1 else ''})")
                else:
                    st.error(f"❌ HTTP {status or 'FAILED'}" + (f" — {error}" if error else ""))
                    st.markdown(f"&nbsp;&nbsp;&nbsp;[{url}]({url})")
    else:
        st.markdown(
            '<div class="info-card">No URLs were found in the assistant response for this scenario — '
            'the URL validity checker had nothing to verify.</div>',
            unsafe_allow_html=True,
        )

    # ── Claim checks ──────────────────────────────────────────────────────────
    claim_checks = ensure_list(row.get(f"{label}_agentic_claim_checks", []))
    _STATUS_ICON = {"verified": "✅", "contradicted": "❌", "unverifiable": "⚠️"}
    _STATUS_ORDER = {"contradicted": 0, "unverifiable": 1, "verified": 2}

    if claim_checks:
        sorted_claims = sorted(
            claim_checks,
            key=lambda c: _STATUS_ORDER.get(str(c.get("status", "")).lower(), 3),
        )
        n_v = sum(1 for c in claim_checks if str(c.get("status","")).lower() == "verified")
        n_c = sum(1 for c in claim_checks if str(c.get("status","")).lower() == "contradicted")
        n_u = sum(1 for c in claim_checks if str(c.get("status","")).lower() == "unverifiable")
        summary = " · ".join(filter(None, [
            f"❌ {n_c} contradicted" if n_c else "",
            f"⚠️ {n_u} unverifiable" if n_u else "",
            f"✅ {n_v} verified" if n_v else "",
        ]))
        with st.expander(f"Claim Verification ({len(claim_checks)} claims — {summary})", expanded=True):
            st.markdown(
                '<p class="caption-text">Claims extracted from the assistant response and checked via web search. '
                'Contradicted and unverifiable claims lower the agentic score.</p>',
                unsafe_allow_html=True,
            )
            for c in sorted_claims:
                claim_text = str(c.get("claim", "—"))
                status_raw = str(c.get("status", "")).lower()
                icon = _STATUS_ICON.get(status_raw, "•")
                label_text = status_raw.capitalize() or "Unknown"
                sources = ensure_list(c.get("sources", []))

                if status_raw == "verified":
                    st.success(f"{icon} **{label_text}** — {claim_text}")
                elif status_raw == "contradicted":
                    st.error(f"{icon} **{label_text}** — {claim_text}")
                else:
                    st.warning(f"{icon} **{label_text}** — {claim_text}")

                if sources:
                    for s in sources[:3]:
                        if isinstance(s, str) and s.startswith("http"):
                            st.markdown(f"&nbsp;&nbsp;&nbsp;↳ [{s}]({s})")
    else:
        st.markdown(
            '<div class="info-card">No claim checks recorded — either no verifiable factual claims were found '
            'or this result was produced before claim tracking was added.</div>',
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# PRECOMPUTE URL RECORDS (used by Tab 3 URL Analysis — done once outside tabs)
# ══════════════════════════════════════════════════════════════════════════════

_is_en = lambda l: l.lower().startswith("en")

# --- Assistant response URL validity records ---
_coverage_records: list[dict] = []
for _row in valid_rows:
    _lang = _row.get("language", "unknown")
    _resp_urls = list(dict.fromkeys(extract_urls_from_text(_row.get("assistant_response", ""))))
    for _lbl in policy_labels:
        _uc_list = ensure_list(_row.get(f"{_lbl}_agentic_url_checks", []))
        _uc_map = {uc.get("url", ""): uc for uc in _uc_list if uc.get("url")}
        for _url in _resp_urls:
            _chk = _uc_map.get(_url)
            _coverage_records.append({
                "url": _url,
                "domain": url_to_domain(_url),
                "policy": _lbl,
                "language": _lang,
                "was_checked": _chk is not None,
                "is_valid": _chk.get("valid") if _chk else None,
                "status_code": _chk.get("status_code") if _chk else None,
            })

# --- Judge research URLs (from search_web + fetch_url in tool logs) ---
_research_records: list[dict] = []
for _row in valid_rows:
    _lang = _row.get("language", "unknown")
    for _lbl in policy_labels:
        _tlog = ensure_list(_row.get(f"{_lbl}_agentic_tool_call_log", []))
        for _url in extract_judge_research_urls(_tlog):
            _research_records.append({
                "url": _url,
                "domain": url_to_domain(_url),
                "policy": _lbl,
                "language": _lang,
            })

# --- Claim check domain records ---
_claim_records: list[dict] = []
for _row in valid_rows:
    _lang = _row.get("language", "unknown")
    for _lbl in policy_labels:
        for _claim in ensure_list(_row.get(f"{_lbl}_agentic_claim_checks", [])):
            _status = str(_claim.get("status", "")).lower()
            for _src in ensure_list(_claim.get("sources", [])):
                if isinstance(_src, str) and _src.startswith("http"):
                    _claim_records.append({
                        "url": _src,
                        "domain": url_to_domain(_src),
                        "status": _status,
                        "policy": _lbl,
                        "language": _lang,
                    })


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — URL ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
with tab_urls:
    _n_asst = len({r["url"] for r in _coverage_records})
    _n_res  = len({r["url"] for r in _research_records})
    _n_claim_src = len({r["url"] for r in _claim_records})

    st.markdown("<h1>URL Analysis</h1>", unsafe_allow_html=True)
    st.markdown(
        '<p class="caption-text">'
        'Three URL populations for this evaluation file: '
        '(1) URLs the <strong>assistant cited</strong> in its responses and their HTTP validity; '
        '(2) URLs the <strong>guardrail judge fetched/searched</strong> to verify claims; '
        '(3) Source domains cited as evidence in <strong>claim verification</strong>.'
        '</p>',
        unsafe_allow_html=True,
    )

    # Top-line KPIs
    k1, k2, k3 = st.columns(3)
    k1.metric("Unique response URLs", f"{_n_asst:,}",
              help="Distinct URLs found via regex in all assistant responses")
    k2.metric("Unique judge research URLs", f"{_n_res:,}",
              help="Distinct URLs fetched or returned in search_web results by the guardrail judge")
    k3.metric("Unique claim-source URLs", f"{_n_claim_src:,}",
              help="Distinct source URLs cited as evidence in claim checks")

    st.markdown("<hr class='custom-hr'>", unsafe_allow_html=True)

    url_subtab1, url_subtab2, url_subtab3 = st.tabs([
        f"1 · Assistant Response URLs  ({_n_asst} unique)",
        f"2 · Judge Research Domains  ({_n_res} unique)",
        f"3 · Claim-Check Sources  ({_n_claim_src} unique)",
    ])

    # ── Sub-tab 1: response URL validity ─────────────────────────────────────
    with url_subtab1:
        st.markdown(
            '<p class="caption-text">Every URL regex-matched from assistant responses, '
            'joined with <code>check_url_validity</code> results stored in the output. '
            'Valid = HTTP &lt; 400 or 401/403 (resource exists, access restricted). '
            'Unchecked = URL was found in the response but the guardrail did not run a check '
            '(common in older output files).</p>',
            unsafe_allow_html=True,
        )

        if not _coverage_records:
            st.markdown(
                '<div class="info-card">No URLs found in any assistant response in this file. '
                'The assistant LLM did not cite external links.</div>',
                unsafe_allow_html=True,
            )
        else:
            _total    = len(_coverage_records)
            _n_valid  = sum(1 for r in _coverage_records if r["is_valid"] is True)
            _n_inv    = sum(1 for r in _coverage_records if r["is_valid"] is False)
            _n_unchkd = sum(1 for r in _coverage_records if r["is_valid"] is None)
            _checked  = _n_valid + _n_inv
            _valid_pct = 100 * _n_valid / _checked if _checked else 0

            cm1, cm2, cm3, cm4, cm5 = st.columns(5)
            cm1.metric("Total URL appearances", f"{_total:,}",
                       help="URL × scenario × policy (same URL may appear in multiple scenarios)")
            cm2.metric("Unique domains", f"{len({r['domain'] for r in _coverage_records}):,}")
            cm3.metric("Valid (HTTP ≤ 403)", f"{_n_valid:,}")
            cm4.metric("Invalid / broken", f"{_n_inv:,}",
                       help="HTTP ≥ 404, timeout, or connection failure — likely hallucinated or stale")
            cm5.metric("Validity rate", f"{_valid_pct:.0f}%",
                       help="Valid / (Valid + Invalid). Unchecked excluded from denominator.")

            if _n_unchkd:
                st.warning(
                    f"⚠️ {_n_unchkd:,} URL appearance(s) have no check result. "
                    "These URLs were found in responses but the guardrail did not validate them — "
                    "typical for output files produced before the automatic URL sweep was added. "
                    "Re-run the evaluation for full coverage."
                )

            st.markdown("<hr class='custom-hr'>", unsafe_allow_html=True)

            # Stacked bar: top 20 domains valid vs invalid vs unchecked
            _dc_all = Counter(r["domain"] for r in _coverage_records)
            _dc_val = Counter(r["domain"] for r in _coverage_records if r["is_valid"] is True)
            _dc_inv = Counter(r["domain"] for r in _coverage_records if r["is_valid"] is False)
            _dc_unc = Counter(r["domain"] for r in _coverage_records if r["is_valid"] is None)
            _top_n = min(20, len(_dc_all))
            _cov_df = pd.DataFrame([
                {
                    "Domain": d,
                    "Valid": _dc_val.get(d, 0),
                    "Invalid": _dc_inv.get(d, 0),
                    "Unchecked": _dc_unc.get(d, 0),
                }
                for d, _ in _dc_all.most_common(_top_n)
            ])

            _fold_cols = ["Valid", "Invalid"]
            if _dc_unc:
                _fold_cols.append("Unchecked")

            st.markdown(f"**Top {_top_n} domains — valid vs invalid vs unchecked**")
            _stacked = alt.Chart(_cov_df).transform_fold(
                _fold_cols, as_=["Status", "Count"]
            ).mark_bar().encode(
                x=alt.X("Count:Q", title="Appearances"),
                y=alt.Y("Domain:N", sort="-x", title=None),
                color=alt.Color("Status:N", scale=alt.Scale(
                    domain=["Valid", "Invalid", "Unchecked"],
                    range=["#10B981", "#EF4444", "#94A3B8"],
                )),
                tooltip=["Domain:N", "Status:N", "Count:Q"],
            ).properties(height=max(200, min(36 * _top_n, 560)))
            st.altair_chart(_stacked, use_container_width=True)

            # Per-policy breakdown
            if len(policy_labels) > 1:
                st.markdown("**Per-policy breakdown**")
                pol_cols = st.columns(len(policy_labels))
                for _pc, _plbl in zip(pol_cols, policy_labels):
                    _pr = [r for r in _coverage_records if r["policy"] == _plbl]
                    _pd_all = Counter(r["domain"] for r in _pr)
                    _pd_val = Counter(r["domain"] for r in _pr if r["is_valid"] is True)
                    _pd_inv = Counter(r["domain"] for r in _pr if r["is_valid"] is False)
                    _pc.markdown(f"`{_plbl}` — {len(_pr):,} appearances")
                    _pc.dataframe(
                        pd.DataFrame([
                            {"Domain": d, "Found": _pd_all[d],
                             "Valid": _pd_val.get(d, 0), "Invalid": _pd_inv.get(d, 0)}
                            for d, _ in _pd_all.most_common(15)
                        ]),
                        use_container_width=True, height=280,
                    )

            with st.expander("Full URL validity table + CSV download", expanded=False):
                _cov_agg: dict = {}
                _all_langs_cov = sorted({r["language"] for r in _coverage_records})
                for _rec in _coverage_records:
                    _key = (_rec["url"], _rec["policy"])
                    if _key not in _cov_agg:
                        _cov_agg[_key] = {
                            "url": _rec["url"], "domain": _rec["domain"],
                            "policy": _rec["policy"], "appearances": 0,
                            "valid": 0, "invalid": 0, "unchecked": 0,
                            "status_code": _rec.get("status_code"),
                            **{f"lang_{l}": 0 for l in _all_langs_cov},
                        }
                    _cov_agg[_key]["appearances"] += 1
                    if _rec["is_valid"] is True:
                        _cov_agg[_key]["valid"] += 1
                    elif _rec["is_valid"] is False:
                        _cov_agg[_key]["invalid"] += 1
                    else:
                        _cov_agg[_key]["unchecked"] += 1
                    if _rec.get("status_code") is not None:
                        _cov_agg[_key]["status_code"] = _rec["status_code"]
                    _cov_agg[_key][f"lang_{_rec['language']}"] += 1
                _df_cov = pd.DataFrame(sorted(_cov_agg.values(), key=lambda x: -x["appearances"]))
                _checked_col = _df_cov["valid"] + _df_cov["invalid"]
                _df_cov["validity_%"] = (
                    _df_cov["valid"] / _checked_col.replace(0, float("nan")) * 100
                ).round(1)
                st.dataframe(_df_cov, use_container_width=True, height=300)
                st.download_button(
                    "⬇️  Download CSV",
                    _df_cov.to_csv(index=False).encode("utf-8"),
                    file_name=f"{selected_file.replace('.json', '')}_response_url_validity.csv",
                    mime="text/csv",
                )

    # ── Sub-tab 2: judge research domains ────────────────────────────────────
    with url_subtab2:
        st.markdown(
            '<p class="caption-text">URLs the <strong>guardrail judge</strong> fetched or received '
            'from web search while verifying the assistant response. Sources: '
            '<code>fetch_url</code> inputs + URLs parsed from <code>search_web</code> result previews. '
            'These show what the web-enabled guardrail actually consulted — not the assistant\'s citations.</p>',
            unsafe_allow_html=True,
        )

        if not _research_records:
            st.markdown(
                '<div class="info-card">No judge research URLs found in this file. '
                'Either the guardrail made no tool calls, or no web search / fetch results '
                'were recorded in the tool log.</div>',
                unsafe_allow_html=True,
            )
        else:
            _r_langs = sorted({r["language"] for r in _research_records})
            _rdc = Counter(r["domain"] for r in _research_records)
            _top_rn = min(20, len(_rdc))

            rk1, rk2, rk3 = st.columns(3)
            rk1.metric("Total URL appearances", f"{sum(_rdc.values()):,}")
            rk2.metric("Unique domains", f"{len(_rdc):,}")
            rk3.metric("Unique URLs", f"{len({r['url'] for r in _research_records}):,}")

            st.markdown("<hr class='custom-hr'>", unsafe_allow_html=True)
            st.markdown(f"**Top {_top_rn} domains consulted by the judge**")

            _rdf = pd.DataFrame(_rdc.most_common(_top_rn), columns=["Domain", "Count"])
            _r_chart = alt.Chart(_rdf).mark_bar(color="#3B82F6").encode(
                x=alt.X("Count:Q", title="Appearances in tool logs"),
                y=alt.Y("Domain:N", sort="-x", title=None),
                tooltip=["Domain:N", "Count:Q"],
            ).properties(height=max(200, min(36 * _top_rn, 560)))
            st.altair_chart(_r_chart, use_container_width=True)

            # EN vs non-EN comparison
            _r_en  = Counter(r["domain"] for r in _research_records if _is_en(r["language"]))
            _r_oth = Counter(r["domain"] for r in _research_records if not _is_en(r["language"]))
            if _r_en or _r_oth:
                st.markdown("**English vs non-English scenario breakdown**")
                rc1, rc2 = st.columns(2)
                for _col, _cnt, _lbl in [(rc1, _r_en, "English scenarios"), (rc2, _r_oth, "Non-English scenarios")]:
                    _col.markdown(f"**{_lbl}** — {sum(_cnt.values()):,} total")
                    if _cnt:
                        _col.dataframe(
                            pd.DataFrame(_cnt.most_common(15), columns=["Domain", "Count"]),
                            use_container_width=True, height=260,
                        )
                    else:
                        _col.caption("No data.")

            with st.expander("Full research URL list + CSV download", expanded=False):
                _r_agg: dict = {}
                for _rec in _research_records:
                    _u = _rec["url"]
                    if _u not in _r_agg:
                        _r_agg[_u] = {"url": _u, "domain": _rec["domain"], "total": 0,
                                      **{f"lang_{l}": 0 for l in _r_langs},
                                      **{f"policy_{p}": 0 for p in policy_labels}}
                    _r_agg[_u]["total"] += 1
                    _r_agg[_u][f"lang_{_rec['language']}"] += 1
                    if _rec["policy"] in policy_labels:
                        _r_agg[_u][f"policy_{_rec['policy']}"] += 1
                _df_r = pd.DataFrame(sorted(_r_agg.values(), key=lambda x: -x["total"]))
                st.dataframe(_df_r, use_container_width=True, height=300)
                st.download_button(
                    "⬇️  Download CSV",
                    _df_r.to_csv(index=False).encode("utf-8"),
                    file_name=f"{selected_file.replace('.json', '')}_judge_research_urls.csv",
                    mime="text/csv",
                )

    # ── Sub-tab 3: claim check source domains ────────────────────────────────
    with url_subtab3:
        st.markdown(
            '<p class="caption-text">Source URLs the judge cited as evidence when '
            'verifying or contradicting claims extracted from the assistant response. '
            'Grouped by claim verdict: <strong>verified</strong>, <strong>contradicted</strong>, '
            '<strong>unverifiable</strong>.</p>',
            unsafe_allow_html=True,
        )

        if not _claim_records:
            st.markdown(
                '<div class="info-card">No source URLs recorded in claim checks for this file. '
                'Claim checks may have been run without source tracking, or no claims were found.</div>',
                unsafe_allow_html=True,
            )
        else:
            _cd_all  = Counter(r["domain"] for r in _claim_records)
            _cd_v    = Counter(r["domain"] for r in _claim_records if r["status"] == "verified")
            _cd_c    = Counter(r["domain"] for r in _claim_records if r["status"] == "contradicted")
            _cd_u    = Counter(r["domain"] for r in _claim_records if r["status"] == "unverifiable")
            _top_cn  = min(20, len(_cd_all))

            ck1, ck2, ck3, ck4 = st.columns(4)
            ck1.metric("Total source appearances", f"{sum(_cd_all.values()):,}")
            ck2.metric("Unique domains", f"{len(_cd_all):,}")
            ck3.metric("Domains in verified claims", f"{len(_cd_v):,}")
            ck4.metric("Domains in contradicted claims", f"{len(_cd_c):,}")

            st.markdown("<hr class='custom-hr'>", unsafe_allow_html=True)

            _claim_df = pd.DataFrame([
                {
                    "Domain": d,
                    "Verified": _cd_v.get(d, 0),
                    "Contradicted": _cd_c.get(d, 0),
                    "Unverifiable": _cd_u.get(d, 0),
                }
                for d, _ in _cd_all.most_common(_top_cn)
            ])
            st.markdown(f"**Top {_top_cn} source domains — by claim verdict**")
            _claim_chart = alt.Chart(_claim_df).transform_fold(
                ["Verified", "Contradicted", "Unverifiable"], as_=["Verdict", "Count"]
            ).mark_bar().encode(
                x=alt.X("Count:Q", title="Times cited"),
                y=alt.Y("Domain:N", sort="-x", title=None),
                color=alt.Color("Verdict:N", scale=alt.Scale(
                    domain=["Verified", "Contradicted", "Unverifiable"],
                    range=["#10B981", "#EF4444", "#F59E0B"],
                )),
                tooltip=["Domain:N", "Verdict:N", "Count:Q"],
            ).properties(height=max(200, min(36 * _top_cn, 560)))
            st.altair_chart(_claim_chart, use_container_width=True)

            with st.expander("Full claim-source URL list + CSV", expanded=False):
                _claim_agg: dict = {}
                for _rec in _claim_records:
                    _u = _rec["url"]
                    if _u not in _claim_agg:
                        _claim_agg[_u] = {"url": _u, "domain": _rec["domain"],
                                          "verified": 0, "contradicted": 0, "unverifiable": 0, "total": 0}
                    _claim_agg[_u][_rec["status"]] = _claim_agg[_u].get(_rec["status"], 0) + 1
                    _claim_agg[_u]["total"] += 1
                _df_claim = pd.DataFrame(sorted(_claim_agg.values(), key=lambda x: -x["total"]))
                st.dataframe(_df_claim, use_container_width=True, height=300)
                st.download_button(
                    "⬇️  Download CSV",
                    _df_claim.to_csv(index=False).encode("utf-8"),
                    file_name=f"{selected_file.replace('.json', '')}_claim_source_domains.csv",
                    mime="text/csv",
                )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — EN vs FA POLICY COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
with tab_compare_policy:
    st.markdown("<h1>English vs Farsi Policy</h1>", unsafe_allow_html=True)
    st.markdown(
        '<p class="caption-text">Same response, same guardrail, same criteria — does the language '
        'the <em>policy</em> is written in change the score? Compares <code>policy</code> (English) '
        'vs <code>policy_fa</code> (Farsi) for every scenario.</p>',
        unsafe_allow_html=True,
    )

    en_label = next((l for l in policy_labels if not l.endswith("_fa")), None)
    fa_label = next((l for l in policy_labels if l.endswith("_fa")), None)

    if not en_label or not fa_label:
        st.warning(
            f"Both an English policy (`policy`) and Farsi policy (`policy_fa`) are required. "
            f"Found in this file: {policy_labels}"
        )
    else:
        compare_rows = []
        for row in valid_rows:
            en_na = safe_float(row.get(f"{en_label}_nonagentic_score"))
            fa_na = safe_float(row.get(f"{fa_label}_nonagentic_score"))
            en_ag = safe_float(row.get(f"{en_label}_agentic_score"))
            fa_ag = safe_float(row.get(f"{fa_label}_agentic_score"))

            def safe_diff(a, b):
                return round(a - b, 3) if a is not None and b is not None else None

            compare_rows.append({
                "ID": row.get("id", ""),
                "Lang": row.get("language", ""),
                "Scenario": str(row.get("scenario", ""))[:60] + "…",
                "EN non-ag": round(en_na, 3) if en_na is not None else None,
                "FA non-ag": round(fa_na, 3) if fa_na is not None else None,
                "EN−FA (non-ag)": safe_diff(en_na, fa_na),
                "EN agentic": round(en_ag, 3) if en_ag is not None else None,
                "FA agentic": round(fa_ag, 3) if fa_ag is not None else None,
                "EN−FA (agentic)": safe_diff(en_ag, fa_ag),
            })

        df_cmp = pd.DataFrame(compare_rows)

        def _color_diff(val):
            if val is None:
                return ""
            try:
                v = float(val)
            except (TypeError, ValueError):
                return ""
            if abs(v) < 0.05:
                return "color: #10B981"
            if abs(v) < 0.15:
                return "color: #F59E0B"
            return "color: #EF4444; font-weight: 700"

        st.dataframe(
            df_cmp.style.map(_color_diff, subset=["EN−FA (non-ag)", "EN−FA (agentic)"]),
            use_container_width=True,
            height=400,
        )

        # Summary metrics
        st.markdown("<hr class='custom-hr'>", unsafe_allow_html=True)
        st.markdown("## Language Bias Summary")
        diffs_na = [r["EN−FA (non-ag)"] for r in compare_rows if r["EN−FA (non-ag)"] is not None]
        diffs_ag = [r["EN−FA (agentic)"] for r in compare_rows if r["EN−FA (agentic)"] is not None]
        n_en_higher_na = sum(1 for d in diffs_na if d > 0.05)
        n_fa_higher_na = sum(1 for d in diffs_na if d < -0.05)
        n_en_higher_ag = sum(1 for d in diffs_ag if d > 0.05)
        n_fa_higher_ag = sum(1 for d in diffs_ag if d < -0.05)

        sb1, sb2, sb3, sb4 = st.columns(4)
        sb1.metric("Avg |EN−FA| non-ag", f"{avg([abs(d) for d in diffs_na]):.3f}" if diffs_na else "—",
                   help="Average absolute difference between EN and FA policy scores (non-agentic)")
        sb2.metric("Avg |EN−FA| agentic", f"{avg([abs(d) for d in diffs_ag]):.3f}" if diffs_ag else "—",
                   help="Average absolute difference between EN and FA policy scores (agentic)")
        sb3.metric("EN scores higher (>0.05)", f"{n_en_higher_na} / {len(diffs_na)}" if diffs_na else "—")
        sb4.metric("FA scores higher (>0.05)", f"{n_fa_higher_na} / {len(diffs_na)}" if diffs_na else "—")

        # Scatter: EN score vs FA score
        scatter_rows = [
            {"EN score": r["EN non-ag"], "FA score": r["FA non-ag"],
             "Mode": "Non-agentic", "ID": r["ID"], "Lang": r["Lang"]}
            for r in compare_rows if r["EN non-ag"] is not None and r["FA non-ag"] is not None
        ] + [
            {"EN score": r["EN agentic"], "FA score": r["FA agentic"],
             "Mode": "Agentic", "ID": r["ID"], "Lang": r["Lang"]}
            for r in compare_rows if r["EN agentic"] is not None and r["FA agentic"] is not None
        ]
        if scatter_rows:
            df_scatter = pd.DataFrame(scatter_rows)
            diagonal = pd.DataFrame({"EN score": [0, 1], "FA score": [0, 1]})
            scatter_chart = alt.Chart(df_scatter).mark_circle(size=70, opacity=0.75).encode(
                x=alt.X("EN score:Q", scale=alt.Scale(domain=[0, 1])),
                y=alt.Y("FA score:Q", scale=alt.Scale(domain=[0, 1])),
                color=alt.Color("Mode:N", scale=alt.Scale(
                    domain=["Non-agentic", "Agentic"],
                    range=["#94A3B8", model_color(guardrail_raw)],
                )),
                tooltip=["ID:N", "Lang:N", "Mode:N", "EN score:Q", "FA score:Q"],
            )
            diag_line = alt.Chart(diagonal).mark_line(color="#CBD5E1", strokeDash=[4, 4]).encode(
                x="EN score:Q", y="FA score:Q"
            )
            st.markdown("**Score scatter — EN vs FA** (points above diagonal = FA scored higher)")
            st.altair_chart((diag_line + scatter_chart).properties(height=350), use_container_width=True)

        # Deep dive
        st.markdown("<hr class='custom-hr'>", unsafe_allow_html=True)
        st.markdown("## Deep Dive — Single Scenario")
        opts_cmp = {
            f"[{r.get('id',i)}] ({r.get('language','?')}) {str(r.get('scenario',''))[:60]}…": i
            for i, r in enumerate(valid_rows)
        }
        sel_cmp = st.selectbox("Scenario", list(opts_cmp.keys()), key="cmp_sel")
        row_cmp = valid_rows[opts_cmp[sel_cmp]]
        col_en, col_fa = st.columns(2)

        for col, lbl, flag in [(col_en, en_label, "🇬🇧 English policy"), (col_fa, fa_label, "🇮🇷 Farsi policy")]:
            with col:
                st.markdown(f"### {flag} — `{lbl}`")
                na_s = safe_float(row_cmp.get(f"{lbl}_nonagentic_score"))
                ag_s = safe_float(row_cmp.get(f"{lbl}_agentic_score"))
                ag_delta = safe_float(row_cmp.get(f"{lbl}_score_delta"))
                st.markdown("**Non-agentic**")
                c1, c2 = st.columns(2)
                c1.metric("Score", f"{na_s:.3f}" if na_s is not None else "—")
                c2.markdown(fmt_valid_html(score_to_valid(na_s)), unsafe_allow_html=True)
                with st.expander("Explanation"):
                    st.write(row_cmp.get(f"{lbl}_nonagentic_explanation", "—"))
                st.markdown("**Agentic**")
                c3, c4 = st.columns(2)
                c3.metric("Score", f"{ag_s:.3f}" if ag_s is not None else "—",
                          delta=f"{ag_delta:+.3f}" if ag_delta is not None else None)
                c4.markdown(fmt_valid_html(score_to_valid(ag_s)), unsafe_allow_html=True)
                with st.expander("Explanation"):
                    st.write(row_cmp.get(f"{lbl}_agentic_explanation", "—"))


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — CROSS-MODEL COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
with tab_compare_models:
    st.markdown("<h1>Compare Guardrail Models</h1>", unsafe_allow_html=True)
    st.markdown(
        '<p class="caption-text">Load multiple output files from the same experiment '
        'to compare how Claude, Gemini, and GPT perform as guardrail judges on identical scenarios.</p>',
        unsafe_allow_html=True,
    )

    # Group files by experiment prefix
    _experiment_groups: dict[str, list[Path]] = {}
    for f in json_files:
        n = f.name
        if "original_policies" in n:
            grp = "Original Policies"
        elif "concrete_policies" in n:
            grp = "Concrete Policies"
        elif "_3_" in n:
            grp = "Run 3"
        else:
            grp = "Other"
        _experiment_groups.setdefault(grp, []).append(f)

    selected_group = st.selectbox(
        "Experiment group",
        list(_experiment_groups.keys()),
        help="Files in the same group share scenarios — meaningful to compare",
    )
    group_files = _experiment_groups[selected_group]

    # Show which files are in the group
    st.markdown('<div class="section-label">Files in this group</div>', unsafe_allow_html=True)
    for gf in group_files:
        st.markdown(f"- `{gf.name}`")

    # Load all files in the group
    @st.cache_data(show_spinner=False)
    def _load_group(paths_str: str) -> dict[str, list[dict]]:
        result = {}
        for p in paths_str.split("|"):
            try:
                with open(p) as fh:
                    d = json.load(fh)
                vrows = [r for r in d if any(k.endswith("_nonagentic_score") for k in r)]
                if vrows:
                    result[Path(p).name] = vrows
            except Exception:
                pass
        return result

    group_data = _load_group("|".join(str(f) for f in group_files))

    if len(group_data) < 2:
        st.info("Need at least 2 valid files in this group to compare. Try a different experiment group.")
    else:
        # Build comparison: for each scenario ID × policy, show all models side by side
        # Get all scenario IDs (use ID from first file)
        first_rows = next(iter(group_data.values()))
        first_policy_labels = detect_policy_labels(first_rows[0])

        comp_policy = st.radio("Policy to compare", first_policy_labels, horizontal=True, key="cmp_pol")
        comp_mode   = st.radio("Mode", ["Non-agentic", "Agentic"], horizontal=True, key="cmp_mode")
        mode_key    = "nonagentic" if comp_mode == "Non-agentic" else "agentic"

        # Guardrail model per file
        file_guardrail = {}
        for fname, rows in group_data.items():
            gm = rows[0].get("guardrail_model", fname)
            file_guardrail[fname] = pretty_guardrail(gm) if gm else pretty_file_label(fname)

        # Build comparison table: rows = scenarios, cols = files
        scenario_index: dict[str, dict] = {}
        for fname, rows in group_data.items():
            for r in rows:
                sid = str(r.get("id", ""))
                if sid not in scenario_index:
                    scenario_index[sid] = {
                        "ID": r.get("id", ""),
                        "Lang": r.get("language", ""),
                        "Scenario": str(r.get("scenario", ""))[:60] + "…",
                    }
                score = safe_float(r.get(f"{comp_policy}_{mode_key}_score"))
                scenario_index[sid][file_guardrail[fname]] = round(score, 3) if score is not None else None

        df_cross = pd.DataFrame(list(scenario_index.values()))
        model_cols = [file_guardrail[f] for f in group_data]

        # Color each score cell
        def _color_score(val):
            if val is None:
                return "color: #94A3B8"
            try:
                v = float(val)
            except (TypeError, ValueError):
                return ""
            if v >= 0.85:
                return "color: #10B981; font-weight: 600"
            if v >= 0.60:
                return "color: #F59E0B; font-weight: 600"
            return "color: #EF4444; font-weight: 700"

        valid_model_cols = [c for c in model_cols if c in df_cross.columns]
        st.dataframe(
            df_cross.style.map(_color_score, subset=valid_model_cols),
            use_container_width=True,
            height=450,
        )

        st.markdown("<hr class='custom-hr'>", unsafe_allow_html=True)

        # ── Aggregate comparison ──────────────────────────────────────────────
        st.markdown("## Aggregate Comparison")
        agg_rows = []
        for fname, rows in group_data.items():
            gm_display = file_guardrail[fname]
            for pol in first_policy_labels:
                na_s = [safe_float(r.get(f"{pol}_nonagentic_score")) for r in rows]
                ag_s = [safe_float(r.get(f"{pol}_agentic_score")) for r in rows]
                na_s = [s for s in na_s if s is not None]
                ag_s = [s for s in ag_s if s is not None]
                deltas = [safe_float(r.get(f"{pol}_score_delta")) for r in rows]
                deltas = [d for d in deltas if d is not None]
                n_changed = sum(1 for r in rows if r.get(f"{pol}_judgment_changed") is True)

                agg_rows.append({
                    "Guardrail model": gm_display,
                    "Policy": pol,
                    "Avg non-ag score": round(avg(na_s), 3) if na_s else None,
                    "Non-ag pass rate": f"{100*sum(1 for s in na_s if s>_VALID_THRESHOLD)/len(na_s):.0f}%" if na_s else "—",
                    "Avg agentic score": round(avg(ag_s), 3) if ag_s else None,
                    "Ag pass rate": f"{100*sum(1 for s in ag_s if s>_VALID_THRESHOLD)/len(ag_s):.0f}%" if ag_s else "—",
                    "Avg delta": round(avg(deltas), 3) if deltas else None,
                    "Changed": f"{n_changed}/{len(rows)}",
                })

        df_agg = pd.DataFrame(agg_rows)
        score_agg_cols = ["Avg non-ag score", "Avg agentic score", "Avg delta"]
        st.dataframe(
            df_agg.style.map(_color_score, subset=[c for c in score_agg_cols if c in df_agg.columns]),
            use_container_width=True,
        )

        # ── Bar chart comparing models ────────────────────────────────────────
        st.markdown("<hr class='custom-hr'>", unsafe_allow_html=True)
        st.markdown("## Score Distribution by Guardrail Model")
        dist_rows_cross = []
        for fname, rows in group_data.items():
            gm_display = file_guardrail[fname]
            for r in rows:
                for pol in first_policy_labels:
                    for mode_label, mk in [("Non-agentic", "nonagentic"), ("Agentic", "agentic")]:
                        s = safe_float(r.get(f"{pol}_{mk}_score"))
                        if s is not None:
                            dist_rows_cross.append({
                                "Score": s,
                                "Model": gm_display,
                                "Mode": mode_label,
                                "Policy": pol,
                            })

        if dist_rows_cross:
            df_dist_cross = pd.DataFrame(dist_rows_cross)
            cross_filter_mode = st.radio("Mode", ["Non-agentic", "Agentic"], horizontal=True, key="cross_dist_mode")
            df_dist_filt = df_dist_cross[df_dist_cross["Mode"] == cross_filter_mode]

            box_base = alt.Chart(df_dist_filt).mark_boxplot(extent="min-max", size=40).encode(
                x=alt.X("Model:N", title="Guardrail model"),
                y=alt.Y("Score:Q", scale=alt.Scale(domain=[0, 1])),
                color=alt.Color("Model:N", legend=None),
            ).properties(height=280)
            if len(first_policy_labels) > 1:
                box_base = box_base.encode(column=alt.Column("Policy:N"))
            st.altair_chart(box_base, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — TOKEN USAGE
# ══════════════════════════════════════════════════════════════════════════════
with tab_tokens:
    st.markdown("<h1>Token Usage</h1>", unsafe_allow_html=True)
    st.markdown(
        '<p class="caption-text">'
        '<strong>Non-agentic:</strong> counted with tiktoken (same tokenizer the model uses) — '
        'prompt = role instruction + policy + rubric + system prompt + scenario + response. '
        '<strong>Agentic:</strong> exact figures from the provider API, summed across all LLM turns. '
        '<strong>Peak context</strong> = largest single-turn prompt — shows context window pressure.'
        '</p>',
        unsafe_allow_html=True,
    )
    st.markdown("<hr class='custom-hr'>", unsafe_allow_html=True)

    # ── Summary stats ─────────────────────────────────────────────────────────
    for label in policy_labels:
        st.markdown(f'<div class="section-label">Policy: <code>{label}</code></div>', unsafe_allow_html=True)
        na_totals  = [safe_int(r.get(f"{label}_nonagentic_total_tokens")) for r in valid_rows]
        ag_totals  = [safe_int(r.get(f"{label}_agentic_total_tokens")) for r in valid_rows]
        ag_peaks   = [safe_int(r.get(f"{label}_agentic_peak_prompt_tokens")) for r in valid_rows]
        ag_prompts = [safe_int(r.get(f"{label}_agentic_prompt_tokens_total")) for r in valid_rows]
        ag_comps   = [safe_int(r.get(f"{label}_agentic_completion_tokens_total")) for r in valid_rows]

        na_totals  = [x for x in na_totals if x is not None]
        ag_totals  = [x for x in ag_totals if x is not None]
        ag_peaks   = [x for x in ag_peaks if x is not None]
        ag_prompts = [x for x in ag_prompts if x is not None]
        ag_comps   = [x for x in ag_comps if x is not None]

        def fmt_avg(lst): return f"{int(avg(lst)):,}" if lst else "—"
        def fmt_max(lst): return f"{max(lst):,}" if lst else "—"

        t1, t2, t3, t4, t5 = st.columns(5)
        t1.metric("Avg non-ag tokens", fmt_avg(na_totals), help="tiktoken estimate")
        t2.metric("Avg agentic tokens", fmt_avg(ag_totals), help="exact from API")
        t3.metric("Avg agentic prompt", fmt_avg(ag_prompts))
        t4.metric("Avg agentic completion", fmt_avg(ag_comps))
        t5.metric("Max peak context", fmt_max(ag_peaks), help="Highest single-turn prompt across all scenarios")

        if na_totals and ag_totals:
            avg_mult = avg(ag_totals) / avg(na_totals)
            st.markdown(
                f'<p class="caption-text">Agentic uses <strong>{avg_mult:.1f}×</strong> more tokens on average</p>',
                unsafe_allow_html=True,
            )

        st.markdown("<hr class='custom-hr'>", unsafe_allow_html=True)

    # ── Per-scenario token table ───────────────────────────────────────────────
    st.markdown("## Per-Scenario Token Table")
    token_rows = []
    for row in valid_rows:
        for label in policy_labels:
            na_p = safe_int(row.get(f"{label}_nonagentic_prompt_tokens"))
            na_c = safe_int(row.get(f"{label}_nonagentic_completion_tokens"))
            na_t = safe_int(row.get(f"{label}_nonagentic_total_tokens"))
            ag_p = safe_int(row.get(f"{label}_agentic_prompt_tokens_total"))
            ag_c = safe_int(row.get(f"{label}_agentic_completion_tokens_total"))
            ag_t = safe_int(row.get(f"{label}_agentic_total_tokens"))
            ag_pk = safe_int(row.get(f"{label}_agentic_peak_prompt_tokens"))
            tc = safe_int(row.get(f"{label}_agentic_tool_calls_made")) or 0
            mult = round(ag_t / na_t, 1) if ag_t and na_t and na_t > 0 else None

            token_rows.append({
                "ID": row.get("id", ""),
                "Lang": row.get("language", ""),
                "Policy": label,
                "Tool calls": tc,
                "Non-ag prompt": na_p,
                "Non-ag completion": na_c,
                "Non-ag total": na_t,
                "Ag prompt total": ag_p,
                "Ag completion total": ag_c,
                "Ag total": ag_t,
                "Ag peak context": ag_pk,
                "Ag/Non-ag ×": mult,
            })

    df_tok = pd.DataFrame(token_rows)
    st.dataframe(df_tok, use_container_width=True, height=350)

    st.markdown("<hr class='custom-hr'>", unsafe_allow_html=True)

    # ── Bar chart: non-ag vs agentic totals ───────────────────────────────────
    st.markdown("## Non-Agentic vs Agentic Tokens per Scenario")
    chart_policy = st.radio("Policy", policy_labels, horizontal=True, key="tok_chart_policy")

    chart_rows = []
    for row in valid_rows:
        sid = f"[{row.get('id','')}] {row.get('language','')}"
        na_t = safe_int(row.get(f"{chart_policy}_nonagentic_total_tokens"))
        ag_t = safe_int(row.get(f"{chart_policy}_agentic_total_tokens"))
        if na_t:
            chart_rows.append({"Scenario": sid, "Type": "Non-agentic", "Tokens": na_t})
        if ag_t:
            chart_rows.append({"Scenario": sid, "Type": "Agentic", "Tokens": ag_t})

    if chart_rows:
        df_ch = pd.DataFrame(chart_rows)
        bar = alt.Chart(df_ch).mark_bar().encode(
            x=alt.X("Scenario:O", sort=None, title="Scenario"),
            y=alt.Y("Tokens:Q"),
            color=alt.Color("Type:N", scale=alt.Scale(
                domain=["Non-agentic", "Agentic"],
                range=["#94A3B8", model_color(guardrail_raw)],
            )),
            xOffset="Type:N",
            tooltip=["Scenario", "Type", "Tokens"],
        ).properties(height=320)
        st.altair_chart(bar, use_container_width=True)
    else:
        st.info("No token data available for this policy.")

    # ── Context window growth ─────────────────────────────────────────────────
    st.markdown("<hr class='custom-hr'>", unsafe_allow_html=True)
    st.markdown("## Context Window Growth per Turn")
    st.markdown(
        '<p class="caption-text">Shows how the context window fills up as tool results are appended. '
        'Each bar = one LLM call in the agentic loop.</p>',
        unsafe_allow_html=True,
    )

    growth_opts = {
        f"[{r.get('id',i)}] ({r.get('language','?')}) {str(r.get('scenario',''))[:60]}…": i
        for i, r in enumerate(valid_rows)
    }
    sel_growth = st.selectbox("Scenario", list(growth_opts.keys()), key="growth_sel")
    row_growth = valid_rows[growth_opts[sel_growth]]
    growth_policy = st.radio("Policy", policy_labels, horizontal=True, key="growth_policy")

    per_turn = ensure_list(row_growth.get(f"{growth_policy}_agentic_token_usage_per_turn", []))
    valid_per_turn = [t for t in per_turn if t.get("prompt_tokens") is not None]

    if valid_per_turn:
        df_growth = pd.DataFrame(valid_per_turn)
        df_growth["turn_label"] = df_growth["turn"].astype(str)
        df_growth["prompt_tokens"] = df_growth["prompt_tokens"].astype(int)
        df_growth["completion_tokens"] = df_growth["completion_tokens"].astype(int)
        growth_chart = alt.Chart(df_growth).transform_fold(
            ["prompt_tokens", "completion_tokens"],
            as_=["Type", "Tokens"],
        ).mark_bar().encode(
            x=alt.X("turn_label:O", title="Turn (LLM call #)"),
            y=alt.Y("Tokens:Q"),
            color=alt.Color("Type:N", scale=alt.Scale(
                domain=["prompt_tokens", "completion_tokens"],
                range=["#3B82F6", "#F59E0B"],
            )),
            tooltip=["turn_label:O", "Type:N", "Tokens:Q"],
        ).properties(height=260)
        st.altair_chart(growth_chart, use_container_width=True)

        with st.expander("Raw per-turn data"):
            st.dataframe(pd.DataFrame(per_turn), use_container_width=True)
    else:
        st.markdown(
            '<div class="info-card">No per-turn token data for this scenario/policy — '
            'either no tool calls were made or the provider did not return usage metadata.</div>',
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 7 — JUDGMENT TIMING
# ══════════════════════════════════════════════════════════════════════════════
with tab_timing:
    st.markdown("<h1>Judgment Timing</h1>", unsafe_allow_html=True)
    st.markdown(
        '<p class="caption-text">'
        'Wall-clock seconds from when the full text (policy + rubric + scenario + response) '
        'entered the judge model until it produced a score and pass/fail verdict. '
        '<strong>Non-agentic</strong>: single LLM call (pure inference latency). '
        '<strong>Agentic</strong>: all LLM turns + all tool calls (search, fetch, URL checks) + post-loop URL sweep. '
        'Requires output files produced after timing was added to the pipeline.'
        '</p>',
        unsafe_allow_html=True,
    )
    st.markdown("<hr class='custom-hr'>", unsafe_allow_html=True)

    # Collect timing data across all policies
    _timing_has_data = False
    for _tlabel in policy_labels:
        _na_times = [safe_float(r.get(f"{_tlabel}_nonagentic_judgment_time_s")) for r in valid_rows]
        _ag_times = [safe_float(r.get(f"{_tlabel}_agentic_judgment_time_s")) for r in valid_rows]
        _na_times_v = [t for t in _na_times if t is not None]
        _ag_times_v = [t for t in _ag_times if t is not None]

        if not _na_times_v and not _ag_times_v:
            continue
        _timing_has_data = True

        st.markdown(
            f'<div class="section-label">Policy: <code>{_tlabel}</code></div>',
            unsafe_allow_html=True,
        )

        # ── Summary KPI row ───────────────────────────────────────────────────
        _avg_na_t = avg(_na_times_v)
        _avg_ag_t = avg(_ag_times_v)
        _max_ag_t = max(_ag_times_v) if _ag_times_v else None
        _min_ag_t = min(_ag_times_v) if _ag_times_v else None
        _mult     = _avg_ag_t / _avg_na_t if _avg_ag_t and _avg_na_t and _avg_na_t > 0 else None

        tm1, tm2, tm3, tm4, tm5 = st.columns(5)
        tm1.metric(
            "Avg non-ag time",
            f"{_avg_na_t:.2f}s" if _avg_na_t is not None else "—",
            help="Average wall-clock time for a single non-agentic LLM judge call",
        )
        tm2.metric(
            "Avg agentic time",
            f"{_avg_ag_t:.2f}s" if _avg_ag_t is not None else "—",
            help="Average total time for the full agentic loop (all LLM turns + all tool calls)",
        )
        tm3.metric(
            "Max agentic time",
            f"{_max_ag_t:.2f}s" if _max_ag_t is not None else "—",
            help="Slowest agentic evaluation — often scenarios with many tool calls",
        )
        tm4.metric(
            "Min agentic time",
            f"{_min_ag_t:.2f}s" if _min_ag_t is not None else "—",
        )
        tm5.metric(
            "Agentic / Non-ag",
            f"{_mult:.1f}×" if _mult is not None else "—",
            help="How many times slower the agentic path is on average",
        )

        # ── Bar chart: non-ag vs agentic time per scenario ────────────────────
        _time_chart_rows = []
        for _r in valid_rows:
            _sid = f"[{_r.get('id','')}] {_r.get('language','')}"
            _nat = safe_float(_r.get(f"{_tlabel}_nonagentic_judgment_time_s"))
            _agt = safe_float(_r.get(f"{_tlabel}_agentic_judgment_time_s"))
            if _nat is not None:
                _time_chart_rows.append({"Scenario": _sid, "Mode": "Non-agentic", "Time (s)": _nat})
            if _agt is not None:
                _time_chart_rows.append({"Scenario": _sid, "Mode": "Agentic", "Time (s)": _agt})

        if _time_chart_rows:
            _df_tc = pd.DataFrame(_time_chart_rows)
            _time_bar = alt.Chart(_df_tc).mark_bar().encode(
                x=alt.X("Scenario:O", sort=None, title="Scenario"),
                y=alt.Y("Time (s):Q", title="Seconds"),
                color=alt.Color("Mode:N", scale=alt.Scale(
                    domain=["Non-agentic", "Agentic"],
                    range=["#94A3B8", model_color(guardrail_raw)],
                )),
                xOffset="Mode:N",
                tooltip=["Scenario:O", "Mode:N", "Time (s):Q"],
            ).properties(height=280).configure_view(strokeOpacity=0).configure_axis(
                labelFont="Inter", titleFont="Inter", labelFontSize=11, titleFontSize=11,
            )
            st.markdown("**Time per scenario — non-agentic vs agentic**")
            st.altair_chart(_time_bar, use_container_width=True)

        # ── Scatter: agentic time vs tool calls made ──────────────────────────
        _scatter_rows = []
        for _r in valid_rows:
            _agt = safe_float(_r.get(f"{_tlabel}_agentic_judgment_time_s"))
            _tc  = safe_float(_r.get(f"{_tlabel}_agentic_tool_calls_made"))
            _sid = str(_r.get("id", ""))
            _lang = _r.get("language", "")
            if _agt is not None and _tc is not None:
                _scatter_rows.append({
                    "Agentic time (s)": _agt,
                    "Tool calls": _tc,
                    "ID": _sid,
                    "Lang": _lang,
                })
        if _scatter_rows:
            _df_sc = pd.DataFrame(_scatter_rows)
            _scatter_chart = alt.Chart(_df_sc).mark_circle(size=80, opacity=0.75).encode(
                x=alt.X("Tool calls:Q", scale=alt.Scale(domain=[-0.5, max(_df_sc["Tool calls"]) + 0.5])),
                y=alt.Y("Agentic time (s):Q"),
                color=alt.value(model_color(guardrail_raw)),
                tooltip=["ID:N", "Lang:N", "Tool calls:Q", "Agentic time (s):Q"],
            ).properties(height=240).configure_view(strokeOpacity=0).configure_axis(
                labelFont="Inter", titleFont="Inter", labelFontSize=11, titleFontSize=11,
            )
            st.markdown("**Agentic time vs tool calls made**")
            st.altair_chart(_scatter_chart, use_container_width=True)

        st.markdown("<hr class='custom-hr'>", unsafe_allow_html=True)

    if not _timing_has_data:
        st.info(
            "No timing data found in this output file. "
            "Re-run the evaluation pipeline to generate timing columns "
            "(`_nonagentic_judgment_time_s` and `_agentic_judgment_time_s`)."
        )
    else:
        # ── Per-scenario timing table ─────────────────────────────────────────
        st.markdown("## Per-Scenario Timing Table")
        _time_table_rows = []
        for _r in valid_rows:
            for _tlabel in policy_labels:
                _nat = safe_float(_r.get(f"{_tlabel}_nonagentic_judgment_time_s"))
                _agt = safe_float(_r.get(f"{_tlabel}_agentic_judgment_time_s"))
                _tc  = safe_float(_r.get(f"{_tlabel}_agentic_tool_calls_made"))
                _mult_row = (
                    round(_agt / _nat, 1) if _agt is not None and _nat is not None and _nat > 0
                    else None
                )
                _time_table_rows.append({
                    "ID": _r.get("id", ""),
                    "Lang": _r.get("language", ""),
                    "Policy": _tlabel,
                    "Tool calls": int(_tc) if _tc is not None else None,
                    "Non-ag time (s)": round(_nat, 3) if _nat is not None else None,
                    "Agentic time (s)": round(_agt, 3) if _agt is not None else None,
                    "Agentic / Non-ag ×": _mult_row,
                })

        _df_time_tbl = pd.DataFrame(_time_table_rows)
        st.dataframe(_df_time_tbl, use_container_width=True, height=350)

        st.download_button(
            "⬇️  Download timing CSV",
            _df_time_tbl.to_csv(index=False).encode("utf-8"),
            file_name=f"{selected_file.replace('.json', '')}_judgment_timing.csv",
            mime="text/csv",
        )

        # ── Cross-model timing comparison (if multiple files available) ───────
        st.markdown("<hr class='custom-hr'>", unsafe_allow_html=True)
        st.markdown("## Cross-Model Timing Comparison")
        st.markdown(
            '<p class="caption-text">Load multiple output files (one per judge model) '
            'to compare judgment latency across Claude, Gemini, and GPT.</p>',
            unsafe_allow_html=True,
        )

        @st.cache_data(show_spinner=False)
        def _load_timing_group(paths_str: str) -> dict[str, list[dict]]:
            result = {}
            for _p in paths_str.split("|"):
                try:
                    with open(_p) as _fh:
                        _d = json.load(_fh)
                    _vrows = [_r for _r in _d if any(_k.endswith("_nonagentic_score") for _k in _r)]
                    if _vrows and any(
                        _r.get(f"{_lbl}_nonagentic_judgment_time_s") is not None
                        for _r in _vrows
                        for _lbl in detect_policy_labels(_vrows[0])
                    ):
                        result[Path(_p).name] = _vrows
                except Exception:
                    pass
            return result

        _all_timing_data = _load_timing_group("|".join(str(_f) for _f in json_files))

        if len(_all_timing_data) >= 2:
            _cross_policy = st.radio(
                "Policy", policy_labels, horizontal=True, key="timing_cross_pol"
            )
            _cross_rows = []
            for _fname, _frows in _all_timing_data.items():
                _gm = _frows[0].get("guardrail_model", _fname)
                _gm_label = pretty_guardrail(_gm) if _gm else pretty_file_label(_fname)
                _fn_times = [
                    safe_float(_r.get(f"{_cross_policy}_nonagentic_judgment_time_s"))
                    for _r in _frows
                ]
                _ag_times_f = [
                    safe_float(_r.get(f"{_cross_policy}_agentic_judgment_time_s"))
                    for _r in _frows
                ]
                _fn_times_v = [t for t in _fn_times if t is not None]
                _ag_times_fv = [t for t in _ag_times_f if t is not None]
                for _t in _fn_times_v:
                    _cross_rows.append({"Model": _gm_label, "Mode": "Non-agentic", "Time (s)": _t})
                for _t in _ag_times_fv:
                    _cross_rows.append({"Model": _gm_label, "Mode": "Agentic", "Time (s)": _t})

            if _cross_rows:
                _df_cross_t = pd.DataFrame(_cross_rows)
                _cross_mode = st.radio(
                    "Mode", ["Non-agentic", "Agentic", "Both"], horizontal=True, key="timing_cross_mode"
                )
                if _cross_mode != "Both":
                    _df_cross_t = _df_cross_t[_df_cross_t["Mode"] == _cross_mode]

                _box_cross = alt.Chart(_df_cross_t).mark_boxplot(extent="min-max", size=40).encode(
                    x=alt.X("Model:N", title="Judge model"),
                    y=alt.Y("Time (s):Q"),
                    color=alt.Color("Mode:N", scale=alt.Scale(
                        domain=["Non-agentic", "Agentic"],
                        range=["#94A3B8", "#3B82F6"],
                    )),
                ).properties(height=300)
                st.altair_chart(_box_cross, use_container_width=True)
            else:
                st.info("No timing data found across the loaded files.")
        else:
            st.info(
                "Only one output file with timing data found. "
                "Run evaluations with multiple judge models to enable cross-model timing comparison."
            )
