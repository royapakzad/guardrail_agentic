"""
visualize_results.py  —  Multilingual LLM Guardrail Evaluation Dashboard
Usage: streamlit run visualize_results.py
"""
from __future__ import annotations

import json
import re
from collections import Counter
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

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
html, body, [class*="css"], .stApp, .stMarkdown, p, span, div, td, th, label {
    font-family: 'Inter', sans-serif !important;
}
#MainMenu, footer, header { visibility: hidden; }

[data-testid="metric-container"] {
    background: #fff; border: 1px solid #E2E8F0; border-radius: 12px;
    padding: 16px 18px !important; box-shadow: 0 1px 3px rgba(0,0,0,.05);
}
[data-testid="stMetricValue"]  { font-size:1.6rem !important; font-weight:800 !important; color:#0F172A !important; }
[data-testid="stMetricLabel"]  { font-size:.70rem !important; font-weight:600 !important; color:#64748B !important; text-transform:uppercase; letter-spacing:.06em; }
[data-testid="stMetricDelta"]  { font-size:.80rem !important; font-weight:500 !important; }

[data-testid="stSidebar"] { background:#0F2448 !important; }
[data-testid="stSidebar"] .stMarkdown,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] div { color:#CBD5E1 !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color:#F1F5F9 !important; font-weight:700 !important; }
[data-testid="stSidebar"] .stSelectbox > div > div { background:#1E3A5F !important; border-color:#334155 !important; color:#E2E8F0 !important; }
[data-testid="stSidebar"] hr { border-color:#1E3A5F !important; }

.stTabs [data-baseweb="tab-list"] { gap:4px; border-bottom:2px solid #E2E8F0; }
.stTabs [data-baseweb="tab"] { font-weight:500; font-size:.88rem; color:#64748B; padding:8px 16px; border-radius:8px 8px 0 0; }
.stTabs [aria-selected="true"] { color:#1B3A6B !important; font-weight:700 !important; border-bottom:2px solid #3B82F6 !important; }

h1 { font-size:1.45rem !important; font-weight:800 !important; color:#0F172A !important; letter-spacing:-.03em; }
h2 { font-size:1.10rem !important; font-weight:700 !important; color:#1E293B !important; }
h3 { font-size:.92rem  !important; font-weight:600 !important; color:#334155 !important; }

[data-testid="stDataFrame"] th { font-weight:600 !important; font-size:.74rem !important; text-transform:uppercase; letter-spacing:.05em; color:#475569 !important; background:#F8FAFC !important; }

.badge-pass { display:inline-block; background:#DCFCE7; color:#166534; font-weight:700; font-size:.70rem; padding:2px 8px; border-radius:999px; }
.badge-fail { display:inline-block; background:#FEE2E2; color:#991B1B; font-weight:700; font-size:.70rem; padding:2px 8px; border-radius:999px; }
.score-big  { font-size:2.8rem; font-weight:800; line-height:1; letter-spacing:-.04em; }
.label-sm   { font-size:.70rem; font-weight:600; text-transform:uppercase; letter-spacing:.08em; color:#94A3B8; margin-bottom:6px; }
.alert-changed { background:#FFF7ED; border:1px solid #FDBA74; border-left:4px solid #F97316; border-radius:8px; padding:10px 14px; color:#7C2D12; font-weight:500; margin:8px 0 12px 0; }
.info-card { background:#F0F9FF; border:1px solid #BAE6FD; border-radius:10px; padding:12px 16px; margin:6px 0; font-size:.86rem; color:#0C4A6E; }
.chip { display:inline-block; padding:3px 10px; border-radius:999px; font-size:.75rem; font-weight:600; margin:2px; }
</style>
""", unsafe_allow_html=True)

# ── CONSTANTS ──────────────────────────────────────────────────────────────────
_VALID_THRESHOLD = 0.6
_PALETTE = ["#7C3AED", "#1A73E8", "#10A37F", "#F59E0B", "#EF4444", "#EC4899", "#06B6D4"]
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

# ── HELPERS ────────────────────────────────────────────────────────────────────

def pretty_guardrail(raw: str) -> str:
    return _GUARDRAIL_DISPLAY.get(raw, raw or "Unknown")

def pretty_file_label(fname: str) -> str:
    n = fname.replace(".json", "")
    parts = []
    if "original_policies" in n:   parts.append("Original Policies")
    elif "concrete_policies" in n: parts.append("Concrete Policies")
    else: parts.append(n.split("_")[0].replace("-", " ").title())
    if   "claude_sonnet_4_6" in n: parts.append("Claude Sonnet 4.6")
    elif "claude_sonnet_4_5" in n: parts.append("Claude Sonnet 4.5")
    elif "gemini_2_5_flash"  in n: parts.append("Gemini 2.5 Flash")
    elif "gpt_5_nano"        in n: parts.append("GPT-5-nano")
    elif "all"               in n: parts.append("All Models")
    return "  ·  ".join(parts) if parts else fname

def detect_policy_labels(row: dict) -> list[str]:
    return sorted(k[: -len("_nonagentic_score")] for k in row if k.endswith("_nonagentic_score"))

def safe_float(val):
    try:    return float(val) if val is not None else None
    except: return None

def safe_int(val):
    try:    return int(val) if val is not None else None
    except: return None

def score_color(v) -> str:
    if v is None: return "#94A3B8"
    try:    f = float(v)
    except: return "#94A3B8"
    return "#10B981" if f >= 0.85 else ("#F59E0B" if f >= _VALID_THRESHOLD else "#EF4444")

def score_to_valid(score) -> Optional[bool]:
    if score is None: return None
    try:    return float(score) > _VALID_THRESHOLD
    except: return None

def fmt_valid_html(v) -> str:
    if v is True:  return '<span class="badge-pass">PASS</span>'
    if v is False: return '<span class="badge-fail">FAIL</span>'
    return "<span style='color:#94A3B8'>—</span>"

def fmt_valid(v) -> str:
    if v is True:  return "✅ PASS"
    if v is False: return "❌ FAIL"
    return "—"

def delta_arrow(delta) -> str:
    if delta is None: return "—"
    try:    d = float(delta)
    except: return "—"
    if d >  0.01: return f"▲ +{d:.3f}"
    if d < -0.01: return f"▼ {d:.3f}"
    return f"≈ {d:.3f}"

def ensure_list(val) -> list:
    if isinstance(val, list): return val
    if isinstance(val, str):
        try:
            p = json.loads(val)
            if isinstance(p, list): return p
        except: pass
    return []

def avg(lst):
    return sum(lst) / len(lst) if lst else None

def extract_urls_from_text(text: str) -> list[str]:
    urls = []
    for m in re.finditer(r'https?://\S+', text or "", re.IGNORECASE):
        url = m.group(0).rstrip('.,;:!?)]\'"<>')
        if url: urls.append(url)
    return urls

def extract_judge_research_urls(tool_log: list) -> list[str]:
    urls = []
    for call in tool_log:
        tool    = call.get("tool", "")
        inp     = call.get("input", {})
        preview = call.get("output_preview", "")
        if tool == "fetch_url":
            url = inp.get("url", "") if isinstance(inp, dict) else ""
            if url: urls.append(url)
        elif tool == "search_web":
            found: list[str] = []
            try:
                results = json.loads(preview) if isinstance(preview, str) else preview
                if isinstance(results, dict) and "results" in results:
                    results = results["results"]
                if isinstance(results, list):
                    for r in results:
                        url = r.get("url", "") if isinstance(r, dict) else ""
                        if url: found.append(url)
            except: pass
            if not found: found = extract_urls_from_text(str(preview))
            urls.extend(found)
    return urls

def url_to_domain(url: str) -> str:
    try:
        netloc = urlparse(url).netloc.lower()
        if netloc.startswith("www."): netloc = netloc[4:]
        return netloc or url
    except: return url

_AX = {"labelFont": "Inter", "titleFont": "Inter", "labelFontSize": 11, "titleFontSize": 11}

def _bare(chart, h: int = 260):
    """Apply minimal shared styling to a plain (non-faceted) chart."""
    return (
        chart.properties(height=h)
        .configure_view(strokeOpacity=0)
        .configure_axis(**_AX)
        .configure_legend(labelFont="Inter", titleFont="Inter", labelFontSize=11, titleFontSize=11)
    )

# ── SIDEBAR ────────────────────────────────────────────────────────────────────
st.sidebar.markdown("# 🛡️ Guardrail Explorer")
st.sidebar.markdown("---")

uploaded_files = st.sidebar.file_uploader(
    "Upload JSON output file(s)",
    type="json",
    accept_multiple_files=True,
    help="Upload one or more evaluation output JSON files to explore",
)

if not uploaded_files:
    st.markdown("""
    <div style='max-width:520px;margin:80px auto;text-align:center;'>
      <div style='font-size:3rem;margin-bottom:12px'>🛡️</div>
      <h1 style='font-size:1.6rem;color:#0F172A;margin-bottom:10px'>Guardrail Eval Explorer</h1>
      <p style='color:#64748B;font-size:.95rem;line-height:1.7'>
        Upload <code>.json</code> output files from your evaluation runs using the sidebar.<br><br>
        <strong>Single file</strong> — explore scores, scenarios, evidence for one run.<br>
        <strong>Multiple files</strong> — compare judge models, policies, and languages side by side.
      </p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── LOAD ALL FILES ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def _parse(content: bytes) -> list[dict]:
    return json.loads(content)

raw: dict[str, list[dict]] = {}
for f in uploaded_files:
    try:
        rows = _parse(f.read())
        valid = [r for r in rows if any(k.endswith("_nonagentic_score") for k in r)]
        if valid:
            raw[f.name] = valid
        else:
            st.sidebar.warning(f"`{f.name}`: no evaluated rows found")
    except Exception as e:
        st.sidebar.error(f"`{f.name}`: {e}")

if not raw:
    st.error("No valid evaluation data found in uploaded files.")
    st.stop()

# ── BUILD FILE → DISPLAY LABEL MAP ────────────────────────────────────────────
def _label(fname: str, rows: list[dict]) -> str:
    gm = rows[0].get("guardrail_model", "")
    return _GUARDRAIL_DISPLAY.get(gm, gm) if gm else pretty_file_label(fname)

file_labels: dict[str, str] = {f: _label(f, r) for f, r in raw.items()}
lc = Counter(file_labels.values())
for fname in list(file_labels):
    if lc[file_labels[fname]] > 1:
        file_labels[fname] += f" ({fname[:12]})"

model_list  = list(dict.fromkeys(file_labels.values()))
model_colors = {m: _PALETTE[i % len(_PALETTE)] for i, m in enumerate(model_list)}

# ── COLLECT GLOBAL METADATA ───────────────────────────────────────────────────
all_policies: list[str] = []
for rows in raw.values():
    for lbl in detect_policy_labels(rows[0]):
        if lbl not in all_policies:
            all_policies.append(lbl)

all_languages = sorted({r.get("language", "?") for rows in raw.values() for r in rows})

# ── SIDEBAR FILTERS ────────────────────────────────────────────────────────────
st.sidebar.markdown("### Filters")
sel_policies = st.sidebar.multiselect("Policy", all_policies, default=all_policies, key="gp")
sel_langs    = st.sidebar.multiselect("Language", all_languages, default=all_languages, key="gl")

st.sidebar.markdown("---")
st.sidebar.markdown("**Loaded files**")
for fname, rows in raw.items():
    lbl   = file_labels[fname]
    color = model_colors.get(lbl, "#3B82F6")
    asst  = rows[0].get("model", "?")
    st.sidebar.markdown(
        f'<span class="chip" style="background:{color}22;color:{color};border:1px solid {color}44">{lbl}</span>',
        unsafe_allow_html=True,
    )
    st.sidebar.caption(f"`{fname}`  \n{len(rows)} scenarios · asst: {asst}")

# ── UNIFIED FLAT DATAFRAME ─────────────────────────────────────────────────────
records = []
for fname, rows in raw.items():
    model = file_labels[fname]
    for row in rows:
        lang = row.get("language", "?")
        if sel_langs and lang not in sel_langs:
            continue
        for pol in detect_policy_labels(row):
            if sel_policies and pol not in sel_policies:
                continue
            na  = safe_float(row.get(f"{pol}_nonagentic_score"))
            ag  = safe_float(row.get(f"{pol}_agentic_score"))
            dlt = safe_float(row.get(f"{pol}_score_delta"))
            changed = row.get(f"{pol}_judgment_changed") is True
            tc  = safe_int(row.get(f"{pol}_agentic_tool_calls_made")) or 0
            records.append({
                "file": fname, "model": model,
                "id": str(row.get("id", "")), "language": lang, "policy": pol,
                "scenario": str(row.get("scenario", ""))[:72],
                "na_score": na, "ag_score": ag, "delta": dlt,
                "judgment_changed": changed, "tool_calls": tc,
                "na_valid": (na > _VALID_THRESHOLD) if na is not None else None,
                "ag_valid": (ag > _VALID_THRESHOLD) if ag is not None else None,
            })

df = pd.DataFrame(records)

if df.empty:
    st.warning("No data matches the current filters. Adjust language or policy in the sidebar.")
    st.stop()

# ── TABS ───────────────────────────────────────────────────────────────────────
tab_summary, tab_compare, tab_inspect, tab_evidence, tab_diag = st.tabs([
    "📊 Summary",
    "⚖️ Compare",
    "🔍 Inspect",
    "🔗 Evidence",
    "⚙️ Diagnostics",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
with tab_summary:
    st.markdown("## Evaluation Summary")

    # ── Top KPIs ──────────────────────────────────────────────────────────────
    na_vals  = df["na_score"].dropna().tolist()
    ag_vals  = df["ag_score"].dropna().tolist()
    n_changed = int(df["judgment_changed"].sum())

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Files loaded",        len(raw))
    k2.metric("Unique scenarios",    df["id"].nunique())
    k3.metric("Non-ag pass rate",
              f"{100*sum(1 for s in na_vals if s>_VALID_THRESHOLD)/len(na_vals):.0f}%" if na_vals else "—")
    k4.metric("Agentic pass rate",
              f"{100*sum(1 for s in ag_vals if s>_VALID_THRESHOLD)/len(ag_vals):.0f}%" if ag_vals else "—",
              delta=f"{(100*sum(1 for s in ag_vals if s>_VALID_THRESHOLD)/len(ag_vals)) - (100*sum(1 for s in na_vals if s>_VALID_THRESHOLD)/len(na_vals)):+.0f}pp" if na_vals and ag_vals else None)
    k5.metric("Judgments flipped",   f"{n_changed:,}",
              help="Scenarios where agentic changed the pass/fail verdict")

    st.markdown("---")

    # ── Score distributions ───────────────────────────────────────────────────
    st.markdown("### Score distributions — non-agentic vs agentic")
    st.caption("Each bar is one 0.1-wide score bin. Bars overlap (stack=None) so you can compare shapes. Dashed line = pass threshold (0.60).")

    d_na = df[df["na_score"].notna()][["model","na_score","policy","language"]].rename(columns={"na_score":"score"}); d_na["mode"] = "Non-agentic"
    d_ag = df[df["ag_score"].notna()][["model","ag_score","policy","language"]].rename(columns={"ag_score":"score"}); d_ag["mode"] = "Agentic"
    df_dist = pd.concat([d_na, d_ag], ignore_index=True)

    dc1, dc2 = st.columns(2)
    for col, mode_name in [(dc1, "Non-agentic"), (dc2, "Agentic")]:
        with col:
            st.markdown(f"**{mode_name}**")
            sub = df_dist[df_dist["mode"] == mode_name]
            if not sub.empty:
                hist = alt.Chart(sub).mark_bar(opacity=0.65, binSpacing=1).encode(
                    x=alt.X("score:Q", bin=alt.Bin(step=0.1), title="Score",
                            scale=alt.Scale(domain=[0, 1])),
                    y=alt.Y("count()", title="Scenarios", stack=None),
                    color=alt.Color("model:N",
                        scale=alt.Scale(domain=model_list,
                                        range=[model_colors[m] for m in model_list]),
                        legend=alt.Legend(title="Judge model", orient="bottom"),
                    ),
                    tooltip=["model:N", alt.Tooltip("score:Q", bin=alt.Bin(step=0.1)), "count()"],
                )
                rule = alt.Chart(pd.DataFrame({"x": [_VALID_THRESHOLD]})).mark_rule(
                    strokeDash=[4, 3], color="#0F172A", strokeWidth=1.5
                ).encode(x="x:Q")
                st.altair_chart(_bare(hist + rule, 220), use_container_width=True)

    st.markdown("---")

    # ── Pass rate by model × policy ───────────────────────────────────────────
    st.markdown("### Pass rates — model × policy × mode")

    pr_rows = []
    for (model, pol), grp in df.groupby(["model", "policy"]):
        na_s = grp["na_score"].dropna().tolist()
        ag_s = grp["ag_score"].dropna().tolist()
        if na_s:
            pr_rows.append({"Model": model, "Policy": pol, "Mode": "Non-agentic",
                             "Pass %": 100 * sum(1 for s in na_s if s > _VALID_THRESHOLD) / len(na_s),
                             "n": len(na_s)})
        if ag_s:
            pr_rows.append({"Model": model, "Policy": pol, "Mode": "Agentic",
                             "Pass %": 100 * sum(1 for s in ag_s if s > _VALID_THRESHOLD) / len(ag_s),
                             "n": len(ag_s)})

    if pr_rows:
        df_pr = pd.DataFrame(pr_rows)
        pr_chart = (
            alt.Chart(df_pr)
            .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
            .encode(
                x=alt.X("Model:N", title=None, axis=alt.Axis(labelAngle=-25, labelFont="Inter")),
                y=alt.Y("Pass %:Q", scale=alt.Scale(domain=[0, 100]),
                        axis=alt.Axis(titleFont="Inter", labelFont="Inter")),
                color=alt.Color("Mode:N",
                    scale=alt.Scale(domain=["Non-agentic", "Agentic"],
                                    range=["#94A3B8", "#3B82F6"]),
                    legend=alt.Legend(title="Mode", orient="bottom"),
                ),
                xOffset="Mode:N",
                column=alt.Column("Policy:N",
                    header=alt.Header(titleFont="Inter", labelFont="Inter")),
                tooltip=["Model:N", "Policy:N", "Mode:N",
                         alt.Tooltip("Pass %:Q", format=".1f"), "n:Q"],
            )
            .properties(height=240, width=alt.Step(28))
        )
        st.altair_chart(pr_chart)

    st.markdown("---")

    # ── Aggregate table ───────────────────────────────────────────────────────
    st.markdown("### Aggregate scores per model × policy")

    agg_rows = []
    for (model, pol), grp in df.groupby(["model", "policy"]):
        na_s   = grp["na_score"].dropna().tolist()
        ag_s   = grp["ag_score"].dropna().tolist()
        deltas = grp["delta"].dropna().tolist()
        n_ch   = int(grp["judgment_changed"].sum())
        n_scen = grp["id"].nunique()
        agg_rows.append({
            "Model": model, "Policy": pol, "Scenarios": n_scen,
            "Avg non-ag score":   round(avg(na_s), 3) if na_s else None,
            "Non-ag pass %":  f"{100*sum(1 for s in na_s if s>_VALID_THRESHOLD)/len(na_s):.0f}%" if na_s else "—",
            "Avg agentic score":  round(avg(ag_s), 3) if ag_s else None,
            "Agentic pass %": f"{100*sum(1 for s in ag_s if s>_VALID_THRESHOLD)/len(ag_s):.0f}%" if ag_s else "—",
            "Avg delta":          round(avg(deltas), 3) if deltas else None,
            "Judgments flipped":  f"{n_ch}/{n_scen}",
        })

    if agg_rows:
        df_agg = pd.DataFrame(agg_rows)

        def _cscore(val):
            try:
                v = float(val)
                if v >= 0.85: return "color:#10B981;font-weight:600"
                if v >= _VALID_THRESHOLD: return "color:#F59E0B;font-weight:600"
                return "color:#EF4444;font-weight:700"
            except: return ""

        st.dataframe(
            df_agg.style.map(_cscore, subset=["Avg non-ag score", "Avg agentic score", "Avg delta"]),
            use_container_width=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — COMPARE
# ══════════════════════════════════════════════════════════════════════════════
with tab_compare:
    st.markdown("## Cross-Model & Cross-Policy Comparison")

    # ── Agentic vs Non-agentic scatter ────────────────────────────────────────
    st.markdown("### Agentic vs non-agentic scores")
    st.caption("Each point = one scenario × policy. **Above diagonal** = agentic scored higher. **Below** = non-agentic scored higher. Dashed lines mark the 0.60 pass threshold.")

    df_scat = df[df["na_score"].notna() & df["ag_score"].notna()].copy()
    if not df_scat.empty:
        diag     = pd.DataFrame({"x": [0, 1], "y": [0, 1]})
        diag_ln  = alt.Chart(diag).mark_line(strokeDash=[5, 3], color="#CBD5E1", strokeWidth=1.5).encode(x="x:Q", y="y:Q")
        thresh_h = alt.Chart(pd.DataFrame({"y": [_VALID_THRESHOLD]})).mark_rule(strokeDash=[3, 3], color="#E2E8F0", strokeWidth=1).encode(y="y:Q")
        thresh_v = alt.Chart(pd.DataFrame({"x": [_VALID_THRESHOLD]})).mark_rule(strokeDash=[3, 3], color="#E2E8F0", strokeWidth=1).encode(x="x:Q")

        scatter = alt.Chart(df_scat).mark_circle(size=72, opacity=0.72).encode(
            x=alt.X("na_score:Q", title="Non-agentic score", scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("ag_score:Q",  title="Agentic score",    scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("model:N",
                scale=alt.Scale(domain=model_list, range=[model_colors[m] for m in model_list]),
                legend=alt.Legend(title="Judge model"),
            ),
            shape=alt.Shape("language:N", legend=alt.Legend(title="Scenario language")),
            tooltip=["model:N", "language:N", "policy:N", "id:N",
                     alt.Tooltip("na_score:Q", format=".3f"),
                     alt.Tooltip("ag_score:Q", format=".3f"),
                     alt.Tooltip("delta:Q",    format="+.3f"),
                     "judgment_changed:N"],
        )
        st.altair_chart(_bare(diag_ln + thresh_h + thresh_v + scatter, 320), use_container_width=True)
    else:
        st.info("No scenarios with both non-agentic and agentic scores in the current filter.")

    st.markdown("---")

    # ── Delta distribution ────────────────────────────────────────────────────
    st.markdown("### Score delta distribution  (agentic − non-agentic)")
    st.caption("Positive = agentic gave a higher score; negative = lower. Vertical line at zero.")

    df_dlt = df[df["delta"].notna()]
    if not df_dlt.empty:
        d_hist = alt.Chart(df_dlt).mark_bar(opacity=0.70, binSpacing=1).encode(
            x=alt.X("delta:Q", bin=alt.Bin(step=0.05), title="Score delta"),
            y=alt.Y("count()", title="Scenarios", stack=None),
            color=alt.Color("model:N",
                scale=alt.Scale(domain=model_list, range=[model_colors[m] for m in model_list]),
                legend=alt.Legend(title="Judge model"),
            ),
            tooltip=["model:N", "count()"],
        )
        zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(
            color="#0F172A", strokeWidth=2, strokeDash=[3, 2]
        ).encode(x="x:Q")
        st.altair_chart(_bare(d_hist + zero, 210), use_container_width=True)

    st.markdown("---")

    # ── Score heatmap (most useful with 2+ files) ─────────────────────────────
    st.markdown("### Score heatmap — scenarios × models")
    st.caption("Each cell = score for one scenario / judge model combination. Red = fail, green = pass. Identical scenarios across files are aligned on the same row.")

    hc1, hc2, hc3 = st.columns([2, 2, 2])
    with hc1:
        heat_mode   = st.radio("Mode", ["Non-agentic", "Agentic"], horizontal=True, key="hm")
    with hc2:
        heat_policy = st.selectbox("Policy", all_policies, key="hp") if all_policies else None
    with hc3:
        heat_lang   = st.multiselect("Language", all_languages, default=all_languages, key="hl")

    score_col = "na_score" if heat_mode == "Non-agentic" else "ag_score"

    if heat_policy:
        df_h = df[(df["policy"] == heat_policy) & (df["language"].isin(heat_lang) if heat_lang else True)].copy()
        df_h = df_h[df_h[score_col].notna()]
        MAX_ROWS = 80
        visible_ids = df_h["id"].unique()[:MAX_ROWS]
        df_h = df_h[df_h["id"].isin(visible_ids)].copy()
        df_h["row_label"] = df_h.apply(
            lambda r: f"[{r['id']}] {r['language']} · {r['scenario'][:38]}…", axis=1
        )
        df_h["score_val"] = df_h[score_col]

        if not df_h.empty:
            heatmap = (
                alt.Chart(df_h)
                .mark_rect()
                .encode(
                    x=alt.X("model:N", title="Judge model",
                            axis=alt.Axis(labelAngle=-20, labelFont="Inter", titleFont="Inter")),
                    y=alt.Y("row_label:N", title=None, sort=None,
                            axis=alt.Axis(labelFont="Inter", labelFontSize=10)),
                    color=alt.Color("score_val:Q", title="Score",
                        scale=alt.Scale(scheme="redyellowgreen", domain=[0, 1]),
                        legend=alt.Legend(titleFont="Inter", labelFont="Inter"),
                    ),
                    tooltip=["model:N", "row_label:N",
                             alt.Tooltip("score_val:Q", format=".3f"),
                             "judgment_changed:N"],
                )
                .properties(height=max(320, len(df_h["row_label"].unique()) * 18 + 40))
                .configure_view(strokeOpacity=0)
                .configure_legend(labelFont="Inter", titleFont="Inter")
            )
            st.altair_chart(heatmap, use_container_width=True)
            if df["id"].nunique() > MAX_ROWS:
                st.caption(f"Showing first {MAX_ROWS} of {df['id'].nunique()} scenarios. Use language filter to narrow down.")

    st.markdown("---")

    # ── Policy language comparison ────────────────────────────────────────────
    en_pols  = [l for l in all_policies if not any(l.endswith(s) for s in ("_fa","_ar","_es","_de","_fr","_zh"))]
    alt_pols = [l for l in all_policies if l not in en_pols]

    if en_pols and alt_pols:
        st.markdown("### Policy language comparison")
        st.caption("Same guardrail, same response — does the language of the policy text shift the score? Points above diagonal = non-English policy scored higher.")

        pc1, pc2, pc3 = st.columns([2, 2, 2])
        with pc1: sel_en  = st.selectbox("English policy",      en_pols,  key="sep")
        with pc2: sel_alt = st.selectbox("Non-English policy",  alt_pols, key="sap")
        with pc3: pol_mode = st.radio("Mode", ["Non-agentic", "Agentic"], horizontal=True, key="pm")

        sk = "na_score" if pol_mode == "Non-agentic" else "ag_score"
        df_ep = df[df["policy"] == sel_en][["model","id","language",sk]].rename(columns={sk:"en_score"})
        df_ap = df[df["policy"] == sel_alt][["model","id","language",sk]].rename(columns={sk:"alt_score"})
        df_pol = df_ep.merge(df_ap, on=["model","id","language"]).dropna(subset=["en_score","alt_score"])

        if not df_pol.empty:
            diag2    = pd.DataFrame({"x":[0,1],"y":[0,1]})
            diag_ln2 = alt.Chart(diag2).mark_line(strokeDash=[5,3],color="#CBD5E1",strokeWidth=1.5).encode(x="x:Q",y="y:Q")
            pol_sc   = alt.Chart(df_pol).mark_circle(size=72,opacity=0.75).encode(
                x=alt.X("en_score:Q",  title=f"{sel_en} score",  scale=alt.Scale(domain=[0,1])),
                y=alt.Y("alt_score:Q", title=f"{sel_alt} score", scale=alt.Scale(domain=[0,1])),
                color=alt.Color("model:N",
                    scale=alt.Scale(domain=model_list,range=[model_colors[m] for m in model_list]),
                    legend=alt.Legend(title="Judge model"),
                ),
                shape=alt.Shape("language:N"),
                tooltip=["model:N","id:N","language:N",
                         alt.Tooltip("en_score:Q", format=".3f"),
                         alt.Tooltip("alt_score:Q",format=".3f")],
            )
            st.altair_chart(_bare(diag_ln2 + pol_sc, 300), use_container_width=True)

            df_pol["diff"] = df_pol["en_score"] - df_pol["alt_score"]
            avg_d = avg(df_pol["diff"].tolist())
            abs_d = avg(df_pol["diff"].abs().tolist())
            st.caption(
                f"Avg (EN − {sel_alt}) = **{avg_d:+.3f}** &nbsp;·&nbsp; "
                f"Avg |diff| = **{abs_d:.3f}** &nbsp;·&nbsp; "
                f"EN higher (>0.05) in {(df_pol['diff']>0.05).sum()} / {len(df_pol)} cases &nbsp;·&nbsp; "
                f"{sel_alt} higher in {(df_pol['diff']<-0.05).sum()} cases"
            )
        else:
            st.info("Not enough overlapping scenarios to compare these two policies.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — INSPECT
# ══════════════════════════════════════════════════════════════════════════════
with tab_inspect:
    st.markdown("## Scenario Browser")

    # ── Filters + Table ───────────────────────────────────────────────────────
    fc1, fc2, fc3, fc4 = st.columns([2, 2, 2, 2])
    with fc1: ins_lang    = st.multiselect("Language", all_languages, default=all_languages, key="il")
    with fc2: ins_pol     = st.selectbox("Policy", all_policies, key="ipl") if all_policies else None
    with fc3: ins_mode    = st.radio("Score", ["Non-agentic","Agentic"], horizontal=True, key="im")
    with fc4: ins_changed = st.checkbox("Judgment changed only", key="ic")

    df_ins = df.copy()
    if ins_lang:  df_ins = df_ins[df_ins["language"].isin(ins_lang)]
    if ins_pol:   df_ins = df_ins[df_ins["policy"] == ins_pol]
    if ins_changed: df_ins = df_ins[df_ins["judgment_changed"] == True]

    sk_ins = "na_score" if ins_mode == "Non-agentic" else "ag_score"

    tbl = []
    for _, r in df_ins.iterrows():
        na_s = r.get("na_score"); ag_s = r.get("ag_score"); d = r.get("delta")
        tbl.append({
            "ID": r["id"], "Lang": r["language"], "Model": r["model"], "Policy": r["policy"],
            "Scenario": r["scenario"][:68],
            "Non-ag": round(na_s, 3) if na_s is not None else None,
            "✓ non-ag": fmt_valid(score_to_valid(na_s)),
            "Agentic": round(ag_s, 3) if ag_s is not None else None,
            "✓ agentic": fmt_valid(score_to_valid(ag_s)),
            "Δ": delta_arrow(d),
            "⚡": "yes" if r.get("judgment_changed") else "",
            "Tools": r.get("tool_calls", 0),
        })

    df_tbl = pd.DataFrame(tbl)

    def _hl(row): return ["background:#FFF7ED"]*len(row) if row["⚡"]=="yes" else [""]*len(row)
    def _cs(val):
        try:
            v = float(val)
            if v >= 0.85: return "color:#10B981;font-weight:600"
            if v >= _VALID_THRESHOLD: return "color:#F59E0B;font-weight:600"
            return "color:#EF4444;font-weight:700"
        except: return ""

    st.dataframe(
        df_tbl.style.apply(_hl, axis=1).map(_cs, subset=["Non-ag","Agentic"]),
        use_container_width=True, height=300,
    )

    st.markdown("---")
    st.markdown("### Single scenario deep-dive")

    dc1, dc2, dc3 = st.columns([3, 2, 2])
    with dc1:
        all_ids = sorted(df["id"].unique())
        sel_sid = st.selectbox(
            "Scenario ID", all_ids,
            format_func=lambda s: f"[{s}] {df[df['id']==s]['scenario'].iloc[0][:52]}…" if not df[df['id']==s].empty else s,
            key="ds",
        )
    with dc2:
        sel_file = st.selectbox("File", list(raw.keys()), format_func=lambda f: file_labels[f], key="df")
    with dc3:
        sel_pol_d = st.selectbox("Policy", all_policies, key="dp") if all_policies else None

    det_rows = [r for r in raw[sel_file] if str(r.get("id","")) == str(sel_sid)]
    if not det_rows:
        st.warning(f"Scenario {sel_sid} not found in {file_labels[sel_file]}.")
    elif not sel_pol_d:
        st.warning("No policy selected.")
    else:
        det = det_rows[0]
        lbl = sel_pol_d

        # Scenario + Response
        s1, s2 = st.columns(2)
        with s1:
            st.markdown('<p class="label-sm">Scenario</p>', unsafe_allow_html=True)
            st.info(det.get("scenario","—"))
            st.caption(f"Language: **{det.get('language','?')}** · ID: {det.get('id','?')}")
        with s2:
            st.markdown('<p class="label-sm">Assistant Response</p>', unsafe_allow_html=True)
            st.success(det.get("assistant_response","—"))
            st.caption(f"Model: **{det.get('model','?')}**")

        st.markdown("---")

        # Verdicts
        na_s  = safe_float(det.get(f"{lbl}_nonagentic_score"))
        ag_s  = safe_float(det.get(f"{lbl}_agentic_score"))
        dlt   = safe_float(det.get(f"{lbl}_score_delta"))
        chgd  = det.get(f"{lbl}_judgment_changed")
        tc    = det.get(f"{lbl}_agentic_tool_calls_made", 0) or 0

        if chgd:
            nv = "PASS" if score_to_valid(na_s) else "FAIL"
            av = "PASS" if score_to_valid(ag_s) else "FAIL"
            st.markdown(
                f'<div class="alert-changed">⚡ Judgment changed — Non-agentic: <strong>{nv}</strong> → Agentic: <strong>{av}</strong></div>',
                unsafe_allow_html=True,
            )

        v1, _mid, v2 = st.columns([5,1,5])
        with v1:
            st.markdown('<p class="label-sm">🔒 Non-agentic · single LLM call · no tools</p>', unsafe_allow_html=True)
            st.markdown(f'<div class="score-big" style="color:{score_color(na_s)}">{f"{na_s:.3f}" if na_s is not None else "—"}</div>', unsafe_allow_html=True)
            st.markdown(fmt_valid_html(score_to_valid(na_s)), unsafe_allow_html=True)
            st.markdown(f"> {det.get(f'{lbl}_nonagentic_explanation','—')}")
        with _mid:
            st.markdown("<br><br><br><div style='text-align:center;font-size:1.5rem;color:#CBD5E1'>↔</div>", unsafe_allow_html=True)
        with v2:
            st.markdown(f'<p class="label-sm">🌐 Agentic · {tc} tool call(s)</p>', unsafe_allow_html=True)
            st.markdown(f'<div class="score-big" style="color:{score_color(ag_s)}">{f"{ag_s:.3f}" if ag_s is not None else "—"}</div>', unsafe_allow_html=True)
            st.markdown(fmt_valid_html(score_to_valid(ag_s)), unsafe_allow_html=True)
            if dlt is not None:
                dc = "#10B981" if dlt > 0.01 else ("#EF4444" if dlt < -0.01 else "#94A3B8")
                st.markdown(f'<span style="color:{dc};font-weight:600;font-size:.9rem">{delta_arrow(dlt)} vs non-agentic</span>', unsafe_allow_html=True)
            st.markdown(f"> {det.get(f'{lbl}_agentic_explanation','—')}")

        st.markdown("---")

        # Claim checks
        claims = ensure_list(det.get(f"{lbl}_agentic_claim_checks", []))
        _SICON = {"verified":"✅","contradicted":"❌","unverifiable":"⚠️"}
        _SORD  = {"contradicted":0,"unverifiable":1,"verified":2}
        if claims:
            sc = sorted(claims, key=lambda c: _SORD.get(str(c.get("status","")).lower(), 3))
            nv2 = sum(1 for c in claims if str(c.get("status","")).lower()=="verified")
            nc  = sum(1 for c in claims if str(c.get("status","")).lower()=="contradicted")
            nu  = sum(1 for c in claims if str(c.get("status","")).lower()=="unverifiable")
            summ = " · ".join(filter(None,[f"❌ {nc} contradicted" if nc else "",f"⚠️ {nu} unverifiable" if nu else "",f"✅ {nv2} verified" if nv2 else ""]))
            with st.expander(f"Claim verification — {len(claims)} claims ({summ})", expanded=True):
                for c in sc:
                    txt = str(c.get("claim","—")); st_raw = str(c.get("status","")).lower()
                    icon2 = _SICON.get(st_raw,"•")
                    if   st_raw=="verified":     st.success(f"{icon2} **{st_raw.capitalize()}** — {txt}")
                    elif st_raw=="contradicted": st.error(  f"{icon2} **{st_raw.capitalize()}** — {txt}")
                    else:                        st.warning(f"{icon2} **{st_raw.capitalize()}** — {txt}")

        # Tool log
        tool_log   = ensure_list(det.get(f"{lbl}_agentic_tool_call_log", []))
        url_checks = ensure_list(det.get(f"{lbl}_agentic_url_checks", []))

        if tool_log:
            with st.expander(f"Tool call log — {len(tool_log)} calls", expanded=False):
                for i, call in enumerate(tool_log, 1):
                    tn = call.get("tool","?"); inp = call.get("input",{}); prev = call.get("output_preview","")
                    icon3 = {"search_web":"🔎","fetch_url":"📄","check_url_validity":"🔗"}.get(tn,"🔧")
                    lc1, lc2 = st.columns(2)
                    with lc1:
                        st.markdown(f"**{icon3} {i}. `{tn}`**")
                        if isinstance(inp, dict):
                            u = inp.get("url",""); q = inp.get("query","")
                            if u: st.markdown(f"[{u}]({u})")
                            elif q: st.code(q)
                            else: st.code(json.dumps(inp, ensure_ascii=False)[:200])
                    with lc2:
                        try:
                            p = json.loads(prev) if isinstance(prev, str) else prev
                            if tn=="search_web" and isinstance(p, list):
                                for res in p[:3]:
                                    ur = res.get("url",""); ti = res.get("title",ur); sn = res.get("snippet","")
                                    if ur: st.markdown(f"**[{ti}]({ur})**")
                                    if sn: st.caption(sn[:150])
                            elif tn=="check_url_validity" and isinstance(p, dict):
                                vf = p.get("valid",False); sc2 = p.get("status_code","?")
                                st.markdown(f"{'✅ Valid' if vf else '❌ Invalid'} — HTTP **{sc2}**")
                                if p.get("error"): st.error(p["error"])
                            else:
                                st.code(json.dumps(p, ensure_ascii=False)[:400])
                        except: st.text(str(prev)[:300])
                    if i < len(tool_log):
                        st.markdown('<hr style="border:none;border-top:1px solid #F1F5F9;margin:6px 0">', unsafe_allow_html=True)

        if url_checks:
            with st.expander(f"URL validity — {len(url_checks)} URLs", expanded=False):
                for uc in url_checks:
                    vf = uc.get("valid",False); sc3 = uc.get("status_code","?")
                    url = uc.get("url","?"); final = uc.get("final_url", url); err = uc.get("error")
                    if vf: st.success(f"✅ HTTP {sc3} — [{url}]({url})")
                    else:  st.error(  f"❌ HTTP {sc3 or 'FAILED'}{' — '+err if err else ''} — [{url}]({url})")
                    if final and final != url: st.caption(f"↳ Redirected to: {final}")

        # Token + timing summary
        na_tok = safe_int(det.get(f"{lbl}_nonagentic_total_tokens"))
        ag_tok = safe_int(det.get(f"{lbl}_agentic_total_tokens"))
        na_t   = safe_float(det.get(f"{lbl}_nonagentic_judgment_time_s"))
        ag_t   = safe_float(det.get(f"{lbl}_agentic_judgment_time_s"))
        if any(v is not None for v in [na_tok, ag_tok, na_t, ag_t]):
            with st.expander("Token usage & timing", expanded=False):
                m1,m2,m3,m4,m5 = st.columns(5)
                m1.metric("Non-ag tokens",  f"{na_tok:,}" if na_tok else "—")
                m2.metric("Agentic tokens", f"{ag_tok:,}" if ag_tok else "—")
                m3.metric("Agentic/Non-ag", f"{ag_tok/na_tok:.1f}×" if na_tok and ag_tok else "—")
                m4.metric("Non-ag time",    f"{na_t:.2f}s" if na_t else "—")
                m5.metric("Agentic time",   f"{ag_t:.2f}s" if ag_t else "—")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — EVIDENCE
# ══════════════════════════════════════════════════════════════════════════════
with tab_evidence:
    st.markdown("## Evidence & URL Analysis")

    # Build records
    cov_recs: list[dict] = []
    res_recs: list[dict] = []
    cl_recs:  list[dict] = []

    for fname, rows in raw.items():
        model = file_labels[fname]
        for row in rows:
            lang = row.get("language","?")
            if sel_langs and lang not in sel_langs: continue
            resp_urls = list(dict.fromkeys(extract_urls_from_text(row.get("assistant_response",""))))
            for pol in detect_policy_labels(row):
                if sel_policies and pol not in sel_policies: continue
                uc_list = ensure_list(row.get(f"{pol}_agentic_url_checks",[]))
                uc_map  = {u.get("url",""): u for u in uc_list if u.get("url")}
                for url in resp_urls:
                    chk = uc_map.get(url)
                    cov_recs.append({
                        "url": url, "domain": url_to_domain(url), "model": model,
                        "policy": pol, "language": lang,
                        "is_valid": chk.get("valid") if chk else None,
                        "status_code": chk.get("status_code") if chk else None,
                    })
                tlog = ensure_list(row.get(f"{pol}_agentic_tool_call_log",[]))
                for url in extract_judge_research_urls(tlog):
                    res_recs.append({"url":url,"domain":url_to_domain(url),"model":model,"policy":pol,"language":lang})
                for claim in ensure_list(row.get(f"{pol}_agentic_claim_checks",[])):
                    st2 = str(claim.get("status","")).lower()
                    for src in ensure_list(claim.get("sources",[])):
                        if isinstance(src,str) and src.startswith("http"):
                            cl_recs.append({"url":src,"domain":url_to_domain(src),"status":st2,"model":model,"policy":pol,"language":lang})

    ev1, ev2, ev3 = st.tabs([
        f"Response URLs ({len({r['url'] for r in cov_recs})} unique)",
        f"Judge Research ({len({r['url'] for r in res_recs})} unique)",
        f"Claim Evidence ({len({r['url'] for r in cl_recs})} unique)",
    ])

    with ev1:
        if not cov_recs:
            st.markdown('<div class="info-card">No URLs found in assistant responses with current filters.</div>', unsafe_allow_html=True)
        else:
            n_v  = sum(1 for r in cov_recs if r["is_valid"] is True)
            n_i  = sum(1 for r in cov_recs if r["is_valid"] is False)
            n_u  = sum(1 for r in cov_recs if r["is_valid"] is None)
            chkd = n_v + n_i
            k1,k2,k3,k4 = st.columns(4)
            k1.metric("Unique domains",  len({r["domain"] for r in cov_recs}))
            k2.metric("Valid URLs",       n_v)
            k3.metric("Broken URLs",      n_i)
            k4.metric("Validity rate",   f"{100*n_v/chkd:.0f}%" if chkd else "—")
            if n_u: st.warning(f"⚠️ {n_u} appearances unchecked — re-run evaluation for full coverage")
            dc = Counter(r["domain"] for r in cov_recs)
            dc_v = Counter(r["domain"] for r in cov_recs if r["is_valid"] is True)
            dc_i = Counter(r["domain"] for r in cov_recs if r["is_valid"] is False)
            tn = min(20, len(dc))
            cdf = pd.DataFrame([{"Domain":d,"Valid":dc_v.get(d,0),"Invalid":dc_i.get(d,0)} for d,_ in dc.most_common(tn)])
            ch = alt.Chart(cdf).transform_fold(["Valid","Invalid"],as_=["Status","Count"]).mark_bar().encode(
                x=alt.X("Count:Q"), y=alt.Y("Domain:N",sort="-x",title=None),
                color=alt.Color("Status:N",scale=alt.Scale(domain=["Valid","Invalid"],range=["#10B981","#EF4444"])),
                tooltip=["Domain:N","Status:N","Count:Q"],
            ).properties(height=max(200,min(34*tn,480)))
            st.altair_chart(ch.configure_view(strokeOpacity=0).configure_axis(**_AX), use_container_width=True)

    with ev2:
        if not res_recs:
            st.markdown('<div class="info-card">No judge research URLs — either no tool calls were made or no search results were recorded.</div>', unsafe_allow_html=True)
        else:
            rdc = Counter(r["domain"] for r in res_recs)
            tn2 = min(20, len(rdc))
            k1,k2 = st.columns(2)
            k1.metric("Unique URLs",    len({r["url"] for r in res_recs}))
            k2.metric("Unique domains", len(rdc))
            rdf = pd.DataFrame(rdc.most_common(tn2), columns=["Domain","Count"])
            rc = alt.Chart(rdf).mark_bar(color="#3B82F6").encode(
                x=alt.X("Count:Q"), y=alt.Y("Domain:N",sort="-x",title=None), tooltip=["Domain:N","Count:Q"],
            ).properties(height=max(200,min(34*tn2,480)))
            st.altair_chart(rc.configure_view(strokeOpacity=0).configure_axis(**_AX), use_container_width=True)

    with ev3:
        if not cl_recs:
            st.markdown('<div class="info-card">No claim-check source URLs recorded.</div>', unsafe_allow_html=True)
        else:
            cd  = Counter(r["domain"] for r in cl_recs)
            cd_v = Counter(r["domain"] for r in cl_recs if r["status"]=="verified")
            cd_c = Counter(r["domain"] for r in cl_recs if r["status"]=="contradicted")
            cd_u2 = Counter(r["domain"] for r in cl_recs if r["status"]=="unverifiable")
            tn3 = min(20, len(cd))
            k1,k2,k3 = st.columns(3)
            k1.metric("Unique domains",          len(cd))
            k2.metric("Domains in verified",     len(cd_v))
            k3.metric("Domains in contradicted", len(cd_c))
            cldf = pd.DataFrame([{"Domain":d,"Verified":cd_v.get(d,0),"Contradicted":cd_c.get(d,0),"Unverifiable":cd_u2.get(d,0)} for d,_ in cd.most_common(tn3)])
            clch = alt.Chart(cldf).transform_fold(["Verified","Contradicted","Unverifiable"],as_=["Verdict","Count"]).mark_bar().encode(
                x=alt.X("Count:Q"), y=alt.Y("Domain:N",sort="-x",title=None),
                color=alt.Color("Verdict:N",scale=alt.Scale(domain=["Verified","Contradicted","Unverifiable"],range=["#10B981","#EF4444","#F59E0B"])),
                tooltip=["Domain:N","Verdict:N","Count:Q"],
            ).properties(height=max(200,min(34*tn3,480)))
            st.altair_chart(clch.configure_view(strokeOpacity=0).configure_axis(**_AX), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — DIAGNOSTICS
# ══════════════════════════════════════════════════════════════════════════════
with tab_diag:
    st.markdown("## Token Usage & Timing")
    diag_tok, diag_tim = st.tabs(["📈 Tokens", "⏱️ Timing"])

    # ── Tokens ────────────────────────────────────────────────────────────────
    with diag_tok:
        tok_rows = []
        for fname, rows in raw.items():
            model = file_labels[fname]
            for row in rows:
                lang = row.get("language","?")
                if sel_langs and lang not in sel_langs: continue
                for pol in detect_policy_labels(row):
                    if sel_policies and pol not in sel_policies: continue
                    na_t2 = safe_int(row.get(f"{pol}_nonagentic_total_tokens"))
                    ag_t2 = safe_int(row.get(f"{pol}_agentic_total_tokens"))
                    ag_pk = safe_int(row.get(f"{pol}_agentic_peak_prompt_tokens"))
                    tc2   = safe_int(row.get(f"{pol}_agentic_tool_calls_made")) or 0
                    tok_rows.append({"model":model,"policy":pol,"language":lang,"na_total":na_t2,"ag_total":ag_t2,"ag_peak":ag_pk,"tool_calls":tc2})

        if not tok_rows or not any(r["na_total"] or r["ag_total"] for r in tok_rows):
            st.info("No token data available in loaded files.")
        else:
            df_tok2 = pd.DataFrame(tok_rows)
            tagg = []
            for (model,pol),grp in df_tok2.groupby(["model","policy"]):
                na_ts2 = grp["na_total"].dropna().tolist()
                ag_ts2 = grp["ag_total"].dropna().tolist()
                ag_pks = grp["ag_peak"].dropna().tolist()
                tagg.append({
                    "Model":model,"Policy":pol,
                    "Avg non-ag tokens": int(avg(na_ts2)) if na_ts2 else None,
                    "Avg agentic tokens":int(avg(ag_ts2)) if ag_ts2 else None,
                    "Max peak context":  max(ag_pks)      if ag_pks else None,
                    "Agentic/Non-ag ×":  round(avg(ag_ts2)/avg(na_ts2),1) if na_ts2 and ag_ts2 and avg(na_ts2)>0 else None,
                })
            st.dataframe(pd.DataFrame(tagg), use_container_width=True)

            chrows = []
            for _, r in df_tok2.iterrows():
                if r["na_total"]: chrows.append({"Model":r["model"],"Policy":r["policy"],"Type":"Non-agentic","Tokens":r["na_total"]})
                if r["ag_total"]: chrows.append({"Model":r["model"],"Policy":r["policy"],"Type":"Agentic",    "Tokens":r["ag_total"]})
            if chrows:
                df_ch2 = pd.DataFrame(chrows)
                avg_ch = df_ch2.groupby(["Model","Policy","Type"])["Tokens"].mean().reset_index()
                tc3 = (
                    alt.Chart(avg_ch)
                    .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
                    .encode(
                        x=alt.X("Model:N", title=None, axis=alt.Axis(labelAngle=-25, labelFont="Inter")),
                        y=alt.Y("Tokens:Q", axis=alt.Axis(labelFont="Inter", titleFont="Inter")),
                        color=alt.Color("Type:N",
                            scale=alt.Scale(domain=["Non-agentic","Agentic"], range=["#94A3B8","#3B82F6"]),
                            legend=alt.Legend(title="Mode", orient="bottom"),
                        ),
                        xOffset="Type:N",
                        column=alt.Column("Policy:N", header=alt.Header(labelFont="Inter", titleFont="Inter")),
                        tooltip=["Model:N","Policy:N","Type:N", alt.Tooltip("Tokens:Q", format=",.0f")],
                    )
                    .properties(height=260, width=alt.Step(30))
                )
                st.altair_chart(tc3)

    # ── Timing ────────────────────────────────────────────────────────────────
    with diag_tim:
        tim_rows = []
        for fname, rows in raw.items():
            model = file_labels[fname]
            for row in rows:
                lang = row.get("language","?")
                if sel_langs and lang not in sel_langs: continue
                for pol in detect_policy_labels(row):
                    if sel_policies and pol not in sel_policies: continue
                    nt = safe_float(row.get(f"{pol}_nonagentic_judgment_time_s"))
                    at = safe_float(row.get(f"{pol}_agentic_judgment_time_s"))
                    tc4 = safe_int(row.get(f"{pol}_agentic_tool_calls_made")) or 0
                    if nt or at:
                        tim_rows.append({"model":model,"policy":pol,"language":lang,"na_time":nt,"ag_time":at,"tool_calls":tc4})

        if not tim_rows:
            st.info("No timing data found. Re-run evaluation to generate timing columns.")
        else:
            df_tim = pd.DataFrame(tim_rows)
            tagg2 = []
            for (model,pol),grp in df_tim.groupby(["model","policy"]):
                nts = grp["na_time"].dropna().tolist()
                ats = grp["ag_time"].dropna().tolist()
                tagg2.append({
                    "Model":model,"Policy":pol,
                    "Avg non-ag (s)":  round(avg(nts),2) if nts else None,
                    "Avg agentic (s)": round(avg(ats),2) if ats else None,
                    "Max agentic (s)": round(max(ats),2) if ats else None,
                    "Agentic/Non-ag ×":round(avg(ats)/avg(nts),1) if nts and ats and avg(nts)>0 else None,
                })
            st.dataframe(pd.DataFrame(tagg2), use_container_width=True)

            brows = []
            for _,r in df_tim.iterrows():
                if r["na_time"]: brows.append({"Model":r["model"],"Policy":r["policy"],"Mode":"Non-agentic","Time (s)":r["na_time"]})
                if r["ag_time"]: brows.append({"Model":r["model"],"Policy":r["policy"],"Mode":"Agentic",    "Time (s)":r["ag_time"]})
            if brows:
                df_bx = pd.DataFrame(brows)
                bx = (
                    alt.Chart(df_bx)
                    .mark_boxplot(extent="min-max", size=32)
                    .encode(
                        x=alt.X("Model:N", title=None, axis=alt.Axis(labelAngle=-25, labelFont="Inter")),
                        y=alt.Y("Time (s):Q", axis=alt.Axis(labelFont="Inter", titleFont="Inter")),
                        color=alt.Color("Mode:N",
                            scale=alt.Scale(domain=["Non-agentic","Agentic"], range=["#94A3B8","#3B82F6"]),
                            legend=alt.Legend(title="Mode", orient="bottom"),
                        ),
                        column=alt.Column("Policy:N", header=alt.Header(labelFont="Inter", titleFont="Inter")),
                    )
                    .properties(height=280, width=alt.Step(40))
                )
                st.altair_chart(bx)

            # Scatter: agentic time vs tool calls
            sc_rows = []
            for _,r in df_tim.iterrows():
                if r["ag_time"] is not None and r["tool_calls"] is not None:
                    sc_rows.append({"Model":r["model"],"Tool calls":r["tool_calls"],"Agentic time (s)":r["ag_time"],"Policy":r["policy"]})
            if sc_rows:
                df_sc2 = pd.DataFrame(sc_rows)
                sc2 = alt.Chart(df_sc2).mark_circle(size=70, opacity=0.75).encode(
                    x=alt.X("Tool calls:Q"),
                    y=alt.Y("Agentic time (s):Q"),
                    color=alt.Color("Model:N",
                        scale=alt.Scale(domain=model_list, range=[model_colors[m] for m in model_list]),
                        legend=alt.Legend(title="Judge model"),
                    ),
                    tooltip=["Model:N","Policy:N","Tool calls:Q",alt.Tooltip("Agentic time (s):Q",format=".2f")],
                )
                st.markdown("**Agentic time vs tool calls made**")
                st.altair_chart(_bare(sc2, 230), use_container_width=True)
