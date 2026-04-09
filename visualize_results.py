"""
visualize_results.py
--------------------
Interactive Streamlit dashboard for exploring guardrail evaluation results.

Usage:
    streamlit run visualize_results.py

Then open http://localhost:8501 in your browser.
Use the sidebar to load any JSON output file from the outputs/ folder.

Install Streamlit if needed:
    pip install streamlit
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import streamlit as st

# ── page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Guardrail Evaluation Explorer",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── helpers ───────────────────────────────────────────────────────────────────

def load_json(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def detect_policy_labels(row: dict) -> list[str]:
    """
    Find all policy labels present in a result row by looking for columns
    that end in '_nonagentic_score'. Returns e.g. ['policy', 'policy_fa'].
    """
    labels = []
    for key in row:
        if key.endswith("_nonagentic_score"):
            labels.append(key[: -len("_nonagentic_score")])
    return sorted(labels)


def score_color(score) -> str:
    """Return a hex color for a 0–1 compliance score."""
    if score is None:
        return "#888888"
    try:
        s = float(score)
    except (TypeError, ValueError):
        return "#888888"
    if s >= 0.85:
        return "#2ecc71"   # green
    if s >= 0.60:
        return "#f39c12"   # amber
    return "#e74c3c"       # red


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


def fmt_valid(v) -> str:
    if v is True:
        return "✅ PASS"
    if v is False:
        return "❌ FAIL"
    return "—"


def ensure_list(val) -> list:
    """JSON values stored in the file may be a list or a JSON-encoded string."""
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


# ── sidebar: file loader ──────────────────────────────────────────────────────

st.sidebar.title("🛡️ Guardrail Explorer")
st.sidebar.markdown("---")

outputs_dir = Path("outputs")
json_files = sorted(outputs_dir.glob("*.json")) if outputs_dir.exists() else []
json_file_names = [f.name for f in json_files]

if not json_file_names:
    st.error("No JSON files found in outputs/. Run a guardrail evaluation first.")
    st.stop()

selected_file = st.sidebar.selectbox(
    "Output file",
    json_file_names,
    index=len(json_file_names) - 1,  # default to most recent
)

data = load_json(str(outputs_dir / selected_file))

# Filter out error-only rows (rows where evaluation failed entirely)
valid_rows = [r for r in data if any(k.endswith("_nonagentic_score") for k in r)]
error_rows = [r for r in data if r not in valid_rows]

if not valid_rows:
    st.error(f"No successfully evaluated rows found in {selected_file}.")
    if error_rows:
        st.json(error_rows[0])
    st.stop()

policy_labels = detect_policy_labels(valid_rows[0])

st.sidebar.markdown(f"**{len(valid_rows)}** scenarios loaded")
st.sidebar.markdown(f"**Policies:** {', '.join(policy_labels)}")
if error_rows:
    st.sidebar.warning(f"{len(error_rows)} rows had errors (excluded from analysis)")

# ── tabs ──────────────────────────────────────────────────────────────────────

tab_overview, tab_detail, tab_policy_compare, tab_tokens = st.tabs([
    "📊 Overview",
    "🔍 Scenario Detail",
    "🌐 EN vs FA Policy",
    "📈 Token Usage",
])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════

with tab_overview:
    st.header("Evaluation Overview")

    # Run metadata from first valid row
    meta = valid_rows[0]
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    col_m1.metric("Provider", meta.get("provider", "—"))
    col_m2.metric("Assistant model", meta.get("model", "—"))
    col_m3.metric("Guardrail model", meta.get("guardrail_model", "—"))
    col_m4.metric("Max tool calls", meta.get("max_tool_calls_allowed", "—"))

    st.markdown("---")

    # Summary table — one row per scenario × policy
    st.subheader("Results table")
    st.markdown(
        "Color coding: 🟢 score ≥ 0.85 · 🟡 0.60–0.84 · 🔴 < 0.60  |  "
        "**Judgment changed** = agentic flipped the pass/fail verdict"
    )

    rows_for_table = []
    for row in valid_rows:
        for label in policy_labels:
            na_score = row.get(f"{label}_nonagentic_score")
            ag_score = row.get(f"{label}_agentic_score")
            delta    = row.get(f"{label}_score_delta")
            changed  = row.get(f"{label}_judgment_changed")
            used_tools = row.get(f"{label}_agentic_used_tools", False)
            tool_calls = row.get(f"{label}_agentic_tool_calls_made", 0)

            rows_for_table.append({
                "ID": row.get("id", ""),
                "Lang": row.get("language", ""),
                "Scenario (truncated)": str(row.get("scenario", ""))[:80] + "…",
                "Policy": label,
                "Non-agentic valid": fmt_valid(row.get(f"{label}_nonagentic_valid")),
                "Non-agentic score": round(float(na_score), 3) if na_score is not None else None,
                "Agentic valid": fmt_valid(row.get(f"{label}_agentic_valid")),
                "Agentic score": round(float(ag_score), 3) if ag_score is not None else None,
                "Score delta": delta_arrow(delta),
                "Judgment changed": "⚡ YES" if changed else ("no" if changed is False else "—"),
                "Tools used": f"{tool_calls} calls" if used_tools else "none",
            })

    import pandas as pd
    df = pd.DataFrame(rows_for_table)

    # Highlight rows where judgment changed
    def highlight_changed(row):
        if row["Judgment changed"] == "⚡ YES":
            return ["background-color: #fff3cd"] * len(row)
        return [""] * len(row)

    st.dataframe(
        df.style.apply(highlight_changed, axis=1),
        use_container_width=True,
        height=400,
    )

    st.markdown("---")

    # Summary stats per policy
    st.subheader("Score summary by policy")
    stat_cols = st.columns(len(policy_labels))
    for col, label in zip(stat_cols, policy_labels):
        na_scores = [
            float(r[f"{label}_nonagentic_score"])
            for r in valid_rows
            if r.get(f"{label}_nonagentic_score") is not None
        ]
        ag_scores = [
            float(r[f"{label}_agentic_score"])
            for r in valid_rows
            if r.get(f"{label}_agentic_score") is not None
        ]
        deltas = [
            float(r[f"{label}_score_delta"])
            for r in valid_rows
            if r.get(f"{label}_score_delta") is not None
        ]
        changed_count = sum(
            1 for r in valid_rows if r.get(f"{label}_judgment_changed") is True
        )

        col.markdown(f"### `{label}`")
        col.metric(
            "Avg non-agentic score",
            f"{sum(na_scores)/len(na_scores):.3f}" if na_scores else "—",
        )
        col.metric(
            "Avg agentic score",
            f"{sum(ag_scores)/len(ag_scores):.3f}" if ag_scores else "—",
        )
        col.metric(
            "Avg score delta",
            f"{sum(deltas)/len(deltas):+.3f}" if deltas else "—",
        )
        col.metric("Judgments changed", f"{changed_count} / {len(valid_rows)}")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — SCENARIO DETAIL
# ═══════════════════════════════════════════════════════════════════════════════

with tab_detail:
    st.header("Scenario Detail")

    # Scenario selector
    scenario_options = {
        f"[{r.get('id',i)}] ({r.get('language','?')}) {str(r.get('scenario',''))[:70]}…": i
        for i, r in enumerate(valid_rows)
    }
    selected_label = st.selectbox("Select scenario", list(scenario_options.keys()))
    row = valid_rows[scenario_options[selected_label]]

    # Policy selector (if more than one)
    selected_policy = st.radio(
        "Policy to examine",
        policy_labels,
        horizontal=True,
    )
    label = selected_policy

    st.markdown("---")

    # ── Scenario + Response ──────────────────────────────────────────────────
    st.subheader("📝 Scenario & Assistant Response")
    col_s, col_r = st.columns(2)

    with col_s:
        st.markdown("**Scenario** (sent to assistant LLM)")
        st.info(row.get("scenario", "—"))
        st.caption(f"Language: {row.get('language','?')}  |  ID: {row.get('id','?')}")

    with col_r:
        st.markdown("**Assistant Response**")
        st.success(row.get("assistant_response", "—"))
        st.caption(f"Model: {row.get('model','?')}  |  Provider: {row.get('provider','?')}")

    st.markdown("---")

    # ── Side-by-side guardrail comparison ────────────────────────────────────
    st.subheader(f"⚖️ Guardrail Comparison — policy: `{label}`")

    na_score   = row.get(f"{label}_nonagentic_score")
    ag_score   = row.get(f"{label}_agentic_score")
    delta      = row.get(f"{label}_score_delta")
    changed    = row.get(f"{label}_judgment_changed")
    tool_count = row.get(f"{label}_agentic_tool_calls_made", 0)

    if changed:
        st.warning("⚡ **Judgment changed** — the agentic guardrail reached a different pass/fail verdict after retrieval.")

    col_na, col_mid, col_ag = st.columns([5, 1, 5])

    # Non-agentic column
    with col_na:
        st.markdown("### 🔒 Non-Agentic")
        st.caption("Single LLM call · no tools · built-in knowledge only")
        na_color = score_color(na_score)
        na_score_disp = f"{float(na_score):.3f}" if na_score is not None else "—"
        st.markdown(
            f"<div style='font-size:2.5rem; font-weight:bold; color:{na_color}'>"
            f"{na_score_disp}</div>",
            unsafe_allow_html=True,
        )
        st.markdown(fmt_valid(row.get(f"{label}_nonagentic_valid")))
        st.markdown("**Explanation:**")
        st.markdown(
            f"> {row.get(f'{label}_nonagentic_explanation', '—')}"
        )

    # Middle divider
    with col_mid:
        st.markdown("<br><br><br><br><div style='text-align:center; font-size:1.5rem'>↔</div>", unsafe_allow_html=True)

    # Agentic column
    with col_ag:
        st.markdown("### 🌐 Agentic")
        st.caption(f"Multi-turn loop · {tool_count} tool call(s) · web retrieval + URL checks")
        ag_color = score_color(ag_score)
        ag_score_disp = f"{float(ag_score):.3f}" if ag_score is not None else "—"
        st.markdown(
            f"<div style='font-size:2.5rem; font-weight:bold; color:{ag_color}'>"
            f"{ag_score_disp}</div>",
            unsafe_allow_html=True,
        )
        st.markdown(fmt_valid(row.get(f"{label}_agentic_valid")))

        # Score delta badge
        delta_str = delta_arrow(delta)
        delta_color = "#2ecc71" if (delta or 0) > 0.01 else ("#e74c3c" if (delta or 0) < -0.01 else "#888")
        st.markdown(
            f"<span style='font-size:1.1rem; color:{delta_color}'>{delta_str} vs non-agentic</span>",
            unsafe_allow_html=True,
        )
        st.markdown("**Explanation:**")
        st.markdown(f"> {row.get(f'{label}_agentic_explanation', '—')}")

    st.markdown("---")

    # ── Token usage ──────────────────────────────────────────────────────────
    st.subheader("🔢 Token Usage")
    na_prompt   = row.get(f"{label}_nonagentic_prompt_tokens")
    na_complete = row.get(f"{label}_nonagentic_completion_tokens")
    na_total    = row.get(f"{label}_nonagentic_total_tokens")
    ag_prompt   = row.get(f"{label}_agentic_prompt_tokens_total")
    ag_complete = row.get(f"{label}_agentic_completion_tokens_total")
    ag_total    = row.get(f"{label}_agentic_total_tokens")
    ag_peak     = row.get(f"{label}_agentic_peak_prompt_tokens")
    ag_per_turn = ensure_list(row.get(f"{label}_agentic_token_usage_per_turn", []))

    tok_col1, tok_col2 = st.columns(2)

    with tok_col1:
        st.markdown("**🔒 Non-Agentic** *(tiktoken — same tokenizer the model uses)*")
        t1, t2, t3 = st.columns(3)
        t1.metric("Prompt tokens", f"{int(na_prompt):,}" if na_prompt is not None else "—",
                  help="Tokens in eval_input_text: role instruction + policy + rubric + system prompt + scenario + assistant response")
        t2.metric("Completion tokens", f"{int(na_complete):,}" if na_complete is not None else "—",
                  help="Tokens in the returned explanation/justification")
        t3.metric("Total tokens", f"{int(na_total):,}" if na_total is not None else "—")

    with tok_col2:
        st.markdown("**🌐 Agentic** *(exact — from provider API, all turns summed)*")
        t4, t5, t6, t7 = st.columns(4)
        t4.metric("Prompt total", f"{int(ag_prompt):,}" if ag_prompt is not None else "—",
                  help="Sum of prompt tokens across all LLM turns (grows each turn as tool results are added)")
        t5.metric("Completion total", f"{int(ag_complete):,}" if ag_complete is not None else "—",
                  help="Sum of completion tokens: reasoning + tool call requests + final verdict")
        t6.metric("Total tokens", f"{int(ag_total):,}" if ag_total is not None else "—")
        t7.metric("Peak context", f"{int(ag_peak):,}" if ag_peak is not None else "—",
                  help="Largest single-turn prompt — shows peak context window pressure")
        if ag_total is not None and na_total:
            multiplier = int(ag_total) / int(na_total)
            st.caption(f"Agentic used **{multiplier:.1f}×** more tokens than non-agentic")

    if ag_per_turn:
        import pandas as pd
        with st.expander("📊 Per-turn token breakdown (context window growth)", expanded=False):
            st.markdown(
                "Each row is one LLM call. Prompt tokens grow as tool results are added to "
                "the conversation history — the peak shows how close you get to the model's "
                "context window limit."
            )
            df_turns = pd.DataFrame(ag_per_turn)
            df_turns["has_tool_calls"] = df_turns["has_tool_calls"].map(
                {True: "yes", False: "no", None: "—"}
            )
            st.dataframe(df_turns, use_container_width=True)

            # Simple bar chart of prompt tokens per turn
            try:
                import altair as alt
                chart_data = pd.DataFrame({
                    "Turn": [str(t["turn"]) for t in ag_per_turn],
                    "Prompt tokens": [t["prompt_tokens"] for t in ag_per_turn],
                    "Completion tokens": [t["completion_tokens"] for t in ag_per_turn],
                })
                chart = alt.Chart(chart_data).transform_fold(
                    ["Prompt tokens", "Completion tokens"],
                    as_=["Type", "Tokens"],
                ).mark_bar().encode(
                    x=alt.X("Turn:O", title="Turn"),
                    y=alt.Y("Tokens:Q", title="Tokens"),
                    color=alt.Color("Type:N", scale=alt.Scale(
                        domain=["Prompt tokens", "Completion tokens"],
                        range=["#4e79a7", "#f28e2b"],
                    )),
                    tooltip=["Turn", "Type", "Tokens"],
                ).properties(height=250, title="Tokens per turn (prompt = context window consumed)")
                st.altair_chart(chart, use_container_width=True)
            except ImportError:
                st.info("Install `altair` for turn-by-turn charts: `pip install altair`")

    st.markdown("---")

    # ── Tool calls ───────────────────────────────────────────────────────────
    sources = ensure_list(row.get(f"{label}_agentic_sources_used", []))
    tool_log = ensure_list(row.get(f"{label}_agentic_tool_call_log", []))
    url_checks = ensure_list(row.get(f"{label}_agentic_url_checks", []))

    if tool_log:
        with st.expander(f"🔧 Tool Call Log ({len(tool_log)} calls)", expanded=True):
            for i, call in enumerate(tool_log, 1):
                tool_name = call.get("tool", "?")
                inp = call.get("input", {})
                preview = call.get("output_preview", "")

                if tool_name == "search_web":
                    icon = "🔎"
                elif tool_name == "fetch_url":
                    icon = "📄"
                elif tool_name == "check_url_validity":
                    icon = "🔗"
                else:
                    icon = "🔧"

                st.markdown(f"**{icon} Call {i}: `{tool_name}`**")

                tc1, tc2 = st.columns(2)

                # ── Input column ──────────────────────────────────────────
                with tc1:
                    st.markdown("*Input:*")
                    # Make the input URL clickable when it's a URL-based tool
                    if tool_name in ("fetch_url", "check_url_validity"):
                        input_url = inp.get("url", "")
                        st.markdown(f"**URL:** [{input_url}]({input_url})")
                    else:
                        st.code(json.dumps(inp, indent=2, ensure_ascii=False), language="json")

                # ── Output column ─────────────────────────────────────────
                with tc2:
                    st.markdown("*Output:*")
                    try:
                        parsed = json.loads(preview) if isinstance(preview, str) else preview

                        # search_web → list of {title, url, snippet}
                        if tool_name == "search_web" and isinstance(parsed, list):
                            for res in parsed:
                                url_r = res.get("url", "")
                                title_r = res.get("title", url_r)
                                snippet_r = res.get("snippet", "")
                                st.markdown(f"**[{title_r}]({url_r})**")
                                if snippet_r:
                                    st.caption(snippet_r[:200])
                                st.markdown("")   # spacing

                        # fetch_url → {url, content}
                        elif tool_name == "fetch_url" and isinstance(parsed, dict) and "content" in parsed:
                            fetched_url = parsed.get("url", inp.get("url", ""))
                            st.markdown(f"**Source:** [{fetched_url}]({fetched_url})")
                            st.caption(
                                f"Fetched {len(parsed['content'])} characters of text"
                            )
                            st.text(parsed["content"][:400])

                        # check_url_validity → {url, valid, status_code, final_url, ...}
                        elif tool_name == "check_url_validity" and isinstance(parsed, dict):
                            checked_url = parsed.get("url", "")
                            final_url = parsed.get("final_url", checked_url)
                            valid_flag = parsed.get("valid", False)
                            status = parsed.get("status_code", "?")
                            redirects = parsed.get("redirect_count", 0)
                            error = parsed.get("error")
                            badge = "✅ Valid" if valid_flag else "❌ Invalid"
                            st.markdown(f"{badge} — HTTP **{status}**")
                            st.markdown(f"URL: [{checked_url}]({checked_url})")
                            if final_url and final_url != checked_url:
                                st.markdown(
                                    f"Redirected to: [{final_url}]({final_url})"
                                    f" ({redirects} redirect{'s' if redirects != 1 else ''})"
                                )
                            if error:
                                st.error(f"Error: {error}")

                        # fallback — raw JSON
                        else:
                            st.code(
                                json.dumps(parsed, indent=2, ensure_ascii=False)[:600],
                                language="json",
                            )

                    except (json.JSONDecodeError, ValueError, TypeError):
                        st.text(str(preview)[:400])

                st.markdown("---")
    else:
        st.info(f"No tool calls were made for policy `{label}` (agentic_used_tools = {row.get(f'{label}_agentic_used_tools', False)}).")

    # ── URL checks ───────────────────────────────────────────────────────────
    if url_checks:
        with st.expander(f"🔗 URL Validity Checks ({len(url_checks)} URLs)", expanded=True):
            for uc in url_checks:
                valid_flag = uc.get("valid", False)
                status = uc.get("status_code", "?")
                url = uc.get("url", "?")
                final = uc.get("final_url", url)
                redirects = uc.get("redirect_count", 0)
                error = uc.get("error")

                url_link = f"[{url}]({url})"
                final_link = f"[{final}]({final})" if final and final != url else ""
                redirect_note = f" → {final_link} ({redirects} redirect)" if redirects and final_link else ""

                if valid_flag:
                    st.success(f"✅ **HTTP {status}** — {url_link}{redirect_note}",
                               icon=None)
                    # st.success doesn't render markdown links, so add a separate line
                    st.markdown(f"&nbsp;&nbsp;&nbsp;🔗 {url_link}{redirect_note}")
                else:
                    st.error(f"❌ **HTTP {status or 'FAILED'}**" + (f" — Error: {error}" if error else ""))
                    st.markdown(f"&nbsp;&nbsp;&nbsp;🔗 {url_link}")
    else:
        st.caption("No URLs were found in the assistant response — URL checker had nothing to check.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — EN vs FA POLICY COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════

with tab_policy_compare:
    st.header("English vs Farsi Policy Comparison")
    st.markdown(
        "Same response, same guardrail, same criteria — but does the language "
        "the policy is written in change the score? This tab compares `policy` "
        "(English) vs `policy_fa` (Farsi) for every scenario."
    )

    en_label = next((l for l in policy_labels if not l.endswith("_fa")), None)
    fa_label = next((l for l in policy_labels if l.endswith("_fa")), None)

    if not en_label or not fa_label:
        st.warning(
            "Could not find both an English policy column and a Farsi policy column "
            f"in this file. Found: {policy_labels}"
        )
    else:
        # Build comparison table
        compare_rows = []
        for row in valid_rows:
            en_na = row.get(f"{en_label}_nonagentic_score")
            fa_na = row.get(f"{fa_label}_nonagentic_score")
            en_ag = row.get(f"{en_label}_agentic_score")
            fa_ag = row.get(f"{fa_label}_agentic_score")

            def safe_diff(a, b):
                if a is not None and b is not None:
                    return round(float(a) - float(b), 3)
                return None

            compare_rows.append({
                "ID": row.get("id", ""),
                "Lang": row.get("language", ""),
                "Scenario": str(row.get("scenario", ""))[:60] + "…",
                "EN non-agentic": round(float(en_na), 3) if en_na is not None else None,
                "FA non-agentic": round(float(fa_na), 3) if fa_na is not None else None,
                "EN−FA (non-ag)": safe_diff(en_na, fa_na),
                "EN agentic": round(float(en_ag), 3) if en_ag is not None else None,
                "FA agentic": round(float(fa_ag), 3) if fa_ag is not None else None,
                "EN−FA (agentic)": safe_diff(en_ag, fa_ag),
            })

        import pandas as pd
        df_compare = pd.DataFrame(compare_rows)

        def color_diff(val):
            if val is None or val == "":
                return ""
            try:
                v = float(val)
            except (TypeError, ValueError):
                return ""
            if abs(v) < 0.05:
                return "color: green"
            if abs(v) < 0.15:
                return "color: orange"
            return "color: red; font-weight: bold"

        st.dataframe(
            df_compare.style.applymap(color_diff, subset=["EN−FA (non-ag)", "EN−FA (agentic)"]),
            use_container_width=True,
        )

        st.markdown("---")
        st.subheader("Deep dive: select a scenario")

        scenario_opts_cmp = {
            f"[{r.get('id',i)}] ({r.get('language','?')}) {str(r.get('scenario',''))[:60]}…": i
            for i, r in enumerate(valid_rows)
        }
        sel_cmp = st.selectbox("Scenario", list(scenario_opts_cmp.keys()), key="cmp_sel")
        row_cmp = valid_rows[scenario_opts_cmp[sel_cmp]]

        col_en, col_fa = st.columns(2)

        for col, lbl, flag in [(col_en, en_label, "🇬🇧 English policy"), (col_fa, fa_label, "🇮🇷 Farsi policy")]:
            with col:
                st.markdown(f"### {flag} (`{lbl}`)")

                # Non-agentic
                na_s = row_cmp.get(f"{lbl}_nonagentic_score")
                na_v = row_cmp.get(f"{lbl}_nonagentic_valid")
                ag_s = row_cmp.get(f"{lbl}_agentic_score")
                ag_v = row_cmp.get(f"{lbl}_agentic_valid")

                st.markdown("**Non-agentic**")
                c1, c2 = st.columns(2)
                c1.metric("Score", f"{float(na_s):.3f}" if na_s is not None else "—")
                c2.markdown(fmt_valid(na_v))
                with st.expander("Explanation"):
                    st.write(row_cmp.get(f"{lbl}_nonagentic_explanation", "—"))

                st.markdown("**Agentic**")
                c3, c4 = st.columns(2)
                c3.metric("Score", f"{float(ag_s):.3f}" if ag_s is not None else "—",
                          delta=f"{float(row_cmp.get(f'{lbl}_score_delta', 0) or 0):+.3f}")
                c4.markdown(fmt_valid(ag_v))
                with st.expander("Explanation"):
                    st.write(row_cmp.get(f"{lbl}_agentic_explanation", "—"))


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — TOKEN USAGE
# ═══════════════════════════════════════════════════════════════════════════════

with tab_tokens:
    import pandas as pd

    st.header("Token Usage — Non-Agentic vs Agentic")
    st.markdown(
        "Compares token consumption between both guardrail paths.\n\n"
        "**Non-agentic:** counted with `tiktoken` (same tokenizer the model uses). "
        "Prompt = the full `eval_input_text` we built (role instruction + policy + rubric + "
        "system prompt + scenario + assistant response). "
        "Completion = the returned explanation text. "
        "These are the same numbers the provider would report in `resp.usage` — we just "
        "count from our side since the `any-guardrail` library doesn't expose that object.\n\n"
        "**Agentic:** exact figures from `resp.usage` returned by the provider API, summed "
        "across all LLM turns in the tool-calling loop (initial call + one call per tool batch). "
        "**Peak context** is the largest single-turn prompt token count — how close the model "
        "got to its context window limit as tool results accumulated."
    )

    # ── Helper to safely parse a token count ─────────────────────────────────
    def safe_int(val):
        try:
            return int(val) if val is not None else None
        except (TypeError, ValueError):
            return None

    # ── Build per-scenario token table ────────────────────────────────────────
    token_rows = []
    for row in valid_rows:
        for label in policy_labels:
            na_prompt  = safe_int(row.get(f"{label}_nonagentic_prompt_tokens"))
            na_comp    = safe_int(row.get(f"{label}_nonagentic_completion_tokens"))
            na_total   = safe_int(row.get(f"{label}_nonagentic_total_tokens"))
            ag_prompt  = safe_int(row.get(f"{label}_agentic_prompt_tokens_total"))
            ag_comp    = safe_int(row.get(f"{label}_agentic_completion_tokens_total"))
            ag_total   = safe_int(row.get(f"{label}_agentic_total_tokens"))
            ag_peak    = safe_int(row.get(f"{label}_agentic_peak_prompt_tokens"))
            tool_calls = safe_int(row.get(f"{label}_agentic_tool_calls_made")) or 0

            multiplier = None
            if ag_total is not None and na_total and na_total > 0:
                multiplier = round(ag_total / na_total, 1)

            token_rows.append({
                "ID": row.get("id", ""),
                "Lang": row.get("language", ""),
                "Policy": label,
                "Tool calls": tool_calls,
                "Non-ag prompt (tiktoken)": na_prompt,
                "Non-ag completion (tiktoken)": na_comp,
                "Non-ag total (tiktoken)": na_total,
                "Ag prompt total (API)": ag_prompt,
                "Ag completion total (API)": ag_comp,
                "Ag total (API)": ag_total,
                "Ag peak context": ag_peak,
                "Ag/Non-ag multiplier": multiplier,
            })

    df_tok = pd.DataFrame(token_rows)
    st.dataframe(df_tok, use_container_width=True, height=350)

    st.markdown("---")

    # ── Summary stats ─────────────────────────────────────────────────────────
    st.subheader("Summary statistics")
    stat_cols_t = st.columns(len(policy_labels))
    for col, label in zip(stat_cols_t, policy_labels):
        col.markdown(f"### `{label}`")

        na_totals = [safe_int(r.get(f"{label}_nonagentic_total_tokens"))
                     for r in valid_rows
                     if r.get(f"{label}_nonagentic_total_tokens") is not None]
        ag_totals = [safe_int(r.get(f"{label}_agentic_total_tokens"))
                     for r in valid_rows
                     if r.get(f"{label}_agentic_total_tokens") is not None]
        ag_peaks  = [safe_int(r.get(f"{label}_agentic_peak_prompt_tokens"))
                     for r in valid_rows
                     if r.get(f"{label}_agentic_peak_prompt_tokens") is not None]
        ag_prompts = [safe_int(r.get(f"{label}_agentic_prompt_tokens_total"))
                      for r in valid_rows
                      if r.get(f"{label}_agentic_prompt_tokens_total") is not None]
        ag_comps   = [safe_int(r.get(f"{label}_agentic_completion_tokens_total"))
                      for r in valid_rows
                      if r.get(f"{label}_agentic_completion_tokens_total") is not None]

        def avg(lst):
            return f"{sum(lst) // len(lst):,}" if lst else "—"
        def mx(lst):
            return f"{max(lst):,}" if lst else "—"

        col.metric("Avg non-ag tokens (tiktoken)", avg(na_totals))
        col.metric("Avg agentic tokens (exact)", avg(ag_totals))
        col.metric("Avg agentic prompt tokens", avg(ag_prompts))
        col.metric("Avg agentic completion tokens", avg(ag_comps))
        col.metric("Max peak context (any scenario)", mx(ag_peaks),
                   help="Highest single-turn prompt token count across all scenarios — how close to context limit")

    st.markdown("---")

    # ── Bar chart: non-agentic vs agentic total tokens ────────────────────────
    st.subheader("Non-agentic vs agentic total tokens per scenario")
    st.caption("Non-agentic = estimated · Agentic = exact from API · pick a policy to chart")

    chart_policy = st.radio("Policy", policy_labels, horizontal=True, key="tok_chart_policy")

    chart_rows = []
    for row in valid_rows:
        sid = f"[{row.get('id','')}] {row.get('language','')}"
        na_t = safe_int(row.get(f"{chart_policy}_nonagentic_total_tokens"))
        ag_t = safe_int(row.get(f"{chart_policy}_agentic_total_tokens"))
        if na_t is not None:
            chart_rows.append({"Scenario": sid, "Type": "Non-agentic (tiktoken)", "Tokens": na_t})
        if ag_t is not None:
            chart_rows.append({"Scenario": sid, "Type": "Agentic (API exact)", "Tokens": ag_t})

    if chart_rows:
        try:
            import altair as alt
            df_chart = pd.DataFrame(chart_rows)
            bar = alt.Chart(df_chart).mark_bar().encode(
                x=alt.X("Scenario:O", sort=None, title="Scenario"),
                y=alt.Y("Tokens:Q", title="Total tokens"),
                color=alt.Color("Type:N", scale=alt.Scale(
                    domain=["Non-agentic (tiktoken)", "Agentic (API exact)"],
                    range=["#76b7b2", "#e15759"],
                )),
                xOffset="Type:N",
                tooltip=["Scenario", "Type", "Tokens"],
            ).properties(height=350)
            st.altair_chart(bar, use_container_width=True)
        except ImportError:
            st.dataframe(pd.DataFrame(chart_rows), use_container_width=True)
            st.info("Install `altair` for bar charts: `pip install altair`")

    st.markdown("---")

    # ── Context window growth: agentic prompt tokens per turn ─────────────────
    st.subheader("Context window growth — agentic prompt tokens per turn")
    st.markdown(
        "Shows how the context window fills up as tool results are appended to the "
        "conversation history. Each step corresponds to one LLM call in the agentic loop."
    )

    growth_scenario_opts = {
        f"[{r.get('id',i)}] ({r.get('language','?')}) {str(r.get('scenario',''))[:60]}…": i
        for i, r in enumerate(valid_rows)
    }
    sel_growth = st.selectbox("Scenario", list(growth_scenario_opts.keys()), key="growth_sel")
    row_growth = valid_rows[growth_scenario_opts[sel_growth]]

    growth_policy = st.radio("Policy", policy_labels, horizontal=True, key="growth_policy")

    per_turn = ensure_list(row_growth.get(f"{growth_policy}_agentic_token_usage_per_turn", []))

    if per_turn:
        try:
            import altair as alt
            df_growth = pd.DataFrame(per_turn)
            df_growth["turn_label"] = df_growth["turn"].astype(str)
            growth_chart = alt.Chart(df_growth).transform_fold(
                ["prompt_tokens", "completion_tokens"],
                as_=["Type", "Tokens"],
            ).mark_bar().encode(
                x=alt.X("turn_label:O", title="Turn (LLM call #)"),
                y=alt.Y("Tokens:Q", title="Tokens"),
                color=alt.Color("Type:N", scale=alt.Scale(
                    domain=["prompt_tokens", "completion_tokens"],
                    range=["#4e79a7", "#f28e2b"],
                )),
                tooltip=["turn_label", "Type", "Tokens",
                         alt.Tooltip("has_tool_calls:N", title="Tool calls?")],
            ).properties(
                height=280,
                title="Context window growth (prompt = all prior messages + tool results)",
            )
            st.altair_chart(growth_chart, use_container_width=True)
        except ImportError:
            st.dataframe(pd.DataFrame(per_turn), use_container_width=True)

        # Raw table
        with st.expander("Raw per-turn data"):
            st.dataframe(pd.DataFrame(per_turn), use_container_width=True)
    else:
        st.info(
            "No per-turn token data for this scenario/policy. "
            "This means either no tool calls were made, or the provider did not return usage metadata."
        )
