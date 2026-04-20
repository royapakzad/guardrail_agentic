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
from typing import Any, Optional

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


# Canonical pass/fail threshold — must match VALID_SCORE_THRESHOLD / NONAGENTIC_VALID_THRESHOLD
# in agentic_runner.py and guardrails_runner.py (both set to 0.6, strictly greater than).
_VALID_THRESHOLD = 0.6


def score_to_valid(score) -> Optional[bool]:
    """
    Derive the pass/fail boolean from a numeric score using the canonical threshold.
    Returns None when score is missing or unparseable.

    This is the single source of truth for the visualization. We re-derive from
    score rather than trusting the stored 'valid' field because older output files
    may contain incorrect stored values (pipeline bug where the model's self-reported
    valid was trusted even when the score was below 0.6).
    """
    if score is None:
        return None
    try:
        return float(score) > _VALID_THRESHOLD
    except (TypeError, ValueError):
        return None


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


def extract_urls_from_text(text: str) -> list[str]:
    """
    Extract URLs from raw text using regex.
    Strips trailing punctuation that is part of surrounding prose, not the URL.
    """
    urls: list[str] = []
    for m in re.finditer(r'https?://\S+', text or "", re.IGNORECASE):
        url = m.group(0).rstrip('.,;:!?)]\'"<>')
        if url:
            urls.append(url)
    return urls


def extract_judge_research_urls(tool_log: list) -> list[str]:
    """
    URLs the judge consulted for its own research (Phase 1 — claim verification).
    Sources: fetch_url inputs + search_web result links.
    Does NOT include check_url_validity (those come from the assistant response).
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
            try:
                results = json.loads(preview) if isinstance(preview, str) else preview
                if isinstance(results, dict) and "results" in results:
                    results = results["results"]
                if isinstance(results, list):
                    for r in results:
                        url = r.get("url", "") if isinstance(r, dict) else ""
                        if url:
                            urls.append(url)
            except (json.JSONDecodeError, TypeError):
                pass
    return urls


def extract_url_validity_checks(tool_log: list) -> list[str]:
    """
    URLs that the judge checked via check_url_validity (Phase 2).
    These are URLs extracted FROM the assistant's response by the judge.
    """
    urls: list[str] = []
    for call in tool_log:
        if call.get("tool") == "check_url_validity":
            inp = call.get("input", {})
            url = inp.get("url", "") if isinstance(inp, dict) else ""
            if url:
                urls.append(url)
    return urls


def url_to_domain(url: str) -> str:
    """Extract the parent domain (without www.) from a URL."""
    from urllib.parse import urlparse
    try:
        netloc = urlparse(url).netloc.lower()
        if netloc.startswith("www."):
            netloc = netloc[4:]
        return netloc or url
    except Exception:
        return url


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
                "Non-agentic valid": fmt_valid(score_to_valid(na_score)),
                "Non-agentic score": round(float(na_score), 3) if na_score is not None else None,
                "Agentic valid": fmt_valid(score_to_valid(ag_score)),
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

    st.markdown("---")

    # ── URL domain analysis ──────────────────────────────────────────────────
    st.subheader("🌐 URL Domain Analysis")

    import pandas as pd
    from collections import Counter

    _is_en = lambda l: l.lower().startswith("en")

    # ── Build validity records: one entry per (url, row, policy) ────────────
    # Ground truth: every URL found by regex in the assistant response.
    # After the post-loop URL sweep in agentic_runner.py, every URL in the
    # response has a corresponding check_url_validity result stored in
    # {policy}_agentic_url_checks.  We join them here so we can show
    # validity (valid / invalid / unchecked) rather than just coverage.
    _coverage_records: list[dict] = []
    for _row in valid_rows:
        _lang = _row.get("language", "unknown")
        _resp_urls = list(dict.fromkeys(  # deduplicate while preserving order
            extract_urls_from_text(_row.get("assistant_response", ""))
        ))
        for _lbl in policy_labels:
            # Build a lookup: url → check result dict, from stored url_checks
            _uc_list = ensure_list(_row.get(f"{_lbl}_agentic_url_checks", []))
            _uc_map: dict[str, dict] = {}
            for _uc in _uc_list:
                _u = _uc.get("url", "")
                if _u:
                    _uc_map[_u] = _uc
            for _url in _resp_urls:
                _chk = _uc_map.get(_url)
                _coverage_records.append({
                    "url": _url,
                    "domain": url_to_domain(_url),
                    "policy": _lbl,
                    "language": _lang,
                    # was_checked: True means check_url_validity ran (post-loop sweep
                    # ensures this for all new runs; old files may still have gaps)
                    "was_checked": _chk is not None,
                    # is_valid: True/False from the HTTP check; None if not checked
                    "is_valid": _chk.get("valid") if _chk else None,
                    "status_code": _chk.get("status_code") if _chk else None,
                })

    # ── Build judge-research records: fetch_url + search_web (Phase 1) ──────
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

    # ── Shared renderer (domain chart + policy + language breakdown + CSV) ───
    def _render_domain_breakdown(
        records: list[dict],
        bar_color: str,
        csv_suffix: str,
        policy_cols: list[str],
        count_field: str = "url",   # field to count per record
    ) -> None:
        if not records:
            st.info("No URLs found in this category for the loaded file.")
            return

        _all_langs_local   = sorted({r["language"] for r in records})
        _en_langs_local    = [l for l in _all_langs_local if _is_en(l)]
        _other_langs_local = [l for l in _all_langs_local if not _is_en(l)]

        _dc = Counter(r["domain"] for r in records)
        _t20 = _dc.most_common(20)
        _df_t = pd.DataFrame(_t20, columns=["Domain", "Count"])
        st.markdown(
            f"**Top 20 domains** — {sum(_dc.values()):,} total · "
            f"{len(_dc):,} unique domains"
        )
        try:
            import altair as alt
            st.altair_chart(
                alt.Chart(_df_t).mark_bar(color=bar_color).encode(
                    x=alt.X("Count:Q"),
                    y=alt.Y("Domain:N", sort="-x", title=None),
                    tooltip=["Domain", "Count"],
                ).properties(height=min(38 * len(_t20), 520)),
                use_container_width=True,
            )
        except ImportError:
            st.dataframe(_df_t, use_container_width=True)

        if policy_cols:
            st.markdown("**By policy file**")
            _pc_cols = st.columns(len(policy_cols))
            for _pc, _plbl in zip(_pc_cols, policy_cols):
                _sub = Counter(r["domain"] for r in records if r["policy"] == _plbl)
                _sub_df = pd.DataFrame(_sub.most_common(20), columns=["Domain", "Count"])
                _pc.markdown(f"`{_plbl}` — {sum(_sub.values()):,}")
                _pc.dataframe(_sub_df, use_container_width=True, height=280)

        st.markdown("**By scenario language — English vs non-English**")
        _lc1, _lc2 = st.columns(2)
        with _lc1:
            _enc = Counter(r["domain"] for r in records if _is_en(r["language"]))
            _en_lbl = " / ".join(_en_langs_local) if _en_langs_local else "en"
            st.markdown(f"**English** (`{_en_lbl}`) — {sum(_enc.values()):,}")
            if _enc:
                st.dataframe(
                    pd.DataFrame(_enc.most_common(20), columns=["Domain", "Count"]),
                    use_container_width=True, height=280,
                )
            else:
                st.caption("No English rows.")
        with _lc2:
            _occ = Counter(r["domain"] for r in records if not _is_en(r["language"]))
            _oth_lbl = " / ".join(_other_langs_local) if _other_langs_local else "non-en"
            st.markdown(f"**Non-English** (`{_oth_lbl}`) — {sum(_occ.values()):,}")
            if _occ:
                st.dataframe(
                    pd.DataFrame(_occ.most_common(20), columns=["Domain", "Count"]),
                    use_container_width=True, height=280,
                )
            else:
                st.caption("No non-English rows.")

        with st.expander("Full URL list + CSV download", expanded=False):
            _url_agg: dict[str, dict] = {}
            for _rec in records:
                _u = _rec["url"]
                if _u not in _url_agg:
                    _url_agg[_u] = {
                        "url": _u, "domain": _rec["domain"], "total": 0,
                        **{f"policy_{p}": 0 for p in policy_cols},
                        **{f"lang_{l}": 0 for l in _all_langs_local},
                    }
                _url_agg[_u]["total"] += 1
                if _rec["policy"] in policy_cols:
                    _url_agg[_u][f"policy_{_rec['policy']}"] += 1
                _url_agg[_u][f"lang_{_rec['language']}"] += 1
            _df_full = pd.DataFrame(sorted(_url_agg.values(), key=lambda x: -x["total"]))
            st.dataframe(_df_full, use_container_width=True, height=280)
            st.download_button(
                label="⬇️  Download as CSV",
                data=_df_full.to_csv(index=False).encode("utf-8"),
                file_name=f"{selected_file.replace('.json', '')}_{csv_suffix}.csv",
                mime="text/csv",
            )

    # ── Two tabs ─────────────────────────────────────────────────────────────
    _n_asst  = len({r["url"] for r in _coverage_records})
    _n_res   = len({r["url"] for r in _research_records})
    _utab1, _utab2 = st.tabs([
        f"📄 Assistant Response URL Validity ({_n_asst} unique)",
        f"🔍 Judge Research — fetch + search ({_n_res} unique)",
    ])

    with _utab1:
        st.caption(
            "Every URL found in the assistant response by regex is checked via "
            "`check_url_validity` (a programmatic post-loop sweep guarantees full "
            "coverage for new runs). **Valid** = HTTP < 400 or 401/403. "
            "**Invalid** = HTTP ≥ 404, connection failure, or fabricated URL. "
            "Old output files produced before the sweep was added may show "
            "some URLs as *unchecked*."
        )

        if not _coverage_records:
            st.info("No URLs found in assistant responses in this file.")
        else:
            _all_langs_cov = sorted({r["language"] for r in _coverage_records})
            _en_langs_cov  = [l for l in _all_langs_cov if _is_en(l)]
            _oth_langs_cov = [l for l in _all_langs_cov if not _is_en(l)]

            # ── Validity summary metrics ─────────────────────────────────────
            _total      = len(_coverage_records)
            _n_valid    = sum(1 for r in _coverage_records if r["is_valid"] is True)
            _n_invalid  = sum(1 for r in _coverage_records if r["is_valid"] is False)
            _n_unchkd   = sum(1 for r in _coverage_records if r["is_valid"] is None)
            _valid_pct  = 100 * _n_valid  / _total if _total else 0
            _cm1, _cm2, _cm3, _cm4 = st.columns(4)
            _cm1.metric("URLs in responses", f"{_total:,}",
                        help="Total URL appearances found by regex in all assistant responses")
            _cm2.metric("Valid", f"{_n_valid:,}",
                        help="URL returned HTTP < 400 (or 401/403 — resource exists, access-restricted)")
            _cm3.metric("Invalid / unreachable", f"{_n_invalid:,}",
                        help="HTTP ≥ 404, connection error, or timeout — likely broken or fabricated")
            _cm4.metric("Validity rate", f"{_valid_pct:.1f}%",
                        help="Valid / (Valid + Invalid). Unchecked URLs are excluded from denominator."
                             if _n_unchkd else "Valid / Total")
            if _n_unchkd:
                st.warning(
                    f"⚠️ {_n_unchkd} URL appearance(s) have no check result — "
                    "likely from an output file produced before the automatic URL sweep was added. "
                    "Re-run the evaluation to get full validity data."
                )

            # Per-policy validity
            st.markdown("**Validity rate by policy**")
            _cov_pol_cols = st.columns(len(policy_labels))
            for _cpc, _plbl in zip(_cov_pol_cols, policy_labels):
                _pol_recs   = [r for r in _coverage_records if r["policy"] == _plbl]
                _pol_valid  = sum(1 for r in _pol_recs if r["is_valid"] is True)
                _pol_chkd   = sum(1 for r in _pol_recs if r["is_valid"] is not None)
                _pol_pct    = 100 * _pol_valid / _pol_chkd if _pol_chkd else 0
                _cpc.metric(
                    f"`{_plbl}`", f"{_pol_pct:.1f}%",
                    help=f"{_pol_valid} valid / {_pol_chkd} checked ({len(_pol_recs)} found in responses)",
                )

            st.markdown("---")

            # ── Domain chart: valid vs invalid ───────────────────────────────
            st.markdown("**Top 20 domains — valid vs invalid URLs**")
            _dc_all = Counter(r["domain"] for r in _coverage_records)
            _dc_val = Counter(r["domain"] for r in _coverage_records if r["is_valid"] is True)
            _dc_inv = Counter(r["domain"] for r in _coverage_records if r["is_valid"] is False)
            _dc_unc = Counter(r["domain"] for r in _coverage_records if r["is_valid"] is None)
            _cov_df = pd.DataFrame([
                {
                    "Domain": d,
                    "Valid": _dc_val.get(d, 0),
                    "Invalid": _dc_inv.get(d, 0),
                    "Unchecked": _dc_unc.get(d, 0),
                }
                for d, _ in _dc_all.most_common(20)
            ])
            try:
                import altair as alt
                _fold_cols = ["Valid", "Invalid"]
                if _dc_unc:
                    _fold_cols.append("Unchecked")
                _stacked = alt.Chart(_cov_df).transform_fold(
                    _fold_cols,
                    as_=["Status", "Count"],
                ).mark_bar().encode(
                    x=alt.X("Count:Q"),
                    y=alt.Y("Domain:N", sort="-x", title=None),
                    color=alt.Color("Status:N", scale=alt.Scale(
                        domain=["Valid", "Invalid", "Unchecked"],
                        range=["#59a14f", "#e15759", "#aaaaaa"],
                    )),
                    tooltip=["Domain", "Status:N", "Count:Q"],
                ).properties(height=min(38 * len(_cov_df), 520))
                st.altair_chart(_stacked, use_container_width=True)
            except ImportError:
                st.dataframe(_cov_df, use_container_width=True)

            # By policy
            st.markdown("**By policy file**")
            _cov_p_cols = st.columns(len(policy_labels))
            for _cpc, _plbl in zip(_cov_p_cols, policy_labels):
                _pr = [r for r in _coverage_records if r["policy"] == _plbl]
                _pd_all = Counter(r["domain"] for r in _pr)
                _pd_val = Counter(r["domain"] for r in _pr if r["is_valid"] is True)
                _pd_inv = Counter(r["domain"] for r in _pr if r["is_valid"] is False)
                _pd_df = pd.DataFrame([
                    {
                        "Domain": d, "Found": _pd_all[d],
                        "Valid": _pd_val.get(d, 0),
                        "Invalid": _pd_inv.get(d, 0),
                    }
                    for d, _ in _pd_all.most_common(20)
                ])
                _cpc.markdown(f"`{_plbl}`")
                _cpc.dataframe(_pd_df, use_container_width=True, height=280)

            # By language
            st.markdown("**By scenario language — English vs non-English**")
            _lc1, _lc2 = st.columns(2)
            for _col, _filter, _label, _langs in [
                (_lc1, lambda r: _is_en(r["language"]),
                 "English", " / ".join(_en_langs_cov) if _en_langs_cov else "en"),
                (_lc2, lambda r: not _is_en(r["language"]),
                 "Non-English", " / ".join(_oth_langs_cov) if _oth_langs_cov else "non-en"),
            ]:
                with _col:
                    _lr = [r for r in _coverage_records if _filter(r)]
                    _la_all = Counter(r["domain"] for r in _lr)
                    _la_val = Counter(r["domain"] for r in _lr if r["is_valid"] is True)
                    _la_inv = Counter(r["domain"] for r in _lr if r["is_valid"] is False)
                    _l_valid_total = sum(1 for r in _lr if r["is_valid"] is True)
                    _l_chkd_total  = sum(1 for r in _lr if r["is_valid"] is not None)
                    _l_pct = 100 * _l_valid_total / _l_chkd_total if _l_chkd_total else 0
                    st.markdown(
                        f"**{_label}** (`{_langs}`) — "
                        f"{len(_lr):,} URLs · {_l_valid_total}/{_l_chkd_total} valid ({_l_pct:.1f}%)"
                    )
                    if _la_all:
                        _l_df = pd.DataFrame([
                            {"Domain": d, "Found": _la_all[d],
                             "Valid": _la_val.get(d, 0),
                             "Invalid": _la_inv.get(d, 0)}
                            for d, _ in _la_all.most_common(20)
                        ])
                        st.dataframe(_l_df, use_container_width=True, height=280)
                    else:
                        st.caption(f"No {_label.lower()} rows.")

            # Full CSV
            with st.expander("Full URL validity table + CSV download", expanded=False):
                _cov_agg: dict[tuple, dict] = {}
                for _rec in _coverage_records:
                    _key = (_rec["url"], _rec["policy"])
                    if _key not in _cov_agg:
                        _cov_agg[_key] = {
                            "url": _rec["url"], "domain": _rec["domain"],
                            "policy": _rec["policy"],
                            "appearances": 0, "valid": 0, "invalid": 0, "unchecked": 0,
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
                _df_cov_full = pd.DataFrame(
                    sorted(_cov_agg.values(), key=lambda x: -x["appearances"])
                )
                _checked_col = _df_cov_full["valid"] + _df_cov_full["invalid"]
                _df_cov_full["validity_%"] = (
                    _df_cov_full["valid"] / _checked_col.replace(0, float("nan")) * 100
                ).round(1)
                st.dataframe(_df_cov_full, use_container_width=True, height=280)
                st.download_button(
                    label="⬇️  Download as CSV",
                    data=_df_cov_full.to_csv(index=False).encode("utf-8"),
                    file_name=f"{selected_file.replace('.json', '')}_response_url_validity.csv",
                    mime="text/csv",
                )

    with _utab2:
        st.caption(
            "URLs the **judge consulted for its own research** (Phase 1 — claim verification): "
            "`fetch_url` inputs and links returned in `search_web` results."
        )
        _render_domain_breakdown(
            _research_records, bar_color="#4e79a7",
            csv_suffix="judge_research_urls", policy_cols=policy_labels,
        )


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
        st.markdown(fmt_valid(score_to_valid(na_score)))
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
        st.markdown(fmt_valid(score_to_valid(ag_score)))

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
                valid_turns = [
                    t for t in ag_per_turn
                    if t.get("prompt_tokens") is not None and t.get("completion_tokens") is not None
                ]
                if valid_turns:
                    chart_data = pd.DataFrame({
                        "Turn": [str(t["turn"]) for t in valid_turns],
                        "Prompt tokens": [int(t["prompt_tokens"]) for t in valid_turns],
                        "Completion tokens": [int(t["completion_tokens"]) for t in valid_turns],
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
                        tooltip=["Turn", "Type:N", "Tokens:Q"],
                    ).properties(height=250, title="Tokens per turn (prompt = context window consumed)")
                    st.altair_chart(chart, use_container_width=True)
                else:
                    st.info("No per-turn token data available for this scenario.")
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

    # ── Claim checks ─────────────────────────────────────────────────────────
    claim_checks = ensure_list(row.get(f"{label}_agentic_claim_checks", []))

    _STATUS_ICON = {
        "verified": "✅",
        "contradicted": "❌",
        "unverifiable": "⚠️",
    }
    _STATUS_ORDER = {"contradicted": 0, "unverifiable": 1, "verified": 2}

    if claim_checks:
        # Sort so problems surface first: contradicted → unverifiable → verified
        sorted_claims = sorted(
            claim_checks,
            key=lambda c: _STATUS_ORDER.get(str(c.get("status", "")).lower(), 3),
        )
        n_verified      = sum(1 for c in claim_checks if str(c.get("status","")).lower() == "verified")
        n_contradicted  = sum(1 for c in claim_checks if str(c.get("status","")).lower() == "contradicted")
        n_unverifiable  = sum(1 for c in claim_checks if str(c.get("status","")).lower() == "unverifiable")

        summary_parts = []
        if n_contradicted:
            summary_parts.append(f"❌ {n_contradicted} contradicted")
        if n_unverifiable:
            summary_parts.append(f"⚠️ {n_unverifiable} unverifiable")
        if n_verified:
            summary_parts.append(f"✅ {n_verified} verified")
        summary_str = " · ".join(summary_parts)

        with st.expander(
            f"🔍 Claim Verification ({len(claim_checks)} claims — {summary_str})",
            expanded=True,
        ):
            st.caption(
                "Claims extracted from the assistant response and checked via web search. "
                "Contradicted and unverifiable claims must lower the agentic score."
            )
            for c in sorted_claims:
                claim_text = str(c.get("claim", "—"))
                status_raw = str(c.get("status", "")).lower()
                icon = _STATUS_ICON.get(status_raw, "•")
                label_text = status_raw.capitalize() if status_raw else "Unknown"

                if status_raw == "verified":
                    st.success(f"{icon} **{label_text}** — {claim_text}")
                elif status_raw == "contradicted":
                    st.error(f"{icon} **{label_text}** — {claim_text}")
                else:
                    st.warning(f"{icon} **{label_text}** — {claim_text}")
    else:
        st.caption(
            "No claim checks recorded — either no verifiable factual claims were found "
            "or this result was produced before claim tracking was added."
        )


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
                c2.markdown(fmt_valid(score_to_valid(na_s)))
                with st.expander("Explanation"):
                    st.write(row_cmp.get(f"{lbl}_nonagentic_explanation", "—"))

                st.markdown("**Agentic**")
                c3, c4 = st.columns(2)
                c3.metric("Score", f"{float(ag_s):.3f}" if ag_s is not None else "—",
                          delta=f"{float(row_cmp.get(f'{lbl}_score_delta', 0) or 0):+.3f}")
                c4.markdown(fmt_valid(score_to_valid(ag_s)))
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
            valid_per_turn = [
                t for t in per_turn
                if t.get("prompt_tokens") is not None and t.get("completion_tokens") is not None
            ]
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
                    y=alt.Y("Tokens:Q", title="Tokens"),
                    color=alt.Color("Type:N", scale=alt.Scale(
                        domain=["prompt_tokens", "completion_tokens"],
                        range=["#4e79a7", "#f28e2b"],
                    )),
                    tooltip=["turn_label", "Type:N", "Tokens:Q",
                             alt.Tooltip("has_tool_calls:N", title="Tool calls?")],
                ).properties(
                    height=280,
                    title="Context window growth (prompt = all prior messages + tool results)",
                )
                st.altair_chart(growth_chart, use_container_width=True)
            else:
                st.info("No per-turn token data available for this scenario.")
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
