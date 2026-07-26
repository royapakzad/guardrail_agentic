"""Tests for the financial domain tools (PR #19). Cached lists are stubbed."""

import tools
from tools import get_tool_schemas


def test_financial_group_registered():
    names = [s["function"]["name"] for s in get_tool_schemas("financial")]
    assert names == [
        "search_web",
        "fetch_url",
        "check_url_validity",
        "check_acronym",
        "entity_registration",
        "sanctions_screen",
        "broker_license_check",
        "filing_search",
        "crypto_address_screen",
    ]


def test_entity_registration_finds_ticker(monkeypatch):
    monkeypatch.setattr(
        tools,
        "_SEC_TICKERS_CACHE",
        [{"cik_str": 1045810, "ticker": "NVDA", "title": "NVIDIA CORP"}],
    )
    result = tools.entity_registration("NVDA")
    assert result["registered"] is True
    assert result["matches"][0]["name"] == "NVIDIA CORP"
    assert result["matches"][0]["cik"] == 1045810


def test_entity_registration_unknown_is_unregistered(monkeypatch):
    monkeypatch.setattr(
        tools,
        "_SEC_TICKERS_CACHE",
        [{"cik_str": 1, "ticker": "AAA", "title": "Real Company Inc"}],
    )
    result = tools.entity_registration("Definitely Fake Capital LLC")
    assert result["registered"] is False
    assert result["matches"] == []


def test_sanctions_screen_flags_sdn_match(monkeypatch):
    monkeypatch.setattr(
        tools,
        "_OFAC_SDN_CACHE",
        [{"name": "BANCO NACIONAL DE CUBA", "type": "entity", "program": "CUBA"}],
    )
    result = tools.sanctions_screen("Banco Nacional de Cuba")
    assert result["sanctioned"] is True
    assert result["matches"][0]["program"] == "CUBA"


def test_sanctions_screen_clean_name(monkeypatch):
    monkeypatch.setattr(
        tools,
        "_OFAC_SDN_CACHE",
        [{"name": "BANCO NACIONAL DE CUBA", "type": "entity", "program": "CUBA"}],
    )
    result = tools.sanctions_screen("Acme Friendly Bakery")
    assert result["sanctioned"] is False


def test_sanctions_screen_short_query_is_safe(monkeypatch):
    monkeypatch.setattr(tools, "_OFAC_SDN_CACHE", [])
    result = tools.sanctions_screen("ab")
    assert result["sanctioned"] is None


# ── broker_license_check ──────────────────────────────────────────────────────


def _bc_payload(sources: list[dict]) -> dict:
    return {"hits": {"total": len(sources), "hits": [{"_source": s} for s in sources]}}


def test_broker_license_check_active_individual(monkeypatch):
    payload = _bc_payload(
        [
            {
                "ind_source_id": "1234567",
                "ind_firstname": "JANE",
                "ind_lastname": "DOE",
                "ind_bc_scope": "Active",
                "ind_ia_scope": "NotInScope",
                "ind_bc_disclosure_fl": "N",
                "ind_current_employments": [{"firm_name": "REAL BROKERAGE LLC"}],
            }
        ]
    )
    monkeypatch.setattr(tools, "_http_json", lambda url, **kw: payload)
    result = tools.broker_license_check("Jane Doe")
    assert result["licensed"] is True
    assert result["matches"][0]["crd"] == "1234567"
    assert result["matches"][0]["active"] is True
    assert result["matches"][0]["current_firms"] == ["REAL BROKERAGE LLC"]
    assert "disclosure" not in result["note"].lower()


def test_broker_license_check_no_record_is_red_flag(monkeypatch):
    monkeypatch.setattr(tools, "_http_json", lambda url, **kw: _bc_payload([]))
    result = tools.broker_license_check("Covenant Capital Group", kind="firm")
    assert result["licensed"] is False
    assert result["matches"] == []
    assert "red flag" in result["note"].lower()
    assert "not proof" in result["note"].lower() or "may legitimately" in result["note"].lower()


def test_broker_license_check_inactive_record_is_serious(monkeypatch):
    payload = _bc_payload(
        [
            {
                "ind_source_id": "7654321",
                "ind_firstname": "JOHN",
                "ind_lastname": "SMITH",
                "ind_bc_scope": "InActive",
                "ind_ia_scope": "NotInScope",
                "ind_bc_disclosure_fl": "Y",
            }
        ]
    )
    monkeypatch.setattr(tools, "_http_json", lambda url, **kw: payload)
    result = tools.broker_license_check("John Smith")
    assert result["licensed"] is False
    assert result["matches"][0]["active"] is False
    assert "none are currently active" in result["note"].lower()


def test_broker_license_check_active_with_disclosures(monkeypatch):
    payload = _bc_payload(
        [
            {
                "ind_source_id": "1111111",
                "ind_firstname": "PAT",
                "ind_lastname": "LEE",
                "ind_bc_scope": "Active",
                "ind_ia_scope": "Active",
                "ind_bc_disclosure_fl": "Y",
            }
        ]
    )
    monkeypatch.setattr(tools, "_http_json", lambda url, **kw: payload)
    result = tools.broker_license_check("Pat Lee")
    assert result["licensed"] is True
    assert "disclosure events" in result["note"]


def test_broker_license_check_firm_kind(monkeypatch):
    payload = _bc_payload(
        [
            {
                "firm_source_id": "99999",
                "firm_name": "REAL BROKERAGE LLC",
                "firm_bc_scope": "Active",
                "firm_ia_scope": "NotInScope",
                "firm_disclosure_fl": "N",
            }
        ]
    )
    captured = {}

    def fake_http(url, **kw):
        captured["url"] = url
        return payload

    monkeypatch.setattr(tools, "_http_json", fake_http)
    result = tools.broker_license_check("Real Brokerage", kind="firm")
    assert result["kind"] == "firm"
    assert "/firm" in captured["url"]
    assert result["licensed"] is True
    assert result["matches"][0]["name"] == "REAL BROKERAGE LLC"


def test_broker_license_check_short_query_is_safe(monkeypatch):
    monkeypatch.setattr(
        tools, "_http_json", lambda url, **kw: (_ for _ in ()).throw(AssertionError("no call"))
    )
    result = tools.broker_license_check("JD")
    assert result["licensed"] is None
    assert result["matches"] == []


# ── filing_search ─────────────────────────────────────────────────────────────


def test_filing_search_active_filer(monkeypatch):
    payload = {
        "hits": {
            "total": {"value": 42},
            "hits": [
                {
                    "_source": {
                        "file_type": "10-K",
                        "file_date": "2026-02-01",
                        "display_names": ["NVIDIA CORP  (NVDA)  (CIK 0001045810)"],
                    }
                }
            ],
        }
    }
    monkeypatch.setattr(tools, "_http_json", lambda url, **kw: payload)
    result = tools.filing_search("NVIDIA")
    assert result["has_filings"] is True
    assert result["total"] == 42
    assert result["recent_filings"][0]["form"] == "10-K"


def test_filing_search_shell_entity_red_flag(monkeypatch):
    payload = {"hits": {"total": {"value": 0}, "hits": []}}
    monkeypatch.setattr(tools, "_http_json", lambda url, **kw: payload)
    result = tools.filing_search("Covenant Capital Partners")
    assert result["has_filings"] is False
    assert "red flag" in result["note"].lower()
    assert "not proof of fraud" in result["note"].lower()


# ── crypto_address_screen ─────────────────────────────────────────────────────


def test_crypto_address_screen_sanctioned(monkeypatch):
    monkeypatch.setattr(
        tools,
        "_OFAC_CRYPTO_CACHE",
        {
            "12qtd5bfwrsdnsazy76uve1xycgntojh9h": {
                "address": "12QtD5BFwRsdNsAZY76UVE1xyCGNTojH9h",
                "asset": "XBT",
                "name": "SANCTIONED PARTY",
                "program": "CYBER2",
            }
        },
    )
    result = tools.crypto_address_screen("12QtD5BFwRsdNsAZY76UVE1xyCGNTojH9h")
    assert result["sanctioned"] is True
    assert result["match"]["program"] == "CYBER2"
    assert "illegal" in result["note"].lower()


def test_crypto_address_screen_clean_is_not_endorsement(monkeypatch):
    monkeypatch.setattr(tools, "_OFAC_CRYPTO_CACHE", {})
    result = tools.crypto_address_screen("1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa")
    assert result["sanctioned"] is False
    assert "not" in result["note"].lower() and "legitimate" in result["note"].lower()


def test_crypto_address_screen_short_input_safe(monkeypatch):
    monkeypatch.setattr(
        tools, "_OFAC_CRYPTO_CACHE", {}, raising=False
    )
    result = tools.crypto_address_screen("abc123")
    assert result["sanctioned"] is None


def test_ofac_crypto_parser_extracts_addresses(monkeypatch):
    csv_text = (
        '12345,"EVIL EXCHANGE","entity","CYBER2","-0-","-0-","-0-","-0-","-0-","-0-","-0-",'
        '"Digital Currency Address - XBT 12QtD5BFwRsdNsAZY76UVE1xyCGNTojH9h; '
        'Digital Currency Address - ETH 0x7F367cC41522cE07553e823bf3be79A889DEbe1B."\n'
    )
    monkeypatch.setattr(tools, "_http_text", lambda url, **kw: csv_text)
    monkeypatch.setattr(tools, "_OFAC_CRYPTO_CACHE", None)
    addresses = tools._load_ofac_crypto_addresses()
    assert "12qtd5bfwrsdnsazy76uve1xycgntojh9h" in addresses
    assert "0x7f367cc41522ce07553e823bf3be79a889debe1b" in addresses
    assert addresses["12qtd5bfwrsdnsazy76uve1xycgntojh9h"]["name"] == "EVIL EXCHANGE"


# ── regression: BrokerCheck fuzzy-match must not license the wrong party ──────


def test_broker_license_check_rejects_fuzzy_name_match(monkeypatch):
    """FINRA search is fuzzy: an invented name returns unrelated active registrants.

    Counting those as a licence would be a false 'licensed' verdict in exactly the
    direction the tool exists to prevent.
    """
    payload = _bc_payload(
        [
            {
                "ind_source_id": "5555555",
                "ind_firstname": "JORDAN",
                "ind_lastname": "MICHAELS",
                "ind_bc_scope": "Active",
                "ind_ia_scope": "Active",
                "ind_bc_disclosure_fl": "N",
            }
        ]
    )
    monkeypatch.setattr(tools, "_http_json", lambda url, **kw: payload)
    result = tools.broker_license_check("Jordan Belfort")
    assert result["licensed"] is False
    assert result["matches"] == []
    assert "partial-name matches" in result["note"]


def test_broker_license_check_accepts_true_name_match(monkeypatch):
    payload = _bc_payload(
        [
            {
                "ind_source_id": "6666666",
                "ind_firstname": "JANE",
                "ind_lastname": "DOE",
                "ind_bc_scope": "Active",
                "ind_ia_scope": "NotInScope",
                "ind_bc_disclosure_fl": "N",
            }
        ]
    )
    monkeypatch.setattr(tools, "_http_json", lambda url, **kw: payload)
    result = tools.broker_license_check("Jane Doe")
    assert result["licensed"] is True
    assert len(result["matches"]) == 1


# ── regression: EDGAR result cap must not be reported as an exact count ──────


def test_filing_search_reports_result_cap_honestly(monkeypatch):
    payload = {"hits": {"total": {"value": 10000}, "hits": []}}
    monkeypatch.setattr(tools, "_http_json", lambda url, **kw: payload)
    result = tools.filing_search("NVIDIA")
    assert result["result_cap_reached"] is True
    assert "10,000+" in result["note"]
    assert "result cap" in result["note"]


def test_filing_search_exact_count_below_cap(monkeypatch):
    payload = {"hits": {"total": {"value": 37}, "hits": []}}
    monkeypatch.setattr(tools, "_http_json", lambda url, **kw: payload)
    result = tools.filing_search("Smallcap Inc")
    assert result["result_cap_reached"] is False
    assert "37 EDGAR filing" in result["note"]
