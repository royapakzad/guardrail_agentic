"""Tests for the humanitarian domain tools (PR #18). HTTP is mocked."""

import tools
from tools import get_tool_schemas


def test_humanitarian_group_registered():
    names = [s["function"]["name"] for s in get_tool_schemas("humanitarian")]
    assert names == [
        "search_web",
        "fetch_url",
        "check_url_validity",
        "check_acronym",
        "reliefweb_situation",
        "disaster_alert",
        "health_advisory",
        "aid_org_verify",
    ]


def test_disaster_alert_filters_and_parses_gdacs(monkeypatch):
    fake = {
        "features": [
            {
                "properties": {
                    "eventtype": "TC",
                    "eventname": "Typhoon X",
                    "alertlevel": "Red",
                    "country": "Philippines",
                    "fromdate": "2026-06-01",
                    "url": {"report": "https://gdacs/r"},
                    "description": "Severe",
                }
            },
            {
                "properties": {
                    "eventtype": "EQ",
                    "name": "Quake",
                    "alertlevel": "Green",
                    "country": "Chile",
                }
            },
        ]
    }
    monkeypatch.setattr(tools, "_http_json", lambda *a, **k: fake)
    result = tools.disaster_alert("philippines")
    assert len(result["alerts"]) == 1
    alert = result["alerts"][0]
    assert alert["event_type"] == "TC"
    assert alert["alert_level"] == "Red"
    assert alert["url"] == "https://gdacs/r"


def test_health_advisory_parses_who_indicators(monkeypatch):
    monkeypatch.setattr(
        tools,
        "_http_json",
        lambda *a, **k: {"value": [{"IndicatorCode": "C1", "IndicatorName": "Cholera deaths"}]},
    )
    result = tools.health_advisory("cholera")
    assert result["who_indicators"] == [{"code": "C1", "name": "Cholera deaths"}]


def test_reliefweb_falls_back_to_default_appname_when_unset(monkeypatch):
    monkeypatch.delenv("RELIEFWEB_APPNAME", raising=False)
    captured = {}

    def fake_http_json(url, params=None, **kwargs):
        captured["appname"] = params.get("appname")
        return {"data": []}

    monkeypatch.setattr(tools, "_http_json", fake_http_json)
    tools.reliefweb_situation("Sudan")
    assert captured["appname"] == tools._RELIEFWEB_DEFAULT_APPNAME


def test_reliefweb_uses_env_appname_when_set(monkeypatch):
    monkeypatch.setenv("RELIEFWEB_APPNAME", "my-own-appname")
    captured = {}

    def fake_http_json(url, params=None, **kwargs):
        captured["appname"] = params.get("appname")
        return {"data": []}

    monkeypatch.setattr(tools, "_http_json", fake_http_json)
    tools.reliefweb_situation("Sudan")
    assert captured["appname"] == "my-own-appname"


def test_reliefweb_situation_parses_reports(monkeypatch):
    monkeypatch.setenv("RELIEFWEB_APPNAME", "approved")
    fake = {
        "data": [
            {
                "fields": {
                    "title": "Sudan crisis update",
                    "url": "https://rw/1",
                    "source": [{"name": "OCHA"}],
                    "date": {"created": "2026-06-01"},
                }
            }
        ]
    }
    monkeypatch.setattr(tools, "_http_json", lambda *a, **k: fake)
    result = tools.reliefweb_situation("Sudan")
    assert result["results"][0]["source"] == "OCHA"
    assert result["results"][0]["title"] == "Sudan crisis update"


def test_aid_org_verify_confirms_vetted_source(monkeypatch):
    monkeypatch.setenv("RELIEFWEB_APPNAME", "approved")
    fake = {
        "data": [
            {
                "fields": {
                    "name": "Norwegian Refugee Council",
                    "homepage": "https://nrc.no",
                    "type": {"name": "NGO"},
                }
            }
        ]
    }
    monkeypatch.setattr(tools, "_http_json", lambda *a, **k: fake)
    result = tools.aid_org_verify("Norwegian Refugee Council")
    assert result["verified"] is True
