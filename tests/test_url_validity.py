"""Tests for three-state URL validity introduced in PR #17."""

from unittest.mock import MagicMock, patch

import requests as req_lib
from tools import check_url_validity


def _mock_response(status_code: int, url: str = "http://example.com") -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.url = url
    resp.history = []
    return resp


def test_valid_url_returns_valid_true():
    with patch("requests.head", return_value=_mock_response(200)):
        result = check_url_validity("http://example.com")
    assert result["valid"] is True
    assert result.get("timed_out") is not True
    assert result["status_code"] == 200


def test_404_returns_valid_false():
    with patch("requests.head", return_value=_mock_response(404)):
        result = check_url_validity("http://example.com/missing")
    assert result["valid"] is False
    assert result.get("timed_out") is not True


def test_401_treated_as_valid():
    with patch("requests.head", return_value=_mock_response(401)):
        result = check_url_validity("http://protected.example.com")
    assert result["valid"] is True


def test_403_treated_as_valid():
    with patch("requests.head", return_value=_mock_response(403)):
        result = check_url_validity("http://restricted.example.com")
    assert result["valid"] is True


def test_timeout_returns_timed_out_state():
    with patch("requests.head", side_effect=req_lib.Timeout("timed out")):
        result = check_url_validity("http://slow.example.com")
    # Third state: valid=None, timed_out=True
    assert result["valid"] is None
    assert result.get("timed_out") is True


def test_timed_out_is_not_valid_false():
    with patch("requests.head", side_effect=req_lib.Timeout("timed out")):
        result = check_url_validity("http://slow.example.com")
    assert result["valid"] is not False


def test_405_retries_with_get():
    get_resp = _mock_response(200)
    get_resp.close = MagicMock()
    with (
        patch("requests.head", return_value=_mock_response(405)),
        patch("requests.get", return_value=get_resp),
    ):
        result = check_url_validity("http://head-hostile.example.com")
    assert result["valid"] is True


def test_connection_error_returns_valid_false():
    with patch("requests.head", side_effect=req_lib.ConnectionError("refused")):
        result = check_url_validity("http://offline.example.com")
    assert result["valid"] is False
    assert result.get("timed_out") is not True


def test_result_always_contains_url_field():
    with patch("requests.head", return_value=_mock_response(200)):
        result = check_url_validity("http://example.com")
    assert result["url"] == "http://example.com"
