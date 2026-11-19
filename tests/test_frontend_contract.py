
"""
IE 591 — Portfolio Intelligence
Gate 3.5: Frontend Contract Tests

Validates that the HTML application retains the
required Skill #1 and Skill #2 integration contracts.

These tests inspect source structure and do not
launch a browser or call the live Render service.
"""

from html.parser import HTMLParser
from pathlib import Path
import re

import pytest


ROOT = Path(__file__).resolve().parents[1]
INDEX_HTML = ROOT / "index.html"


class FrontendParser(HTMLParser):
    """Collect HTML elements, attributes and inline scripts."""

    def __init__(self):
        super().__init__()
        self.elements = []
        self.scripts = []
        self.in_script = False

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        self.elements.append((tag, attributes))

        if tag == "script" and "src" not in attributes:
            self.in_script = True

    def handle_endtag(self, tag):
        if tag == "script":
            self.in_script = False

    def handle_data(self, data):
        if self.in_script:
            self.scripts.append(data)

    def find_id(self, element_id):
        for tag, attrs in self.elements:
            if attrs.get("id") == element_id:
                return tag, attrs
        return None


@pytest.fixture(scope="module")
def frontend():
    assert INDEX_HTML.is_file(), (
        "index.html must exist in the repository root."
    )

    source = INDEX_HTML.read_text(encoding="utf-8")

    parser = FrontendParser()
    parser.feed(source)

    return {
        "source": source,
        "parser": parser,
        "javascript": "\n".join(parser.scripts),
    }


def assert_element(frontend, element_id):
    element = frontend["parser"].find_id(element_id)

    assert element is not None, (
        f"Required HTML element missing: {element_id}"
    )

    return element


def assert_function(frontend, function_name):
    javascript = frontend["javascript"]

    pattern = (
        r"\b(?:async\s+)?function\s+"
        + re.escape(function_name)
        + r"\s*\("
    )

    assert re.search(pattern, javascript), (
        f"Required JavaScript function missing: "
        f"{function_name}"
    )


def test_html_document_exists(frontend):
    source = frontend["source"].lower()

    assert "<!doctype html>" in source
    assert "<html" in source
    assert "</html>" in source


def test_application_navigation(frontend):
    for element_id in [
        "sidebar",
        "nav",
        "breadcrumb",
        "pageTitle",
        "dashboard",
        "profile",
        "market",
        "workspace",
    ]:
        assert_element(frontend, element_id)

    assert_function(frontend, "renderNav")
    assert_function(frontend, "go")


def test_investor_profile_fields_preserved(frontend):
    required_fields = [
        "investorId",
        "investorName",
        "investment",
        "horizon",
        "risk",
        "goal",
        "contribution",
        "apiBase",
    ]

    for field in required_fields:
        assert_element(frontend, field)


def test_investor_profile_api_contract(frontend):
    for function_name in [
        "getProfile",
        "formToApi",
        "checkApi",
        "saveProfile",
        "loadProfile",
    ]:
        assert_function(frontend, function_name)

    javascript = frontend["javascript"]

    assert "/api/health" in javascript
    assert "/api/investor-profile" in javascript


def test_market_workspace_fields(frontend):
    for element_id in [
        "marketAnalysisId",
        "marketDataset",
        "marketFeedback",
        "marketCheckBtn",
        "marketAnalyzeBtn",
        "marketExportBtn",
    ]:
        assert_element(frontend, element_id)


def test_market_dataset_options(frontend):
    source = frontend["source"]

    assert 'value="two"' in source
    assert 'value="three"' in source


def test_market_api_contract(frontend):
    javascript = frontend["javascript"]

    assert "/api/market/status" in javascript
    assert "/api/market/analyze" in javascript

    assert_function(frontend, "checkMarketApi")
    assert_function(frontend, "runMarketAnalysis")


def test_market_request_builder(frontend):
    assert_function(frontend, "syntheticMarketPayload")

    javascript = frontend["javascript"]

    for field in [
        "analysis_id",
        "assets",
        "observations",
        "provenance",
        "config",
        "minimum_common_observations",
    ]:
        assert field in javascript


def test_synthetic_asset_universe(frontend):
    javascript = frontend["javascript"]

    for ticker in ["VTI", "BND", "SGOV"]:
        assert ticker in javascript

    assert "synthetic" in javascript.lower()


def test_market_result_containers(frontend):
    required_elements = [
        "marketResults",
        "marketStatusValue",
        "marketQualityValue",
        "marketAssetCount",
        "marketReady",
        "marketStatsBody",
        "marketCorrelation",
        "marketCovariance",
    ]

    for element_id in required_elements:
        assert_element(frontend, element_id)


def test_market_matrix_renderer(frontend):
    assert_function(frontend, "renderMarketMatrix")

    javascript = frontend["javascript"]

    assert "correlation_matrix" in javascript
    assert "covariance_matrix" in javascript


def test_market_quality_reporting(frontend):
    assert_element(frontend, "marketQualityDetails")
    assert_element(frontend, "marketIssues")

    javascript = frontend["javascript"]

    assert "quality" in javascript
    assert "downstream_ready" in javascript
    assert "calculation_version" in javascript


def test_market_json_export(frontend):
    assert_function(frontend, "exportMarketAnalysis")

    javascript = frontend["javascript"]

    assert "JSON.stringify" in javascript
    assert "application/json" in javascript


def test_export_initially_disabled(frontend):
    _, attrs = assert_element(
        frontend, "marketExportBtn"
    )

    assert "disabled" in attrs, (
        "Market export must be disabled until "
        "an analysis is available."
    )


def test_future_skills_remain_available(frontend):
    javascript = frontend["javascript"]

    required_modules = [
        "optimize",
        "risk",
        "charts",
        "3d",
        "forward",
        "monte",
        "compare",
        "decision",
        "reports",
    ]

    for module_id in required_modules:
        assert (
            f'id:"{module_id}"' in javascript
            or f'id: "{module_id}"' in javascript
        ), (
            f"Future workspace missing: {module_id}"
        )


def test_educational_disclosure(frontend):
    source = frontend["source"].lower()

    assert "educational" in source
    assert "synthetic" in source


def test_frontend_uses_hosted_backend(frontend):
    source = frontend["source"]

    assert (
        "ie591-portfolio-intelligence.onrender.com"
        in source
    )


def test_frontend_javascript_present(frontend):
    javascript = frontend["javascript"]

    assert javascript.strip(), (
        "Application JavaScript must be present."
    )
