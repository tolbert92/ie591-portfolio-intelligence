
/**
 * IE 591 — Portfolio Intelligence
 * Gate 3.5 — Automated Browser Integration Tests
 *
 * Runs against the local FastAPI-served HTML application.
 * Market API responses are mocked with deterministic
 * synthetic data to validate frontend behavior.
 *
 * This file requires @playwright/test.
 */

const { test, expect } = require("@playwright/test");

const APP_URL = "http://127.0.0.1:8000/app";

const TWO_ASSET_RESULT = {
  analysis_id: "IE591-GATE34-TEST",
  status: "completed",
  provenance: {
    source_type: "synthetic",
    source_name: "IE 591 Browser Validation Dataset",
    dataset_id: "BROWSER-TWO-ASSET",
    retrieved_at: null,
    source_url: null,
    is_synthetic: true
  },
  config: {
    frequency: "monthly",
    price_basis: "adjusted_close",
    return_method: "simple",
    missing_data_policy: "common_dates",
    annualization_periods: 12,
    minimum_common_observations: 3
  },
  quality: {
    status: "pass",
    source_observation_count: 8,
    valid_common_price_dates: 4,
    valid_common_return_periods: 3,
    excluded_dates: [],
    issues: []
  },
  asset_statistics: [
    {
      ticker: "VTI",
      observation_count: 3,
      annualized_arithmetic_return: 0.1595961949,
      annualized_volatility: 0.0713192243,
      cumulative_return: 0.04
    },
    {
      ticker: "BND",
      observation_count: 3,
      annualized_arithmetic_return: 0.0998428035,
      annualized_volatility: 0.0447582782,
      cumulative_return: 0.025
    }
  ],
  correlation_matrix: {
    tickers: ["VTI", "BND"],
    values: [
      [1, 0.9999984517],
      [0.9999984517, 1]
    ]
  },
  covariance_matrix: {
    tickers: ["VTI", "BND"],
    values: [
      [0.0050864318, 0.0031921207],
      [0.0031921207, 0.0020033035]
    ]
  },
  calculation_version: "1.0.0",
  downstream_ready: true
};

const THREE_ASSET_RESULT = {
  ...TWO_ASSET_RESULT,
  analysis_id: "IE591-GATE34-THREE",
  provenance: {
    ...TWO_ASSET_RESULT.provenance,
    dataset_id: "BROWSER-THREE-ASSET"
  },
  quality: {
    ...TWO_ASSET_RESULT.quality,
    source_observation_count: 12
  },
  asset_statistics: [
    ...TWO_ASSET_RESULT.asset_statistics,
    {
      ticker: "SGOV",
      observation_count: 3,
      annualized_arithmetic_return: 0.0259461972,
      annualized_volatility: 0.002645753,
      cumulative_return: 0.0065
    }
  ],
  correlation_matrix: {
    tickers: ["VTI", "BND", "SGOV"],
    values: [
      [1, 0.9999984517, -0.9949274575],
      [0.9999984517, 1, -0.995102935],
      [-0.9949274575, -0.995102935, 1]
    ]
  },
  covariance_matrix: {
    tickers: ["VTI", "BND", "SGOV"],
    values: [
      [0.0050864318, 0.0031921207, -0.0001877359],
      [0.0031921207, 0.0020033035, -0.0001178394],
      [-0.0001877359, -0.0001178394, 0.000007]
    ]
  }
};

async function mockMarketApi(page) {
  await page.route("**/api/market/status", async route => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        skill_id: 2,
        engine: "available",
        calculation_version: "1.0.0",
        external_data_fetching: false
      })
    });
  });

  await page.route("**/api/market/analyze", async route => {
    const request = route.request().postDataJSON();

    const result = request.assets.length === 3
      ? THREE_ASSET_RESULT
      : TWO_ASSET_RESULT;

    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify(result)
    });
  });
}

test.beforeEach(async ({ page }) => {
  await mockMarketApi(page);
  await page.goto(APP_URL);

  await page
    .getByRole("button", { name: /Market Intelligence/i })
    .click();
});

test("Market Intelligence workspace opens", async ({ page }) => {
  await expect(
    page.getByRole("heading", {
      name: "Market Intelligence — Analytical Workspace"
    })
  ).toBeVisible();

  await expect(
    page.locator("#marketDataset")
  ).toBeVisible();

  await expect(
    page.locator("#marketAnalyzeBtn")
  ).toBeEnabled();
});

test("Market engine connection succeeds", async ({ page }) => {
  await page.locator("#marketCheckBtn").click();

  await expect(
    page.locator("#marketFeedback")
  ).toContainText("Market engine available");

  await expect(
    page.locator("#marketFeedback")
  ).toContainText("1.0.0");
});

test("Two-asset analysis displays results", async ({ page }) => {
  await page.locator("#marketDataset").selectOption("two");

  await page.locator("#marketAnalyzeBtn").click();

  await expect(
    page.locator("#marketStatusValue")
  ).toHaveText("completed");

  await expect(
    page.locator("#marketAssetCount")
  ).toHaveText("2");

  await expect(
    page.locator("#marketReady")
  ).toHaveText("Yes");

  await expect(
    page.locator("#marketStatsBody")
  ).toContainText("15.96%");

  await expect(
    page.locator("#marketStatsBody")
  ).toContainText("9.98%");
});

test("Three-asset analysis displays results", async ({ page }) => {
  await page.locator("#marketDataset").selectOption("three");

  await page.locator("#marketAnalyzeBtn").click();

  await expect(
    page.locator("#marketAssetCount")
  ).toHaveText("3");

  await expect(
    page.locator("#marketStatsBody")
  ).toContainText("SGOV");

  await expect(
    page.locator("#marketCorrelation table")
  ).toBeVisible();

  await expect(
    page.locator("#marketCovariance table")
  ).toBeVisible();
});

test("Export is disabled before analysis", async ({ page }) => {
  await expect(
    page.locator("#marketExportBtn")
  ).toBeDisabled();
});

test("Export downloads completed analysis", async ({ page }) => {
  await page.locator("#marketAnalyzeBtn").click();

  await expect(
    page.locator("#marketExportBtn")
  ).toBeEnabled();

  const downloadPromise = page.waitForEvent("download");

  await page.locator("#marketExportBtn").click();

  const download = await downloadPromise;

  expect(download.suggestedFilename())
    .toBe("ie591-market-analysis.json");
});

test("API errors display feedback", async ({ page }) => {
  await page.unroute("**/api/market/analyze");

  await page.route("**/api/market/analyze", async route => {
    await route.fulfill({
      status: 422,
      contentType: "application/json",
      body: JSON.stringify({
        detail: "Synthetic validation error"
      })
    });
  });

  await page.locator("#marketAnalyzeBtn").click();

  await expect(
    page.locator("#marketFeedback")
  ).toContainText("Market analysis failed");

  await expect(
    page.locator("#marketExportBtn")
  ).toBeDisabled();
});

test("Investor Profile remains accessible", async ({ page }) => {
  await page
    .getByRole("button", { name: /Investor Profile/i })
    .first()
    .click();

  await expect(
    page.getByRole("heading", {
      name: "Investor Profile Builder"
    })
  ).toBeVisible();

  await expect(
    page.locator("#saveProfileBtn")
  ).toBeEnabled();
});
