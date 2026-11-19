
/**
 * IE 591 — Portfolio Intelligence
 * Gate 3.5 — Playwright Browser Test Configuration
 *
 * Starts the FastAPI application locally and runs
 * browser integration tests using Chromium.
 *
 * University of Tennessee
 * Educational application validation
 */

const { defineConfig, devices } = require("@playwright/test");

module.exports = defineConfig({
  testDir: "./browser-tests",

  timeout: 30000,

  expect: {
    timeout: 10000
  },

  fullyParallel: false,

  forbidOnly: !!process.env.CI,

  retries: process.env.CI ? 2 : 0,

  workers: process.env.CI ? 1 : undefined,

  reporter: process.env.CI
    ? [
        ["list"],
        ["html", {
          open: "never",
          outputFolder: "playwright-report"
        }]
      ]
    : [["list"]],

  use: {
    baseURL: "http://127.0.0.1:8000",

    trace: "retain-on-failure",

    screenshot: "only-on-failure",

    video: "retain-on-failure"
  },

  projects: [
    {
      name: "chromium",
      use: {
        ...devices["Desktop Chrome"]
      }
    }
  ],

  webServer: {
    command: "python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000",

    url: "http://127.0.0.1:8000/api/health",

    reuseExistingServer: !process.env.CI,

    timeout: 120000
  },

  outputDir: "test-results"
});
