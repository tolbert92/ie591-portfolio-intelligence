):
    payload = deepcopy(request_payload)

    # Analytical minimum volatility is
    # sqrt(0.008), approximately 8.94%.

    payload["constraints"][
        "maximum_annual_volatility"
    ] = 0.02

    with pytest.raises(OptimizationSolverError):
        solve_portfolio(
            build_request(payload)
        )


def test_invalid_covariance_rejected(
    request_payload
):
    payload = deepcopy(request_payload)

    # Symmetric with positive diagonal,
    # but not positive semidefinite.

    payload["covariance_matrix"] = [
        [0.01, 0.05],
        [0.05, 0.01],
    ]

    with pytest.raises(
        OptimizationSolverError,
        match="positive semidefinite",
    ):
        solve_portfolio(
            build_request(payload)
        )


def test_unsupported_objective_rejected(optimization_request):
    with pytest.raises(
        OptimizationSolverError,
        match="Unsupported optimization objective",
    ):
        solve_portfolio(
            optimization_request,
            objective="maximize_dividends",
        )


# --------------------------------------------------
# Solver diagnostics
# --------------------------------------------------

def test_solver_reports_iterations(optimization_request):
    result = solve_portfolio(optimization_request)

    assert result.iterations >= 0
    assert result.solver_message


def test_solver_reports_constraint_violation(optimization_request):
    result = solve_portfolio(optimization_request)

    assert (
        result.maximum_constraint_violation
        <= 1e-6
    )


def test_solver_uses_multiple_starting_points(optimization_request):
    result = solve_portfolio(optimization_request)

    assert result.starting_points_attempted >= 2


def test_solver_objective_value_is_finite(optimization_request):
    result = solve_portfolio(optimization_request)

    assert math.isfinite(
        result.objective_value
    )


def test_solver_deterministic(optimization_request):
    first = solve_portfolio(optimization_request)
    second = solve_portfolio(optimization_request)

    assert first.weights == pytest.approx(
        second.weights,
        abs=1e-8,
    )

    assert (
        first.metrics.annualized_variance
        == pytest.approx(
            second.metrics.annualized_variance,
            abs=1e-10,
        )
    )


def test_solver_does_not_mutate_request(optimization_request):
    original = optimization_request.model_dump(mode="json")

    solve_portfolio(optimization_request)

    assert optimization_request.model_dump(mode="json") == original


# --------------------------------------------------
# Three-asset portfolio
# --------------------------------------------------

def test_three_asset_minimum_variance(request_payload):
    payload = deepcopy(request_payload)

    payload["assets"].append({
        "ticker": "ASSET_C",
        "asset_name": "Synthetic Cash Equivalent",
        "asset_class": "cash_equivalent",
        "expected_annual_return": 0.03,
        "minimum_weight": 0.0,
        "maximum_weight": 1.0,
    })

    payload["covariance_matrix"] = [
        [0.04, 0.0, 0.0],
        [0.0, 0.01, 0.0],
        [0.0, 0.0, 0.0025],
    ]

    result = solve_portfolio(
        build_request(payload)
    )

    # For diagonal covariance, minimum-variance
    # weights are proportional to inverse variance.
    #
    # 1/0.04 = 25
    # 1/0.01 = 100
    # 1/0.0025 = 400
    #
    # Total = 525

    expected_weights = (
        25 / 525,
        100 / 525,
        400 / 525,
    )

    assert result.weights == pytest.approx(
        expected_weights,
        abs=1e-4,
    )


def test_three_asset_maximum_sharpe(request_payload):
    payload = deepcopy(request_payload)

    payload["assets"].append({
        "ticker": "ASSET_C",
        "asset_name": "Synthetic Cash Equivalent",
        "asset_class": "cash_equivalent",
        "expected_annual_return": 0.03,
        "minimum_weight": 0.0,
        "maximum_weight": 1.0,
    })

    payload["covariance_matrix"] = [
        [0.04, 0.0, 0.0],
        [0.0, 0.01, 0.0],
        [0.0, 0.0, 0.0025],
    ]

    result = solve_portfolio(
        build_request(payload),
        objective="maximum_sharpe",
    )

    # Excess return / variance:
    # A = 0.10 / 0.04 = 2.5
    # B = 0.04 / 0.01 = 4.0
    # C = 0.01 / 0.0025 = 4.0

    expected_weights = (
        2.5 / 10.5,
        4.0 / 10.5,
        4.0 / 10.5,
    )

    assert result.weights == pytest.approx(
        expected_weights,
