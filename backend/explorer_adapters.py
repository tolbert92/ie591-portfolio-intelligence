"""Connect completed Skill #5 frontiers to the Skill #6 explorer."""

from __future__ import annotations

from .explorer_models import ExplorerDataset
from .visualization_models import EfficientFrontier, VisualizationDataset


def _select_frontier(
    dataset: VisualizationDataset,
    portfolio_id: str,
    analysis_id: str | None,
) -> EfficientFrontier:
    """Resolve a frontier by portfolio, with an optional source analysis ID."""
    matches = [
        frontier
        for frontier in dataset.efficient_frontiers
        if frontier.portfolio_id == portfolio_id
        and (analysis_id is None or frontier.provenance.analysis_id == analysis_id)
    ]
    if not matches:
        raise ValueError(f"No matching efficient frontier for portfolio {portfolio_id!r}")
    if len(matches) > 1:
        if analysis_id is None:
            raise ValueError(
                f"Multiple frontiers for portfolio {portfolio_id!r}; specify an analysis ID"
            )
        raise ValueError(
            f"Multiple frontiers for portfolio {portfolio_id!r} and analysis {analysis_id!r}"
        )
    return matches[0]


def available_tickers(
    dataset: VisualizationDataset,
    portfolio_id: str,
    *,
    analysis_id: str | None = None,
) -> tuple[str, ...]:
    """Return tickers with an explicit allocation at every frontier point."""
    frontier = _select_frontier(dataset, portfolio_id, analysis_id)
    tickers = {row.ticker for row in frontier.points[0].allocations}
    for point in frontier.points[1:]:
        tickers.intersection_update(row.ticker for row in point.allocations)
    return tuple(sorted(tickers))


def build_explorer_dataset(
    dataset: VisualizationDataset,
    portfolio_id: str,
    selected_ticker: str,
    *,
    analysis_id: str | None = None,
) -> ExplorerDataset:
    """Use the selected frontier to build risk, return, and weight coordinates."""
    frontier = _select_frontier(dataset, portfolio_id, analysis_id)
    return ExplorerDataset.from_frontier(frontier, selected_ticker)


__all__ = ["available_tickers", "build_explorer_dataset"]
