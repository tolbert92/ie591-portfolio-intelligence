"""
IE 591 — Portfolio Intelligence
Skill #2: Market Intelligence

Gate 3.1 — Data Contracts

Defines validated input and output models for
historical market data analysis.

Educational use. Synthetic data is supported.
No live market data is retrieved by this module.
"""

from datetime import date, datetime
from enum import Enum
from math import isfinite
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)


# ------------------------------------------
# COMMON TYPES AND CONVENTIONS
# ------------------------------------------

Ticker = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        to_upper=True,
        min_length=1,
        max_length=20,
        pattern=r"^[A-Z0-9][A-Z0-9.\-]*$",
    ),
]

Currency = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        to_upper=True,
        min_length=3,
        max_length=3,
        pattern=r"^[A-Z]{3}$",
    ),
]


class DataSourceType(str, Enum):
    SYNTHETIC = "synthetic"
    UPLOADED = "uploaded"
    PROVIDER = "provider"


class ObservationFrequency(str, Enum):
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"


class PriceBasis(str, Enum):
    ADJUSTED_CLOSE = "adjusted_close"
    CLOSE = "close"


class ReturnMethod(str, Enum):
    SIMPLE = "simple"
    LOG = "log"


class MissingDataPolicy(str, Enum):
    COMMON_DATES = "common_dates"
    STRICT = "strict"


class QualityStatus(str, Enum):
    PASS = "pass"
    WARNING = "warning"
    FAIL = "fail"


class Severity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


# ------------------------------------------
# SOURCE PROVENANCE
# ------------------------------------------

class MarketDataProvenance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_type: DataSourceType
    source_name: str = Field(
        min_length=1,
        max_length=120,
    )
    dataset_id: str = Field(
        min_length=1,
        max_length=120,
    )
    retrieved_at: datetime | None = None
    source_url: str | None = Field(
        default=None,
        max_length=2048,
    )
    is_synthetic: bool = False

    @model_validator(mode="after")
    def validate_source(self):
        if (
            self.source_type == DataSourceType.SYNTHETIC
            and not self.is_synthetic
        ):
            raise ValueError(
                "Synthetic sources must have "
                "is_synthetic=True."
            )

        if (
            self.source_type != DataSourceType.SYNTHETIC
            and self.is_synthetic
        ):
            raise ValueError(
                "Non-synthetic sources cannot be "
                "marked as synthetic."
            )

        return self


# ------------------------------------------
# ASSET DEFINITION
# ------------------------------------------

class MarketAsset(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ticker: Ticker
    asset_name: str = Field(
        min_length=1,
        max_length=150,
    )
    asset_class: str = Field(
        min_length=1,
        max_length=80,
    )
    currency: Currency = "USD"


# ------------------------------------------
# HISTORICAL PRICE OBSERVATION
# ------------------------------------------

class PriceObservation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    observation_date: date
    ticker: Ticker
    price: float = Field(gt=0)

    @field_validator("price")
    @classmethod
    def finite_price(cls, value: float) -> float:
        if not isfinite(value):
            raise ValueError(
                "Price must be a finite number."
            )
        return value


# ------------------------------------------
# ANALYSIS CONFIGURATION
# ------------------------------------------

class MarketAnalysisConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    frequency: ObservationFrequency = (
        ObservationFrequency.MONTHLY
    )

    price_basis: PriceBasis = (
        PriceBasis.ADJUSTED_CLOSE
    )

    return_method: ReturnMethod = (
        ReturnMethod.SIMPLE
    )

    missing_data_policy: MissingDataPolicy = (
        MissingDataPolicy.COMMON_DATES
    )

    annualization_periods: int = Field(
        default=12,
        ge=1,
        le=366,
    )

    minimum_common_observations: int = Field(
        default=24,
        ge=3,
    )

    @model_validator(mode="after")
    def validate_annualization(self):
        expected = {
            ObservationFrequency.DAILY: 252,
            ObservationFrequency.WEEKLY: 52,
            ObservationFrequency.MONTHLY: 12,
        }

        if (
            self.annualization_periods
            != expected[self.frequency]
        ):
            raise ValueError(
                "Annualization periods must match "
                "the observation frequency."
            )

        return self


# ------------------------------------------
# MARKET ANALYSIS REQUEST
# ------------------------------------------

class MarketAnalysisRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    analysis_id: str = Field(
        min_length=1,
        max_length=120,
    )

    assets: list[MarketAsset] = Field(
        min_length=1,
        max_length=100,
    )

    observations: list[PriceObservation] = Field(
        min_length=3,
    )

    provenance: MarketDataProvenance

    config: MarketAnalysisConfig = Field(
        default_factory=MarketAnalysisConfig
    )

    @model_validator(mode="after")
    def validate_dataset(self):
        tickers = [asset.ticker for asset in self.assets]

        if len(tickers) != len(set(tickers)):
            raise ValueError(
                "Asset tickers must be unique."
            )

        valid_tickers = set(tickers)

        seen = set()

        for observation in self.observations:
            if observation.ticker not in valid_tickers:
                raise ValueError(
                    "Observation references an "
                    "undefined asset ticker."
                )

            key = (
                observation.ticker,
                observation.observation_date,
            )

            if key in seen:
                raise ValueError(
                    "Duplicate ticker/date "
                    "observation detected."
                )

            seen.add(key)

        return self


# ------------------------------------------
# DATA QUALITY OUTPUT
# ------------------------------------------

class DataQualityIssue(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str
    severity: Severity
    message: str

    ticker: Ticker | None = None
    observation_date: date | None = None


class DataQualityReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: QualityStatus

    source_observation_count: int = Field(ge=0)

    valid_common_price_dates: int = Field(ge=0)

    valid_common_return_periods: int = Field(ge=0)

    excluded_dates: list[date] = Field(
        default_factory=list
    )

    issues: list[DataQualityIssue] = Field(
        default_factory=list
    )


# ------------------------------------------
# RETURN AND VOLATILITY STATISTICS
# ------------------------------------------

class AssetStatistics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ticker: Ticker

    observation_count: int = Field(ge=0)

    mean_periodic_return: float

    annualized_arithmetic_return: float

    annualized_volatility: float = Field(ge=0)

    cumulative_return: float

    @field_validator(
        "mean_periodic_return",
        "annualized_arithmetic_return",
        "annualized_volatility",
        "cumulative_return",
    )
    @classmethod
    def finite_statistics(
        cls,
        value: float,
    ) -> float:
        if not isfinite(value):
            raise ValueError(
                "Statistics must be finite."
            )
        return value


# ------------------------------------------
# CORRELATION AND COVARIANCE
# ------------------------------------------

class MarketMatrix(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tickers: list[Ticker]

    values: list[list[float]]

    @model_validator(mode="after")
    def validate_matrix(self):
        n = len(self.tickers)

        if len(self.values) != n:
            raise ValueError(
                "Matrix row count must match "
                "ticker count."
            )

        for row in self.values:
            if len(row) != n:
                raise ValueError(
                    "Matrix must be square."
                )

            if not all(isfinite(x) for x in row):
                raise ValueError(
                    "Matrix values must be finite."
                )

        return self


# ------------------------------------------
# MARKET ANALYSIS RESPONSE
# ------------------------------------------

class MarketAnalysisResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    analysis_id: str

    status: Literal[
        "completed",
        "completed_with_warnings",
        "rejected",
    ]

    provenance: MarketDataProvenance

    config: MarketAnalysisConfig

    quality: DataQualityReport

    asset_statistics: list[AssetStatistics] = Field(
        default_factory=list
    )

    correlation_matrix: MarketMatrix | None = None

    covariance_matrix: MarketMatrix | None = None

    calculation_version: str = "1.0.0"

    generated_at: datetime

    downstream_ready: bool = False

    @model_validator(mode="after")
    def validate_readiness(self):
        if self.downstream_ready:
            if self.status == "rejected":
                raise ValueError(
                    "Rejected analyses cannot be "
                    "downstream ready."
                )

            if self.quality.status == QualityStatus.FAIL:
                raise ValueError(
                    "Failed data quality cannot be "
                    "downstream ready."
                )

            if (
                self.correlation_matrix is None
                or self.covariance_matrix is None
            ):
                raise ValueError(
                    "Downstream-ready results require "
                    "correlation and covariance matrices."
                )

            if not self.asset_statistics:
                raise ValueError(
                    "Downstream-ready results require "
                    "asset statistics."
                )

        return self
