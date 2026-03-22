from __future__ import annotations

from dataclasses import dataclass


Tree = list[list[float]]


@dataclass(slots=True)
class PricingConfig:
    maturity_years: float = 10.0
    forward_maturity_step: int = 7
    futures_maturity_step: int = 4
    short_rate_percent: float = 5.0
    sigma: float = 0.1
    steps: int = 10
    strike_1_percent: float = 70.0
    strike_2_percent: float = 80.0

    def validate(self) -> None:
        if self.steps <= 0:
            raise ValueError("Количество шагов n должно быть больше нуля.")
        if self.maturity_years <= 0:
            raise ValueError("Срок T должен быть больше нуля.")
        if not 0 <= self.forward_maturity_step <= self.steps:
            raise ValueError("Шаг исполнения форварда t должен быть в диапазоне от 0 до n.")
        if not 0 <= self.futures_maturity_step <= self.steps:
            raise ValueError("Шаг исполнения фьючерса и опциона k должен быть в диапазоне от 0 до n.")
        if self.sigma <= 0:
            raise ValueError("Волатильность sigma должна быть больше нуля.")
        if self.short_rate_percent < 0:
            raise ValueError("Начальная ставка r0 не может быть отрицательной.")
        if self.strike_1_percent < 0 or self.strike_2_percent < 0:
            raise ValueError("Страйки не могут быть отрицательными.")


@dataclass(slots=True)
class ModelParameters:
    dt: float
    short_rate_decimal: float
    up_factor: float
    down_factor: float
    risk_neutral_p: float
    risk_neutral_q: float


@dataclass(slots=True)
class PricingResults:
    config: PricingConfig
    params: ModelParameters
    short_rate_tree_percent: Tree
    zcb_tree_percent: Tree
    zcb_t_tree_percent: Tree
    futures_tree_percent: Tree
    option_1_tree_percent: Tree
    option_2_tree_percent: Tree
    zcb_price_percent: float
    forward_price_percent: float
    futures_price_percent: float
    option_1_price_percent: float
    option_2_price_percent: float
