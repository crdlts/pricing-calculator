from __future__ import annotations

import math
from collections.abc import Callable

import pandas as pd

from models import ModelParameters, PricingConfig, PricingResults, Tree


class BinomialInterestRateModel:

    def __init__(self, config: PricingConfig):
        config.validate()
        self.config = config
        self.params = self._build_parameters(config)

    @staticmethod
    def _build_parameters(config: PricingConfig) -> ModelParameters:
        dt = config.maturity_years / config.steps
        short_rate_decimal = config.short_rate_percent / 100.0
        up_factor = math.exp(config.sigma * math.sqrt(dt))
        down_factor = 1.0 / up_factor
        risk_neutral_p = (
            math.exp(short_rate_decimal * dt) - down_factor
        ) / (up_factor - down_factor)
        risk_neutral_q = 1.0 - risk_neutral_p

        if not 0.0 <= risk_neutral_p <= 1.0:
            raise ValueError(
                "Риск-нейтральная вероятность p вышла за пределы [0, 1]. "
                "Проверьте параметры T, sigma и r0."
            )

        return ModelParameters(
            dt=dt,
            short_rate_decimal=short_rate_decimal,
            up_factor=up_factor,
            down_factor=down_factor,
            risk_neutral_p=risk_neutral_p,
            risk_neutral_q=risk_neutral_q,
        )

    def price(self) -> PricingResults:
        short_rate_tree = self._build_short_rate_tree_percent()
        zcb_tree = self._build_zcb_tree_percent(short_rate_tree, self.config.steps)
        zcb_t_tree = self._build_zcb_tree_percent(
            short_rate_tree,
            self.config.forward_maturity_step,
        )
        futures_tree = self._build_futures_tree_percent(
            zcb_tree,
            self.config.futures_maturity_step,
        )
        option_1_tree = self._build_american_call_tree_percent(
            futures_tree,
            self.config.strike_1_percent,
        )
        option_2_tree = self._build_american_call_tree_percent(
            futures_tree,
            self.config.strike_2_percent,
        )

        zcb_price = zcb_tree[0][0]
        zcb_t_price = zcb_t_tree[0][0]
        forward_price = (zcb_price / zcb_t_price) * 100.0 if zcb_t_price else float("nan")
        futures_price = futures_tree[0][0]
        option_1_price = option_1_tree[0][0]
        option_2_price = option_2_tree[0][0]

        return PricingResults(
            config=self.config,
            params=self.params,
            short_rate_tree_percent=short_rate_tree,
            zcb_tree_percent=zcb_tree,
            zcb_t_tree_percent=zcb_t_tree,
            futures_tree_percent=futures_tree,
            option_1_tree_percent=option_1_tree,
            option_2_tree_percent=option_2_tree,
            zcb_price_percent=zcb_price,
            forward_price_percent=forward_price,
            futures_price_percent=futures_price,
            option_1_price_percent=option_1_price,
            option_2_price_percent=option_2_price,
        )

    def _build_short_rate_tree_percent(self) -> Tree:
        tree: Tree = []
        for time in range(self.config.steps + 1):
            layer = []
            for down_moves in range(time + 1):
                up_moves = time - down_moves
                rate = (
                    self.config.short_rate_percent
                    * (self.params.up_factor ** up_moves)
                    * (self.params.down_factor ** down_moves)
                )
                layer.append(rate)
            tree.append(layer)
        return tree

    def _build_zcb_tree_percent(self, short_rate_tree: Tree, maturity_step: int) -> Tree:
        terminal_values = [100.0] * (maturity_step + 1)

        def node_value(time: int, state: int, next_layer: list[float]) -> float:
            expected_value = self._expected_next_value(next_layer, state)
            return self._discount_expected_value(expected_value, short_rate_tree, time, state)

        return self._rollback_tree(maturity_step, terminal_values, node_value)

    def _build_futures_tree_percent(self, zcb_tree: Tree, maturity_step: int) -> Tree:
        # The Excel baseline rounds the ZCB_k maturity layer before the futures rollback.
        terminal_values = [round(value, 2) for value in zcb_tree[maturity_step]]

        def node_value(_: int, state: int, next_layer: list[float]) -> float:
            return self._expected_next_value(next_layer, state)

        return self._rollback_tree(maturity_step, terminal_values, node_value)

    def _build_american_call_tree_percent(self, futures_tree: Tree, strike_percent: float) -> Tree:
        maturity_step = self.config.futures_maturity_step
        discount = math.exp(
            (self.params.short_rate_decimal * self.config.maturity_years) / max(maturity_step, 1)
        )
        terminal_values = [max(0.0, future - strike_percent) for future in futures_tree[maturity_step]]

        def node_value(time: int, state: int, next_layer: list[float]) -> float:
            continuation_value = self._expected_next_value(next_layer, state) / discount
            immediate_exercise = max(0.0, futures_tree[time][state] - strike_percent)
            return max(continuation_value, immediate_exercise)

        return self._rollback_tree(maturity_step, terminal_values, node_value)

    def _discount_expected_value(
        self,
        expected_value: float,
        short_rate_tree: Tree,
        time: int,
        state: int,
    ) -> float:
        short_rate = short_rate_tree[time][state]
        return expected_value / (1.0 + short_rate / 100.0)

    def _rollback_tree(
        self,
        step_count: int,
        terminal_values: list[float],
        node_value: Callable[[int, int, list[float]], float],
    ) -> Tree:
        tree: Tree = [[] for _ in range(step_count + 1)]
        tree[step_count] = list(terminal_values)

        for time in range(step_count - 1, -1, -1):
            next_layer = tree[time + 1]
            tree[time] = [
                node_value(time, state, next_layer)
                for state in range(time + 1)
            ]

        return tree

    def _expected_next_value(self, next_layer: list[float], state: int) -> float:
        return (
            self.params.risk_neutral_p * next_layer[state]
            + self.params.risk_neutral_q * next_layer[state + 1]
        )


class ResultsFormatter:
    @staticmethod
    def to_dataframe(tree: Tree, precision: int = 4) -> pd.DataFrame:
        steps = len(tree) - 1
        rows = []

        for state in range(steps, -1, -1):
            row = []
            for time in range(steps + 1):
                if state <= time:
                    mirrored_state = time - state
                    value = tree[time][mirrored_state]
                else:
                    value = None
                row.append(round(value, precision) if value is not None else None)
            rows.append(row)

        columns = [f"t={time}" for time in range(steps + 1)]
        index = [f"state={state}" for state in range(steps, -1, -1)]
        return pd.DataFrame(rows, columns=columns, index=index)

    @staticmethod
    def to_percent_table(tree: Tree, precision: int = 2) -> pd.DataFrame:
        df = ResultsFormatter.to_dataframe(tree, precision=precision)
        return df.map(lambda value: f"{value:.{precision}f}%" if pd.notna(value) else "")

    @staticmethod
    def summary_dataframe(results: PricingResults) -> pd.DataFrame:
        return pd.DataFrame(
            [
                ("ZCB_10", results.zcb_price_percent),
                ("Цена форварда на бескупонную облигацию", results.forward_price_percent),
                ("Цена фьючерса", results.futures_price_percent),
                (
                    f"Цена американского Call опциона со страйком {results.config.strike_1_percent:.0f}%",
                    results.option_1_price_percent,
                ),
                (
                    f"Цена американского Call опциона со страйком {results.config.strike_2_percent:.0f}%",
                    results.option_2_price_percent,
                ),
            ],
            columns=["Метрика", "Значение"],
        )

    @staticmethod
    def parameters_dataframe(results: PricingResults) -> pd.DataFrame:
        params = results.params
        return pd.DataFrame(
            {
                "Параметр": ["T", "r_0", "u", "d", "p", "q"],
                "Значение": [
                    params.dt,
                    params.short_rate_decimal,
                    params.up_factor,
                    params.down_factor,
                    params.risk_neutral_p,
                    params.risk_neutral_q,
                ],
            }
        )
