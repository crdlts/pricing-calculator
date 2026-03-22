from __future__ import annotations

import streamlit as st

from engine import BinomialInterestRateModel, ResultsFormatter
from models import PricingConfig, PricingResults


def read_config() -> tuple[bool, PricingConfig]:
    with st.sidebar:
        st.header("Параметры модели")

        maturity_years = st.number_input("T", min_value=0.1, value=10.0, step=0.5)
        steps = st.number_input("n", min_value=1, value=10, step=1)
        short_rate_percent = st.number_input("r_0", min_value=0.0, value=5.0, step=0.5)
        sigma = st.number_input("sigma", min_value=0.0001, value=0.1, step=0.01, format="%.4f")
        forward_step = st.number_input("t", min_value=0, value=7, step=1)
        futures_step = st.number_input("k", min_value=0, value=4, step=1)
        strike_1 = st.number_input("E_1", min_value=0.0, value=70.0, step=1.0)
        strike_2 = st.number_input("E_2", min_value=0.0, value=80.0, step=1.0)
        run = st.button("Рассчитать", use_container_width=True)

    config = PricingConfig(
        maturity_years=float(maturity_years),
        forward_maturity_step=int(forward_step),
        futures_maturity_step=int(futures_step),
        short_rate_percent=float(short_rate_percent),
        sigma=float(sigma),
        steps=int(steps),
        strike_1_percent=float(strike_1),
        strike_2_percent=float(strike_2),
    )
    return run, config


def render_metrics(results: PricingResults) -> None:
    col_1, col_2, col_3, col_4, col_5 = st.columns(5)
    col_1.metric("ZCB", f"{results.zcb_price_percent:.2f}%")
    col_2.metric("Форвард", f"{results.forward_price_percent:.2f}%")
    col_3.metric("Фьючерс", f"{results.futures_price_percent:.2f}%")
    col_4.metric(
        f"Call {results.config.strike_1_percent:.0f}%",
        f"{results.option_1_price_percent:.2f}%",
    )
    col_5.metric(
        f"Call {results.config.strike_2_percent:.0f}%",
        f"{results.option_2_price_percent:.2f}%",
    )


def render_tables(results: PricingResults) -> None:
    st.subheader("Параметры биномиальной модели")
    st.dataframe(ResultsFormatter.parameters_dataframe(results), use_container_width=True)

    st.subheader("Результаты расчёта")
    st.dataframe(ResultsFormatter.summary_dataframe(results), use_container_width=True)

    tabs = st.tabs(
        [
            "Биномиальная модель процентной ставки",
            "Матрица ZCB_10",
            f"Дерево ZCB_{results.config.forward_maturity_step}",
            f"Цена фьючерса с периодом экспирации k={results.config.futures_maturity_step}",
            f"Call {results.config.strike_1_percent:.0f}%",
            f"Call {results.config.strike_2_percent:.0f}%",
        ]
    )

    with tabs[0]:
        st.dataframe(
            ResultsFormatter.to_percent_table(results.short_rate_tree_percent),
            use_container_width=True,
        )
    with tabs[1]:
        st.dataframe(
            ResultsFormatter.to_percent_table(results.zcb_tree_percent),
            use_container_width=True,
        )
    with tabs[2]:
        st.dataframe(
            ResultsFormatter.to_percent_table(results.zcb_t_tree_percent),
            use_container_width=True,
        )
    with tabs[3]:
        st.dataframe(
            ResultsFormatter.to_percent_table(results.futures_tree_percent),
            use_container_width=True,
        )
    with tabs[4]:
        st.dataframe(
            ResultsFormatter.to_percent_table(results.option_1_tree_percent),
            use_container_width=True,
        )
    with tabs[5]:
        st.dataframe(
            ResultsFormatter.to_percent_table(results.option_2_tree_percent),
            use_container_width=True,
        )


def main() -> None:
    st.title("Калькулятор цен облигации, форварда, фьючерса и опциона")
    st.caption(
        "Приложение считает цену бескупонной облигации, форварда, фьючерса и американского опциона "
        "в биномиальной модели процентных ставок."
    )

    run, config = read_config()
    if not run:
        st.info("Задайте параметры слева.")
        return

    try:
        results = BinomialInterestRateModel(config).price()
    except Exception as exc:
        st.error(f"Ошибка расчёта: {exc}")
        return

    render_metrics(results)
    render_tables(results)


if __name__ == "__main__":
    main()
