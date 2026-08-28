import math
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from yfinance import EquityQuery


st.set_page_config(
    page_title="Momentum Leader Screener",
    page_icon="📈",
    layout="wide",
)

st.title("Momentum Leader Screener V1")
st.caption(
    "US-Aktien nach Trendqualität, relativer Stärke und Widerstandsfähigkeit "
    "gegenüber einem Benchmark filtern und ranken."
)

with st.expander("Was dieser Screener misst", expanded=False):
    st.markdown(
        """
        **Trend-Template (Minervini-orientiert):**
        Kurs > SMA 50/150/200, SMA 50 > SMA 150/200, SMA 150 > SMA 200,
        steigende SMA 200, mindestens 30 % über 52W-Tief, höchstens 25 % unter 52W-Hoch,
        sowie ein eigener RS-Perzentilwert ≥ 70.

        **Zusätzliche Momentum-Messung:**
        - 5-/21-/63-/126-/252-Tage-Performance
        - relative Performance zum Benchmark
        - Veränderung der RS-Linie (Aktie / Benchmark)
        - „Red-Day Hold“: Wie oft schlägt die Aktie den Benchmark an dessen Verlusttagen?
        - eigener Momentum Score von 0–100

        **Hinweis:** Das RS-Perzentil ist eine eigene, transparente Berechnung dieser App
        und **nicht** das proprietäre IBD Relative Strength Rating.
        """
    )

st.warning(
    "Datenquelle in V1: yfinance/Yahoo Finance. Geeignet für persönliche Research- und "
    "Prototyping-Zwecke; kein garantierter Echtzeit-/Institutional-Feed. "
    "Keine Anlageberatung."
)


@dataclass
class Settings:
    min_market_cap_b: float
    min_price: float
    min_avg_volume: int
    max_universe: int
    benchmark: str
    min_rs_percentile: int
    max_below_52w_high_pct: float
    min_above_52w_low_pct: float
    min_red_day_hold_pct: float
    min_week_rel_pct: float
    require_template: bool


def pct_return(s: pd.Series, n: int) -> float:
    s = s.dropna()
    if len(s) <= n:
        return np.nan
    old = float(s.iloc[-n-1])
    new = float(s.iloc[-1])
    if old == 0:
        return np.nan
    return (new / old - 1.0) * 100.0


def safe_float(x, default=np.nan):
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def discover_symbols(settings: Settings) -> Tuple[List[str], Dict[str, dict]]:
    """
    Yahoo's EquityQuery is used as a coarse first pass, so we do not need a
    user-supplied TradingView watchlist.
    """
    exchanges = ["NMS", "NYQ", "NGM", "NCM", "ASE"]
    query = EquityQuery("and", [
        EquityQuery("eq", ["region", "us"]),
        EquityQuery("is-in", ["exchange", *exchanges]),
        EquityQuery("gte", ["intradaymarketcap", settings.min_market_cap_b * 1e9]),
        EquityQuery("gte", ["intradayprice", settings.min_price]),
        EquityQuery("gte", ["avgdailyvol3m", settings.min_avg_volume]),
    ])

    symbols: List[str] = []
    meta: Dict[str, dict] = {}
    offset = 0
    batch_size = 250

    progress = st.progress(0, text="Aktienuniversum wird aufgebaut …")

    while len(symbols) < settings.max_universe:
        size = min(batch_size, settings.max_universe - len(symbols))
        try:
            response = yf.screen(
                query,
                offset=offset,
                size=size,
                sortField="intradaymarketcap",
                sortAsc=False,
            )
        except Exception as exc:
            raise RuntimeError(f"Yahoo-Screener konnte nicht geladen werden: {exc}") from exc

        quotes = response.get("quotes", []) if isinstance(response, dict) else []
        if not quotes:
            break

        for q in quotes:
            symbol = q.get("symbol")
            if not symbol:
                continue
            quote_type = str(q.get("quoteType", "")).upper()
            if quote_type and quote_type != "EQUITY":
                continue
            if symbol not in meta:
                symbols.append(symbol)
                meta[symbol] = q

        offset += len(quotes)
        progress.progress(
            min(len(symbols) / max(settings.max_universe, 1), 1.0),
            text=f"{len(symbols)} Aktien im Ausgangsuniversum …",
        )
        if len(quotes) < size:
            break

    progress.empty()
    return symbols[: settings.max_universe], meta


def get_field_frame(raw: pd.DataFrame, field: str, tickers: List[str]) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame()

    if isinstance(raw.columns, pd.MultiIndex):
        # group_by="ticker" gives columns (ticker, field)
        out = {}
        for t in tickers:
            if t in raw.columns.get_level_values(0):
                sub = raw[t]
                if field in sub.columns:
                    out[t] = sub[field]
        return pd.DataFrame(out)

    # Single ticker
    if field in raw.columns and len(tickers) == 1:
        return pd.DataFrame({tickers[0]: raw[field]})
    return pd.DataFrame()


@st.cache_data(ttl=1800, show_spinner=False)
def download_prices(tickers_tuple: tuple, benchmark: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    tickers = list(tickers_tuple)
    all_tickers = list(dict.fromkeys(tickers + [benchmark]))

    close_parts = []
    volume_parts = []
    batch_size = 100

    for i in range(0, len(all_tickers), batch_size):
        batch = all_tickers[i:i + batch_size]
        raw = yf.download(
            batch,
            period="2y",
            interval="1d",
            auto_adjust=True,
            group_by="ticker",
            threads=True,
            progress=False,
            timeout=20,
        )
        close_parts.append(get_field_frame(raw, "Close", batch))
        volume_parts.append(get_field_frame(raw, "Volume", batch))

    close = pd.concat(close_parts, axis=1)
    volume = pd.concat(volume_parts, axis=1)

    close = close.loc[:, ~close.columns.duplicated()].sort_index()
    volume = volume.loc[:, ~volume.columns.duplicated()].sort_index()
    return close, volume


def calc_metrics(
    symbol: str,
    stock: pd.Series,
    bench: pd.Series,
    volume: pd.Series,
    meta: dict,
) -> dict | None:
    pair = pd.concat(
        [stock.rename("stock"), bench.rename("bench")], axis=1
    ).dropna()

    if len(pair) < 260:
        return None

    s = pair["stock"]
    b = pair["bench"]
    current = float(s.iloc[-1])

    sma50 = float(s.rolling(50).mean().iloc[-1])
    sma150 = float(s.rolling(150).mean().iloc[-1])
    sma200_series = s.rolling(200).mean()
    sma200 = float(sma200_series.iloc[-1])
    sma200_21ago = float(sma200_series.iloc[-22])

    window_52 = s.iloc[-252:]
    high52 = float(window_52.max())
    low52 = float(window_52.min())

    above_low = (current / low52 - 1.0) * 100.0 if low52 else np.nan
    below_high = (current / high52 - 1.0) * 100.0 if high52 else np.nan

    r5 = pct_return(s, 5)
    r21 = pct_return(s, 21)
    r63 = pct_return(s, 63)
    r126 = pct_return(s, 126)
    r189 = pct_return(s, 189)
    r252 = pct_return(s, 252)

    br5 = pct_return(b, 5)
    br21 = pct_return(b, 21)
    br63 = pct_return(b, 63)
    br126 = pct_return(b, 126)
    br252 = pct_return(b, 252)

    rel5 = r5 - br5
    rel21 = r21 - br21
    rel63 = r63 - br63
    rel126 = r126 - br126
    rel252 = r252 - br252

    rs_line = (s / b).dropna()
    rs_line_63 = pct_return(rs_line, 63)

    returns = pair.pct_change().dropna().iloc[-60:]
    red = returns["bench"] < 0
    red_n = int(red.sum())
    if red_n:
        red_hold = float(
            (returns.loc[red, "stock"] > returns.loc[red, "bench"]).mean() * 100.0
        )
        down_alpha = float(
            ((returns.loc[red, "stock"] - returns.loc[red, "bench"]).mean()) * 100.0
        )
    else:
        red_hold = np.nan
        down_alpha = np.nan

    avg_vol_50 = safe_float(volume.dropna().iloc[-50:].mean()) if volume is not None else np.nan

    # Cross-sectional RS percentile is assigned later.
    rs_composite = (
        0.40 * r63
        + 0.20 * r126
        + 0.20 * r189
        + 0.20 * r252
    )

    rules_without_rs = {
        "Price>SMA50": current > sma50,
        "Price>SMA150": current > sma150,
        "Price>SMA200": current > sma200,
        "SMA50>SMA150": sma50 > sma150,
        "SMA50>SMA200": sma50 > sma200,
        "SMA150>SMA200": sma150 > sma200,
        "SMA200 rising": sma200 > sma200_21ago,
        ">=30% above 52W low": above_low >= 30.0,
        "<=25% below 52W high": below_high >= -25.0,
    }

    market_cap = (
        meta.get("marketCap")
        or meta.get("intradaymarketcap")
        or meta.get("lastclosemarketcap.lasttwelvemonths")
    )
    market_cap_b = safe_float(market_cap) / 1e9 if market_cap else np.nan

    return {
        "Ticker": symbol,
        "Name": meta.get("shortName") or meta.get("longName") or "",
        "Exchange": meta.get("exchange") or "",
        "Sector": meta.get("sector") or "",
        "Price": current,
        "Market Cap ($B)": market_cap_b,
        "Avg Vol 50D": avg_vol_50,
        "5D %": r5,
        "21D %": r21,
        "63D %": r63,
        "126D %": r126,
        "252D %": r252,
        "5D vs Bench %": rel5,
        "21D vs Bench %": rel21,
        "63D vs Bench %": rel63,
        "126D vs Bench %": rel126,
        "252D vs Bench %": rel252,
        "RS line 63D %": rs_line_63,
        "Red-Day Hold %": red_hold,
        "Down-day alpha avg %": down_alpha,
        "% above 52W low": above_low,
        "% below 52W high": below_high,
        "SMA50": sma50,
        "SMA150": sma150,
        "SMA200": sma200,
        "RS composite": rs_composite,
        "_rules": rules_without_rs,
    }


def add_cross_sectional_scores(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if df.empty:
        return df

    # Transparent 1-99 percentile inside the scanned universe.
    pct = df["RS composite"].rank(method="average", pct=True)
    df["RS percentile"] = np.ceil(pct * 99).clip(1, 99)

    def row_score(row):
        rules = row["_rules"]
        trend_points = (sum(bool(v) for v in rules.values()) / 9.0) * 35.0
        rs_points = (safe_float(row["RS percentile"], 1) / 99.0) * 30.0

        # 5D relative momentum: -5% -> 0, +10% -> 10
        rel5 = safe_float(row["5D vs Bench %"], -5)
        weekly_points = np.clip((rel5 + 5.0) / 15.0, 0, 1) * 10.0

        hold = safe_float(row["Red-Day Hold %"], 0)
        hold_points = np.clip(hold / 100.0, 0, 1) * 15.0

        rs63 = safe_float(row["RS line 63D %"], -10)
        rsline_points = np.clip((rs63 + 10.0) / 40.0, 0, 1) * 10.0

        return round(float(trend_points + rs_points + weekly_points + hold_points + rsline_points), 1)

    df["Momentum Score"] = df.apply(row_score, axis=1)

    def template_pass(row):
        return (
            row["RS percentile"] >= 70
            and all(bool(v) for v in row["_rules"].values())
        )

    df["Minervini Template"] = df.apply(template_pass, axis=1)
    return df


def apply_filters(df: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    out = df.copy()

    out = out[out["RS percentile"] >= settings.min_rs_percentile]
    out = out[out["% below 52W high"] >= -settings.max_below_52w_high_pct]
    out = out[out["% above 52W low"] >= settings.min_above_52w_low_pct]
    out = out[out["Red-Day Hold %"] >= settings.min_red_day_hold_pct]
    out = out[out["5D vs Bench %"] >= settings.min_week_rel_pct]

    if settings.require_template:
        out = out[out["Minervini Template"]]

    return out.sort_values(
        ["Momentum Score", "RS percentile", "5D vs Bench %"],
        ascending=[False, False, False],
    )


with st.sidebar:
    st.header("Filter")

    benchmark_label = st.selectbox(
        "Benchmark",
        ["QQQ (Nasdaq-100 ETF)", "^IXIC (Nasdaq Composite)", "SPY (S&P 500 ETF)"],
        index=0,
    )
    benchmark = benchmark_label.split(" ")[0]

    min_market_cap_b = st.number_input(
        "Min. Marktkapitalisierung ($ Mrd.)",
        min_value=0.1,
        max_value=1000.0,
        value=2.0,
        step=0.5,
    )
    min_price = st.number_input(
        "Min. Aktienkurs ($)",
        min_value=1.0,
        max_value=1000.0,
        value=10.0,
        step=1.0,
    )
    min_avg_volume = st.number_input(
        "Min. Ø Tagesvolumen (3M)",
        min_value=10_000,
        max_value=100_000_000,
        value=500_000,
        step=100_000,
    )
    max_universe = st.slider(
        "Max. Aktien im Ausgangsuniversum",
        min_value=100,
        max_value=4000,
        value=1000,
        step=100,
    )

    st.divider()
    st.subheader("Momentum")

    min_rs_percentile = st.slider(
        "Min. eigenes RS-Perzentil",
        min_value=1,
        max_value=99,
        value=80,
    )
    max_below_52w_high_pct = st.slider(
        "Max. Abstand zum 52W-Hoch (%)",
        min_value=0,
        max_value=50,
        value=15,
    )
    min_above_52w_low_pct = st.slider(
        "Min. Abstand über 52W-Tief (%)",
        min_value=0,
        max_value=200,
        value=30,
    )
    min_red_day_hold_pct = st.slider(
        "Min. Red-Day Hold (%)",
        min_value=0,
        max_value=100,
        value=60,
        help="Anteil der Benchmark-Verlusttage der letzten 60 Sitzungen, "
             "an denen die Aktie besser lief als der Benchmark.",
    )
    min_week_rel_pct = st.slider(
        "Min. 5T-Performance ggü. Benchmark (%)",
        min_value=-10.0,
        max_value=20.0,
        value=0.0,
        step=0.5,
    )
    require_template = st.checkbox(
        "Nur vollständiges Minervini-Trend-Template",
        value=False,
    )

    scan = st.button("🚀 Markt jetzt scannen", type="primary", use_container_width=True)

settings = Settings(
    min_market_cap_b=float(min_market_cap_b),
    min_price=float(min_price),
    min_avg_volume=int(min_avg_volume),
    max_universe=int(max_universe),
    benchmark=benchmark,
    min_rs_percentile=int(min_rs_percentile),
    max_below_52w_high_pct=float(max_below_52w_high_pct),
    min_above_52w_low_pct=float(min_above_52w_low_pct),
    min_red_day_hold_pct=float(min_red_day_hold_pct),
    min_week_rel_pct=float(min_week_rel_pct),
    require_template=bool(require_template),
)

if scan:
    try:
        symbols, metadata = discover_symbols(settings)
        if not symbols:
            st.error("Kein Ausgangsuniversum gefunden. Filter lockern oder später erneut versuchen.")
            st.stop()

        st.info(f"{len(symbols)} Aktien gefunden. Lade Kursdaten und berechne Momentum …")

        close, volume = download_prices(tuple(symbols), settings.benchmark)
        if settings.benchmark not in close.columns:
            st.error(f"Benchmark {settings.benchmark} konnte nicht geladen werden.")
            st.stop()

        bench = close[settings.benchmark].dropna()
        rows = []
        bar = st.progress(0, text="Momentum-Kennzahlen werden berechnet …")

        for i, symbol in enumerate(symbols, start=1):
            if symbol not in close.columns:
                continue
            vol = volume[symbol] if symbol in volume.columns else pd.Series(dtype=float)
            metrics = calc_metrics(
                symbol,
                close[symbol],
                bench,
                vol,
                metadata.get(symbol, {}),
            )
            if metrics is not None:
                rows.append(metrics)

            if i % 10 == 0 or i == len(symbols):
                bar.progress(i / len(symbols), text=f"{i}/{len(symbols)} analysiert …")

        bar.empty()

        full = add_cross_sectional_scores(pd.DataFrame(rows))
        filtered = apply_filters(full, settings)

        st.session_state["momentum_full"] = full
        st.session_state["momentum_filtered"] = filtered
        st.session_state["momentum_settings"] = settings.__dict__
        st.session_state["momentum_close"] = close
        st.success(
            f"Scan abgeschlossen: {len(filtered)} Kandidaten aus "
            f"{len(full)} auswertbaren Aktien."
        )

    except Exception as exc:
        st.exception(exc)

if "momentum_filtered" in st.session_state:
    filtered = st.session_state["momentum_filtered"]
    full = st.session_state["momentum_full"]
    saved_settings = st.session_state["momentum_settings"]
    close = st.session_state.get("momentum_close", pd.DataFrame())
    bench = saved_settings["benchmark"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Kandidaten", len(filtered))
    c2.metric("Auswertbares Universum", len(full))
    c3.metric(
        "Vollständiges Template",
        int(full["Minervini Template"].sum()) if not full.empty else 0,
    )
    c4.metric("Benchmark", bench)

    if filtered.empty:
        st.warning("Aktuell erfüllt keine Aktie alle gewählten Filter.")
    else:
        show = filtered.drop(columns=["_rules", "RS composite"], errors="ignore").copy()

        numeric_cols = [
            "Price", "Market Cap ($B)", "Avg Vol 50D", "5D %", "21D %",
            "63D %", "126D %", "252D %", "5D vs Bench %", "21D vs Bench %",
            "63D vs Bench %", "126D vs Bench %", "252D vs Bench %",
            "RS line 63D %", "Red-Day Hold %", "Down-day alpha avg %",
            "% above 52W low", "% below 52W high", "SMA50", "SMA150",
            "SMA200", "RS percentile", "Momentum Score"
        ]
        for c in numeric_cols:
            if c in show.columns:
                show[c] = pd.to_numeric(show[c], errors="coerce").round(2)

        preferred = [
            "Ticker", "Name", "Momentum Score", "RS percentile",
            "Minervini Template", "5D vs Bench %", "21D vs Bench %",
            "Red-Day Hold %", "RS line 63D %", "% below 52W high",
            "Market Cap ($B)", "Price", "Sector", "Exchange",
            "5D %", "21D %", "63D %", "126D %", "252D %",
        ]
        preferred = [c for c in preferred if c in show.columns]
        rest = [c for c in show.columns if c not in preferred]
        show = show[preferred + rest]

        st.subheader("Ranking")
        st.dataframe(
            show,
            use_container_width=True,
            hide_index=True,
            height=600,
        )

        st.download_button(
            "CSV der Treffer herunterladen",
            data=show.to_csv(index=False).encode("utf-8-sig"),
            file_name="momentum_screener_results.csv",
            mime="text/csv",
        )

        st.subheader("Detailansicht")
        selected = st.selectbox("Aktie auswählen", filtered["Ticker"].tolist())
        row = filtered.loc[filtered["Ticker"] == selected].iloc[0]

        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Momentum Score", f'{row["Momentum Score"]:.1f}/100')
        d2.metric("RS-Perzentil", f'{row["RS percentile"]:.0f}/99')
        d3.metric("5T vs Benchmark", f'{row["5D vs Bench %"]:+.2f}%')
        d4.metric("Red-Day Hold", f'{row["Red-Day Hold %"]:.1f}%')

        if selected in close.columns:
            s = close[selected].dropna().iloc[-300:]
            chart_df = pd.DataFrame({"Close": s})
            chart_df["SMA50"] = chart_df["Close"].rolling(50).mean()
            chart_df["SMA150"] = chart_df["Close"].rolling(150).mean()
            chart_df["SMA200"] = chart_df["Close"].rolling(200).mean()

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df["Close"], name="Close"))
            fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df["SMA50"], name="SMA50"))
            fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df["SMA150"], name="SMA150"))
            fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df["SMA200"], name="SMA200"))
            fig.update_layout(
                height=520,
                margin=dict(l=10, r=10, t=30, b=10),
                legend_orientation="h",
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Trend-Template-Check**")
        rules = row["_rules"]
        checks = pd.DataFrame({
            "Kriterium": list(rules.keys()) + ["RS percentile >= 70"],
            "Erfüllt": list(rules.values()) + [row["RS percentile"] >= 70],
        })
        st.dataframe(checks, hide_index=True, use_container_width=True)

        tv_symbol = selected.replace(".", "-")
        st.link_button(
            f"{selected} in TradingView öffnen",
            f"https://www.tradingview.com/chart/?symbol={tv_symbol}",
        )

        st.caption(
            "Momentum ist ein Ranking- und Research-Signal, keine Aussage darüber, "
            "dass eine Aktie künftig steigen wird. Kurse und Fundamentaldaten können "
            "verzögert, unvollständig oder fehlerhaft sein."
        )
else:
    st.info(
        "Stelle links deine Kriterien ein und drücke **„Markt jetzt scannen“**. "
        "Es ist keine TradingView-Watchlist erforderlich."
    )
