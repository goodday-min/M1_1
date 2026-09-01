"""
SPY · QQQ · QLD 장기 수익률 비교 대시보드 (보너스 A)

실행:
    pip install streamlit
    streamlit run dashboard.py

사전 조건: data/fetch_data.py를 먼저 실행해 data/raw/*.csv, data/common_period.csv가
생성되어 있어야 합니다.
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import streamlit as st

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(THIS_DIR, "src"))
import metrics as m  # noqa: E402

DATA_DIR = os.path.join(THIS_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
TICKERS = ["SPY", "QQQ", "QLD"]

# ---------------------------------------------------------------------------
# 한글 폰트 설정 (matplotlib) — 없으면 경고만 띄우고 계속 진행
# ---------------------------------------------------------------------------
_KOREAN_FONT_CANDIDATES = [
    "AppleGothic", "Malgun Gothic", "NanumGothic", "NanumBarunGothic",
    "Noto Sans CJK KR", "Noto Sans KR", "Noto Sans CJK JP", "Noto Sans CJK SC",
]
_available = {f.name for f in fm.fontManager.ttflist}
_chosen = next((f for f in _KOREAN_FONT_CANDIDATES if f in _available), None)
if _chosen:
    plt.rcParams["font.family"] = _chosen
plt.rcParams["axes.unicode_minus"] = False


# ---------------------------------------------------------------------------
# 데이터 로드
# ---------------------------------------------------------------------------
@st.cache_data
def load_common_period() -> pd.DataFrame:
    """공통구간(2006-06-21~) 정렬 데이터. 결측치는 forward-fill."""
    path = os.path.join(DATA_DIR, "common_period.csv")
    df = pd.read_csv(path, parse_dates=["Date"]).set_index("Date").sort_index()
    return df.ffill()


@st.cache_data
def load_individual_max() -> pd.DataFrame:
    """각 티커의 상장일부터 오늘까지 전체 역사. 상장 전 구간은 NaN으로 남김."""
    frames = {}
    for t in TICKERS:
        path = os.path.join(RAW_DIR, f"{t.lower()}.csv")
        df = pd.read_csv(path, parse_dates=["Date"]).set_index("Date").sort_index()
        frames[t] = df["Adj Close"]
    return pd.DataFrame(frames)


# ---------------------------------------------------------------------------
# 페이지 구성
# ---------------------------------------------------------------------------
st.set_page_config(page_title="SPY·QQQ·QLD 대시보드", layout="wide")
st.title("SPY · QQQ · QLD 장기 수익률 비교 대시보드")
st.caption("데이터 출처: Yahoo Finance (yfinance). 가격 기준: Adj Close (배당 재투자·액면분할 반영).")

with st.sidebar:
    st.header("설정")
    selected_tickers = st.multiselect("비교할 티커", TICKERS, default=TICKERS)
    period_mode = st.radio(
        "기간 모드",
        ["공통구간 (2006-06-21~)", "개별 최대구간 (상장일~)"],
        help="공통구간은 3개 상품을 동일 조건에서 비교, 개별 최대구간은 각 상품의 전체 역사를 봅니다.",
    )

if not selected_tickers:
    st.warning("왼쪽에서 최소 1개 이상의 티커를 선택해주세요.")
    st.stop()

if period_mode.startswith("공통구간"):
    data = load_common_period()
else:
    data = load_individual_max()

data_min = data.index.min().date()
data_max = data.index.max().date()

with st.sidebar:
    date_range = st.slider(
        "날짜 범위",
        min_value=data_min,
        max_value=data_max,
        value=(data_min, data_max),
        format="YYYY-MM-DD",
    )

date_index = pd.DatetimeIndex(data.index)
mask = (date_index.date >= date_range[0]) & (date_index.date <= date_range[1])
sub = data.loc[mask, selected_tickers]

# ---------------------------------------------------------------------------
# 통계 계산 (CAGR / MDD / 연율화 변동성)
# ---------------------------------------------------------------------------
stat_rows = []
skipped = []
for t in selected_tickers:
    series = sub[t].dropna()
    if len(series) < 2:
        skipped.append(t)
        continue
    mdd_stats = m.max_drawdown_stats(series)
    stat_rows.append(
        {
            "티커": t,
            "데이터 시작": series.index.min().date(),
            "데이터 끝": series.index.max().date(),
            "누적수익률(%)": m.cumulative_return(series),
            "CAGR(%)": m.cagr(series) * 100,
            "연율화 변동성(%)": m.annualized_volatility(series),
            "MDD(%)": mdd_stats["mdd_pct"],
        }
    )

if skipped:
    st.info(f"선택한 구간에 데이터가 부족해 제외됨: {', '.join(skipped)} "
             f"(개별 최대구간 모드에서 상장 전 구간을 선택하면 발생할 수 있습니다)")

if not stat_rows:
    st.warning("선택한 구간에 표시할 데이터가 없습니다. 날짜 범위를 조정해주세요.")
    st.stop()

stats_df = pd.DataFrame(stat_rows).set_index("티커")

st.subheader("선택 구간 통계")
st.dataframe(
    stats_df.style.format(
        {
            "누적수익률(%)": "{:.2f}",
            "CAGR(%)": "{:.2f}",
            "연율화 변동성(%)": "{:.2f}",
            "MDD(%)": "{:.2f}",
        }
    ),
    use_container_width=True,
)

# 요약 지표 카드
cols = st.columns(len(stats_df))
for col, (ticker, row) in zip(cols, stats_df.iterrows()):
    col.metric(
        label=ticker,
        value=f"CAGR {row['CAGR(%)']:.1f}%",
        delta=f"MDD {row['MDD(%)']:.1f}%",
        delta_color="inverse",
    )

# ---------------------------------------------------------------------------
# 정규화 누적수익률 차트
# ---------------------------------------------------------------------------
st.subheader("정규화 누적수익률 (구간 시작=100, 로그스케일)")
fig, ax = plt.subplots(figsize=(11, 5))
for t in stats_df.index:
    series = sub[t].dropna()
    idx_series = m.normalize_to_100(series)
    ax.plot(idx_series.index, idx_series.values, label=t)
ax.set_yscale("log")
ax.set_ylabel("Index (log scale, 시작=100)")
ax.legend()
ax.grid(alpha=0.3)
st.pyplot(fig)

# ---------------------------------------------------------------------------
# 낙폭(Drawdown) 차트
# ---------------------------------------------------------------------------
st.subheader("낙폭(Drawdown) Underwater 차트")
fig2, ax2 = plt.subplots(figsize=(11, 4))
for t in stats_df.index:
    series = sub[t].dropna()
    dd = m.drawdown_curve(series)
    ax2.plot(dd.index, dd.values, label=t)
ax2.set_ylabel("Drawdown (%)")
ax2.legend()
ax2.grid(alpha=0.3)
st.pyplot(fig2)

with st.expander("참고: QLD vs QQQ×2 이론값 괴리 (QQQ, QLD 둘 다 선택 시에만 표시)"):
    if "QQQ" in stats_df.index and "QLD" in stats_df.index:
        gap_df = m.leveraged_gap(sub["QQQ"].dropna(), sub["QLD"].dropna(), leverage=2.0)
        st.line_chart(gap_df["gap_pct"])
        st.caption(f"선택 구간 최종 괴리율: {gap_df['gap_pct'].iloc[-1]:.2f}%")
    else:
        st.write("QQQ와 QLD를 모두 선택하면 여기에 변동성 드래그 괴리 차트가 표시됩니다.")

st.divider()
st.caption(
    "이 대시보드는 교육/학습 목적의 분석 도구이며 투자 조언이 아닙니다. "
    "과거 수익률이 미래 수익률을 보장하지 않습니다."
)
