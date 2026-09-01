"""analysis.ipynb를 생성하는 빌더 스크립트 (nbformat 라이브러리 없이 직접 JSON 구성).

실행: python3 build_notebook.py
결과: notebooks/analysis.ipynb
"""

import json
import os

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_PATH = os.path.join(THIS_DIR, "notebooks", "analysis.ipynb")


def md(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text.splitlines(keepends=True),
    }


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


cells = []

cells.append(md(
"""# SPY · QQQ · QLD 장기 수익률 비교 분석

PRD: `claude/PRD.md` 참고 (S&P500 / 나스닥100 / 나스닥100 2배 레버리지 ETF 비교)

**핵심 질문**
1. 공통구간(2006~2026) 누적수익률·CAGR 비교
2. QLD 실제 수익률 vs QQQ×2 이론값 괴리 (변동성 드래그)
3. 2008/2020/2022 위기 구간 MDD·회복기간 비교
4. QQQ vs SPY 수익률 격차의 시기별 변화
5. 변동성 대비 수익률 — 레버리지가 항상 유리한가?
6. *(추가, 주제 4)* 적립식(매월 정액) vs 거치식(일시) 투자 — "언제 사느냐가 얼마나 중요한가?"

**주의**: 이 노트북은 `data/fetch_data.py`를 먼저 실행해 `data/raw/*.csv`와
`data/common_period.csv`가 생성된 이후에 실행해야 합니다."""
))

cells.append(code(
"""import sys, os
sys.path.append(os.path.join(os.getcwd(), "..", "src"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.dates as mdates
import matplotlib.ticker as mticker

import metrics as m

pd.set_option("display.float_format", lambda x: f"{x:,.4f}")
plt.rcParams["figure.figsize"] = (11, 5)
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3

# 한글 폰트 설정 (그래프 제목/축 라벨이 한글이므로 미설정 시 글자가 깨져 보임).
# OS별로 흔히 설치된 폰트 후보를 순서대로 시도하고, 없으면 시스템 기본값을 유지한다.
_KOREAN_FONT_CANDIDATES = [
    "AppleGothic", "Malgun Gothic", "NanumGothic", "NanumBarunGothic",
    "Noto Sans CJK KR", "Noto Sans KR", "Noto Sans CJK JP", "Noto Sans CJK SC",
]
_available = {f.name for f in fm.fontManager.ttflist}
_chosen = next((f for f in _KOREAN_FONT_CANDIDATES if f in _available), None)
if _chosen:
    plt.rcParams["font.family"] = _chosen
else:
    print("[경고] 한글 폰트를 찾지 못했습니다. 그래프의 한글 라벨이 깨져 보일 수 있습니다.")
    print("       (예: `sudo apt-get install fonts-nanum` 등으로 나눔고딕 설치 권장)")
plt.rcParams["axes.unicode_minus"] = False

DATA_DIR = os.path.join("..", "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
FIG_DIR = os.path.join("..", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

TICKERS = ["SPY", "QQQ", "QLD"]
COMMON_START = "2006-06-21"
CRISIS_PERIODS = {
    "2008 금융위기": ("2007-10-01", "2009-03-31"),
    "2020 코로나": ("2020-02-01", "2020-04-30"),
    "2022 금리인상": ("2022-01-01", "2022-12-31"),
}"""
))

cells.append(md("## 1. 데이터 수집\n\n`yfinance`로 수집한 개별 티커 전체 역사 데이터와, 공통구간(2006-06-21~)으로 정렬된 데이터를 불러온다."))

cells.append(code(
"""raw = {}
for t in TICKERS:
    path = os.path.join(RAW_DIR, f"{t.lower()}.csv")
    df = pd.read_csv(path, parse_dates=["Date"]).set_index("Date")
    raw[t] = df
    print(f"{t}: {df.index.min().date()} ~ {df.index.max().date()}  ({len(df)}행)")

common = pd.read_csv(os.path.join(DATA_DIR, "common_period.csv"), parse_dates=["Date"]).set_index("Date")
common = common.sort_index()
print("\\n공통구간:", common.index.min().date(), "~", common.index.max().date(), f"({len(common)}행)")
common.head()"""
))

cells.append(md("## 2. 데이터 기본 정보 확인\n\n기간, 컬럼, 결측치 여부를 확인한다."))

cells.append(code(
"""print("[공통구간 결측치 개수]")
print(common.isna().sum())
print()
print("[공통구간 기본 통계]")
common.describe()"""
))

cells.append(md(
"""## 3. 데이터 정제

- **결측치**: 거래일 중 특정 티커만 값이 없는 경우 직전 값으로 forward-fill (상장 전 자연 결측은 공통구간 사용으로 이미 배제됨).
- **이상치**: 일일 변화율이 ±3표준편차를 초과하는 날은 삭제하지 않고 플래그만 남겨 실제 사건(위기 국면)과 연결해 해석한다."""
))

cells.append(code(
"""common_clean = common.ffill()
n_filled = (common.isna() & ~common_clean.isna()).sum()
print("forward-fill로 채운 값 개수:\\n", n_filled)

daily_returns = common_clean.pct_change()
outlier_flags = pd.DataFrame(index=common_clean.index)
for t in TICKERS:
    ret = daily_returns[t]
    thresh = 3 * ret.std()
    outlier_flags[t] = ret.abs() > thresh

print("\\n±3표준편차 초과 이상치(플래그) 개수:")
print(outlier_flags.sum())
print("\\n이상치 발생일 예시 (QQQ 기준 상위 10개):")
outlier_dates = daily_returns.loc[outlier_flags["QQQ"], "QQQ"].sort_values().index
outlier_dates[:10]"""
))

cells.append(md("## 4. 시계열 기법 적용"))

cells.append(md("### 4-1. 정규화 누적수익률 (로그스케일 인덱싱)\n\n공통구간 시작일을 100으로 맞춰 SPY/QQQ/QLD를 동일 조건에서 비교한다."))

cells.append(code(
"""fig, ax = plt.subplots()
summary_rows = []
for t in TICKERS:
    idx_series = m.normalize_to_100(common_clean[t])
    ax.plot(idx_series.index, idx_series.values, label=t)
    summary_rows.append({
        "ticker": t,
        "cumulative_return_%": m.cumulative_return(common_clean[t]),
        "CAGR_%": m.cagr(common_clean[t]) * 100,
        "annualized_vol_%": m.annualized_volatility(common_clean[t]),
    })

ax.set_yscale("log")
ax.set_title(f"공통구간({COMMON_START}~) 정규화 누적수익률 (로그스케일, 시작=100)")
ax.set_ylabel("Index (log scale)")
ax.legend()
plt.savefig(os.path.join(FIG_DIR, "01_normalized_cumulative_return.png"), dpi=120, bbox_inches="tight")
plt.show()

summary_df = pd.DataFrame(summary_rows).set_index("ticker")
summary_df"""
))

cells.append(md("### 4-2. 이동평균선 (50일/200일) — 골든크로스·데드크로스"))

cells.append(code(
"""target = "QQQ"
ma_df = m.moving_averages(common_clean[target], windows=(50, 200))
crossed = m.golden_dead_cross(ma_df)

fig, ax = plt.subplots()
ax.plot(ma_df.index, ma_df["price"], label=f"{target} 종가", alpha=0.6)
ax.plot(ma_df.index, ma_df["MA50"], label="MA50")
ax.plot(ma_df.index, ma_df["MA200"], label="MA200")
gc = crossed[crossed["golden_cross"]]
dc = crossed[crossed["dead_cross"]]
ax.scatter(gc.index, gc["price"], marker="^", color="green", s=60, label="골든크로스", zorder=5)
ax.scatter(dc.index, dc["price"], marker="v", color="red", s=60, label="데드크로스", zorder=5)
ax.set_title(f"{target} 이동평균(50/200일) 및 골든/데드크로스")
ax.legend()
plt.savefig(os.path.join(FIG_DIR, "02_moving_average_cross.png"), dpi=120, bbox_inches="tight")
plt.show()

print(f"골든크로스 {crossed['golden_cross'].sum()}회, 데드크로스 {crossed['dead_cross'].sum()}회")"""
))

cells.append(md("### 4-3. 낙폭(Drawdown) Underwater 차트 & 최대낙폭(MDD)"))

cells.append(code(
"""fig, ax = plt.subplots()
mdd_rows = []
for t in TICKERS:
    dd = m.drawdown_curve(common_clean[t])
    ax.plot(dd.index, dd.values, label=t)
    stats = m.max_drawdown_stats(common_clean[t])
    stats["ticker"] = t
    mdd_rows.append(stats)

for label, (start, end) in CRISIS_PERIODS.items():
    ax.axvspan(pd.Timestamp(start), pd.Timestamp(end), color="grey", alpha=0.15)
    ax.text(pd.Timestamp(start), ax.get_ylim()[0] * 0.95, label, fontsize=8, rotation=90, va="bottom")

ax.set_title("낙폭(Drawdown) Underwater 차트 — 주요 위기 구간 음영 표시")
ax.set_ylabel("Drawdown (%)")
ax.legend()
plt.savefig(os.path.join(FIG_DIR, "03_drawdown_underwater.png"), dpi=120, bbox_inches="tight")
plt.show()

mdd_df = pd.DataFrame(mdd_rows).set_index("ticker")
mdd_df"""
))

cells.append(md(
"""**위기별 세부 MDD·회복기간 비교**: 아래 셀은 각 위기 구간(2008/2020/2022) 내에서
발생한 국지적 MDD를 별도로 계산한다 (전체 구간 MDD와는 다를 수 있음)."""
))

cells.append(code(
"""crisis_mdd_rows = []
for label, (start, end) in CRISIS_PERIODS.items():
    for t in TICKERS:
        window = common_clean.loc[start:end, t].dropna()
        if len(window) < 2:
            continue
        stats = m.max_drawdown_stats(window)
        crisis_mdd_rows.append({"crisis": label, "ticker": t, **stats})

crisis_mdd_df = pd.DataFrame(crisis_mdd_rows).set_index(["crisis", "ticker"])
crisis_mdd_df"""
))

cells.append(md("### 4-4. 롤링 변동성(연율화) & QLD 실제 vs QQQ×2 이론값 괴리"))

cells.append(code(
"""fig, ax = plt.subplots()
for t in TICKERS:
    vol = m.rolling_volatility(common_clean[t], window=63)
    ax.plot(vol.index, vol.values, label=t)
ax.set_title("롤링 변동성 (63거래일, 연율화 %)")
ax.set_ylabel("Annualized Volatility (%)")
ax.legend()
plt.savefig(os.path.join(FIG_DIR, "04_rolling_volatility.png"), dpi=120, bbox_inches="tight")
plt.show()"""
))

cells.append(code(
"""gap_df = m.leveraged_gap(common_clean["QQQ"], common_clean["QLD"], leverage=2.0)

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(11, 7))
ax1.plot(gap_df.index, gap_df["leveraged_actual"], label="QLD 실제")
ax1.plot(gap_df.index, gap_df["leveraged_theoretical"], label="QQQ×2 이론값", linestyle="--")
ax1.set_yscale("log")
ax1.set_title("QLD 실제 누적수익률 vs QQQ×2 이론값 (로그스케일)")
ax1.legend()

ax2.plot(gap_df.index, gap_df["gap_pct"], color="darkred")
ax2.axhline(0, color="black", linewidth=0.8)
ax2.set_title("괴리율 (실제-이론)/이론 × 100  — 음수일수록 변동성 드래그 심화")
ax2.set_ylabel("Gap (%)")

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "05_qld_theoretical_gap.png"), dpi=120, bbox_inches="tight")
plt.show()

print("최종 시점 괴리율:", f"{gap_df['gap_pct'].iloc[-1]:.2f}%")"""
))

cells.append(md("### 4-5. 연도별 수익률 비교"))

cells.append(code(
"""annual_returns = common_clean.resample("YE").last().pct_change().dropna() * 100
annual_returns.index = annual_returns.index.year

ax = annual_returns.plot(kind="bar", figsize=(13, 5))
ax.set_title("연도별 수익률 비교 (SPY / QQQ / QLD)")
ax.set_ylabel("Annual Return (%)")
ax.axhline(0, color="black", linewidth=0.8)
plt.savefig(os.path.join(FIG_DIR, "06_annual_returns_bar.png"), dpi=120, bbox_inches="tight")
plt.show()

annual_returns"""
))

cells.append(md(
"""### 4-6. 적립식(DCA) vs 거치식(Lump-sum) 투자 비교 (분석 주제 모음 — 주제 4)

"언제 사느냐가 얼마나 중요한가?"에 대한 실증 비교. 매월 첫 거래일에 정액(월 50만원)을
투자하는 적립식(DCA)과, 그 총 투자원금을 구간 시작일에 한 번에 투자하는 거치식(Lump-sum)을
**동일한 총 투자금** 기준으로 비교한다. (`src/metrics.py`의 `dca_vs_lumpsum` 함수)"""
))

cells.append(code(
"""MONTHLY_INVEST_KRW = 500_000  # 월 적립 금액 (원)

dca_results = {t: m.dca_vs_lumpsum(common_clean[t], monthly_amount=MONTHLY_INVEST_KRW) for t in TICKERS}

dca_summary_rows = []
for t in TICKERS:
    r = dca_results[t]
    dca_summary_rows.append({
        "티커": t,
        "개월수": r["n_months"],
        "총투자원금(원)": round(r["total_invested"], 0),
        "적립식_최종가치(원)": round(r["dca_final_value"], 0),
        "적립식_수익률(%)": round(r["dca_return_pct"], 2),
        "거치식_최종가치(원)": round(r["lump_final_value"], 0),
        "거치식_수익률(%)": round(r["lump_return_pct"], 2),
    })
dca_summary_df = pd.DataFrame(dca_summary_rows).set_index("티커")

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharex=True)
for ax, t in zip(axes, TICKERS):
    tl = dca_results[t]["timeline"]
    ax.plot(tl.index, tl["누적투자원금"], label="누적투자원금", color="grey", linestyle="--")
    ax.plot(tl.index, tl["적립식_평가금액"], label="적립식(DCA)", color="tab:blue")
    ax.plot(tl.index, tl["거치식_평가금액"], label="거치식(Lump-sum)", color="tab:red")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, pos: f"{v/1e8:.0f}억" if v >= 1e8 else f"{v/1e4:.0f}만"))
    ax.set_ylabel("평가금액 (원)")
    ax.set_title(f"{t} — 적립식 vs 거치식")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
fig.suptitle("월 50만원 적립식 vs 동일 총액 거치식 투자 비교 (로그스케일)")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "09_dca_vs_lumpsum.png"), dpi=120, bbox_inches="tight")
plt.show()

dca_summary_df"""
))

cells.append(md(
"""## 5. 인사이트 도출 (관찰 → 해석)

실제 데이터(2006-06-21~2026-08-25 공통구간) 실행 결과 기준. 전체 서술은 `REPORT.md` §6 참고.

- **인사이트 1 (질문1)**: 관찰 — 누적수익률 SPY 784.80%/QQQ 2,048.91%/QLD 8,936.17%, CAGR 11.41%/16.42%/25.01%, 변동성도 같은 순서로 19.38%/22.08%/43.96%. 해석 — 초과수익은 그만큼의 변동성을 대가로 한다.
- **인사이트 2 (질문2)**: 관찰 — QLD 실제 vs QQQ×2 이론값 최종 괴리율 -47.44%, 그래프상 선형적으로 꾸준히 벌어짐(위기 구간에서 가속). 해석 — 변동성 드래그는 사건이 아니라 매일 재조정되는 구조에서 누적되는 현상.
- **인사이트 3 (질문3)**: 관찰 — 2008년 QLD -83.13%(회복 1,106일)로 최대, 2022년은 QQQ(-34.83%)가 SPY(-24.50%)보다 더 하락해 순위가 뒤바뀜. 해석 — 위기 원인(신용위기 vs 금리인상)에 따라 가장 취약한 상품이 다름.
- **인사이트 4 (질문4)**: 관찰 — 장기적으론 QQQ가 SPY를 크게 앞서지만(2,048.91% vs 784.80%), 2022년 한 해는 QQQ가 더 크게 하락. 해석 — 저금리기 기술주 프리미엄이 금리인상기엔 페널티로 반전.
- **인사이트 5 (질문5)**: 관찰 — CAGR/변동성 비율이 SPY 0.589, QQQ 0.744, QLD 0.569로 QLD가 절대수익 1위임에도 위험조정 기준으론 최하위권. 해석 — 레버리지가 "항상 유리"한 건 아니며 -83% MDD를 버틸 수 있는지가 더 중요한 변수.
- **인사이트 6 (질문6, 주제4)**: 관찰 — 동일 총투자금(월 50만원×243개월=1억 2,150만원) 기준, 거치식 수익률이 SPY 775.28%, QQQ 2,016.71%, QLD 8,714.34%로 적립식(376.69%/763.97%/3,537.22%)을 세 상품 모두에서 큰 차이로 앞섰다. 해석 — 20년 대부분이 상승장이었던 이번 구간에서는 돈을 더 일찍, 더 오래 시장에 노출시킨 거치식이 유리했다. 다만 이는 "거치식이 항상 우월하다"는 뜻이 아니라, 이 결과 자체가 표본 기간이 상승장 편향임을 보여주는 방증이며, 적립식의 실질적 가치는 최종 수익률이 아니라 (하락장 초입에 목돈을 한 번에 넣는) 매수 타이밍 리스크를 분산시키는 데 있다."""
))

cells.append(md(
"""## 6. 보너스 B-A: QQQ 월별 수익률 STL 분해

`statsmodels.tsa.seasonal.STL`로 월별 수익률의 추세(trend)/계절성(seasonal)/잔차(resid)를 분리한다."""
))

cells.append(code(
"""monthly_ret, stl_result = m.stl_monthly_decompose(common_clean["QQQ"], period=12)

fig = stl_result.plot()
fig.set_size_inches(11, 8)
fig.suptitle("QQQ 월별 수익률 STL 분해 (원본/추세/계절성/잔차)", y=1.02)
plt.savefig(os.path.join(FIG_DIR, "07_stl_decomposition.png"), dpi=120, bbox_inches="tight")
plt.show()

seasonal_amp = stl_result.seasonal.std()
resid_amp = stl_result.resid.std()
print(f"계절성 성분 표준편차: {seasonal_amp:.3f}")
print(f"잔차 성분 표준편차: {resid_amp:.3f}")
print(f"계절성/잔차 비율: {seasonal_amp / resid_amp:.2f} (1보다 많이 작으면 계절성이 노이즈 대비 미미하다는 뜻)")"""
))

cells.append(md(
"""## 7. 보너스 B-B: 베이스라인 예측 (나이브 / 이동평균 / 선형추세)

**목적**: 예측 정확도 자체보다 "왜 단기 주가 예측이 어려운가"(랜덤워크 가설)를 보여주는 데 있다.
이 예측은 투자 판단 근거가 될 수 없다."""
))

cells.append(code(
"""price = common_clean["QQQ"]
train, test = m.train_test_split_tail(price, test_size=20)
horizon = len(test)

pred_naive = m.baseline_naive(train, horizon)
pred_ma = m.baseline_moving_average(train, horizon, window=20)
pred_linear = m.baseline_linear_trend(train, horizon)

results = pd.DataFrame({
    "actual": test.values,
    "naive": pred_naive,
    "moving_average": pred_ma,
    "linear_trend": pred_linear,
}, index=test.index)

mae_table = pd.Series({
    "naive": m.mae(test.values, pred_naive),
    "moving_average": m.mae(test.values, pred_ma),
    "linear_trend": m.mae(test.values, pred_linear),
}, name="MAE")

fig, ax = plt.subplots()
ax.plot(train.index[-60:], train.values[-60:], label="학습구간(최근 60일)", color="grey")
ax.plot(results.index, results["actual"], label="실제값", color="black", linewidth=2)
ax.plot(results.index, results["naive"], label="나이브", linestyle="--")
ax.plot(results.index, results["moving_average"], label="이동평균", linestyle="--")
ax.plot(results.index, results["linear_trend"], label="선형추세", linestyle="--")
ax.set_title("QQQ 종가: 실제값 vs 베이스라인 예측 (마지막 20거래일)")
ax.legend()
plt.savefig(os.path.join(FIG_DIR, "08_baseline_forecast.png"), dpi=120, bbox_inches="tight")
plt.show()

print(mae_table)
print("\\n결론 방향: 베이스라인 대비 뚜렷한 우위가 없다면, 이는 실패가 아니라")
print("단기 가격이 랜덤워크에 가깝다는 효율적 시장 가설과 부합하는 결과로 해석한다.")"""
))

cells.append(md(
"""## 다음 단계

- [x] 실제 컴퓨터(로컬 환경)에서 `pip install -r requirements.txt` → `python data/fetch_data.py` 실행
- [x] 이 노트북을 실제 데이터로 실행해 각 차트/표 확인 (`figures/` 폴더 참고)
- [x] §5 인사이트 섹션을 실제 수치 기반으로 완성
- [x] `REPORT.md` 작성 (분석 주제/질문/데이터 설명/시각화/인사이트/결론·한계/보너스 결과/AI 사용 로그)
- [x] 보너스 A: `dashboard.py` (Streamlit) 구현
- [x] 추가 분석(주제 4): 적립식(DCA) vs 거치식(Lump-sum) 비교 (§4-6, 인사이트 6)
- [ ] GitHub 저장소 정리 및 제출"""
))

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(notebook, f, ensure_ascii=False, indent=1)

print(f"생성 완료: {OUT_PATH} ({len(cells)}개 셀)")
