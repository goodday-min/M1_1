"""
시계열 분석 유틸리티 함수 모음.

PRD §5 "적용할 시계열 기법" 대응:
1. normalize_to_100          - 누적수익률 정규화(인덱싱)
2. moving_averages           - 이동평균선 (50/200일)
3. drawdown_curve / max_drawdown_stats - 최대낙폭(MDD) 및 낙폭곡선
4. rolling_volatility        - 롤링 변동성(연율화 표준편차)
5. leveraged_gap             - QLD 실제 수익률 vs QQQ 이론적 2배 수익률 괴리

PRD §8-B 보너스 대응:
6. stl_monthly_decompose     - 월별 수익률 STL 분해
7. baseline_naive / baseline_moving_average / baseline_linear_trend - 베이스라인 예측
8. mae                       - 평균절대오차

추가 분석 (분석 주제 모음 - 주제 4) 대응:
9. dca_vs_lumpsum            - 적립식(매월 정액 투자) vs 거치식(일시 투자) 수익률 비교

모든 함수는 pandas Series/DataFrame을 입력·출력으로 사용하며,
날짜 인덱스(DatetimeIndex)를 가정한다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


# ---------------------------------------------------------------------------
# 1. 누적수익률 정규화
# ---------------------------------------------------------------------------

def normalize_to_100(price: pd.Series) -> pd.Series:
    """시작일을 100으로 맞춘 지수화 시리즈를 반환한다."""
    price = price.dropna()
    if price.empty:
        raise ValueError("정규화할 데이터가 비어 있습니다.")
    return price / price.iloc[0] * 100.0


def cagr(price: pd.Series) -> float:
    """연평균 복리 수익률(CAGR)을 계산한다.

    CAGR = (기말가격 / 기초가격)^(1/보유연수) - 1
    보유연수는 실제 달력일수 기준(365.25일)으로 계산한다.
    """
    price = price.dropna()
    if len(price) < 2:
        raise ValueError("CAGR 계산에는 최소 2개 이상의 데이터가 필요합니다.")
    n_years = (price.index[-1] - price.index[0]).days / 365.25
    if n_years <= 0:
        raise ValueError("기간이 0년 이하입니다.")
    total_return = price.iloc[-1] / price.iloc[0]
    return total_return ** (1 / n_years) - 1


def cumulative_return(price: pd.Series) -> float:
    """전체 구간 누적수익률(%)을 반환한다."""
    price = price.dropna()
    return (price.iloc[-1] / price.iloc[0] - 1) * 100.0


# ---------------------------------------------------------------------------
# 2. 이동평균선
# ---------------------------------------------------------------------------

def moving_averages(price: pd.Series, windows=(50, 200)) -> pd.DataFrame:
    """지정한 window(거래일)별 이동평균을 컬럼으로 갖는 DataFrame을 반환한다."""
    out = pd.DataFrame({"price": price})
    for w in windows:
        out[f"MA{w}"] = price.rolling(window=w, min_periods=w).mean()
    return out


def golden_dead_cross(ma_df: pd.DataFrame, short_col="MA50", long_col="MA200") -> pd.DataFrame:
    """골든크로스(단기>장기 전환)와 데드크로스(단기<장기 전환) 발생일을 표시한다."""
    diff = ma_df[short_col] - ma_df[long_col]
    sign = np.sign(diff)
    cross = sign.diff()
    result = ma_df.copy()
    result["golden_cross"] = cross == 2  # -1 -> 1
    result["dead_cross"] = cross == -2  # 1 -> -1
    return result


# ---------------------------------------------------------------------------
# 3. 최대낙폭(MDD) / 낙폭곡선
# ---------------------------------------------------------------------------

def drawdown_curve(price: pd.Series) -> pd.Series:
    """낙폭(drawdown) 곡선: 각 시점까지의 최고점 대비 하락률(%, 0 이하 값)."""
    running_max = price.cummax()
    dd = (price / running_max - 1.0) * 100.0
    return dd


def max_drawdown_stats(price: pd.Series) -> dict:
    """최대낙폭(MDD)과 발생일, 회복 소요일수를 계산한다.

    회복일: MDD 저점 이후 최초로 이전 고점을 다시 회복한 날짜.
    아직 회복하지 못했다면 recovery_date=None, recovery_days=None.
    """
    dd = drawdown_curve(price)
    trough_date = dd.idxmin()
    mdd = dd.min()

    peak_before_trough = price.loc[:trough_date].idxmax()
    peak_value = price.loc[peak_before_trough]

    after_trough = price.loc[trough_date:]
    recovered = after_trough[after_trough >= peak_value]
    if len(recovered) > 0:
        recovery_date = recovered.index[0]
        recovery_days = (recovery_date - trough_date).days
    else:
        recovery_date = None
        recovery_days = None

    return {
        "mdd_pct": mdd,
        "peak_date": peak_before_trough,
        "trough_date": trough_date,
        "recovery_date": recovery_date,
        "recovery_days": recovery_days,
    }


# ---------------------------------------------------------------------------
# 4. 롤링 변동성 (연율화 표준편차)
# ---------------------------------------------------------------------------

def rolling_volatility(price: pd.Series, window: int = 63) -> pd.Series:
    """일간 수익률의 롤링 표준편차를 연율화(annualize)하여 반환한다.

    window 기본값 63거래일(약 3개월).
    """
    daily_ret = price.pct_change()
    vol = daily_ret.rolling(window=window).std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    return vol * 100.0  # %


def annualized_volatility(price: pd.Series) -> float:
    """전체 구간 연율화 변동성(%)을 반환한다."""
    daily_ret = price.pct_change().dropna()
    return daily_ret.std() * np.sqrt(TRADING_DAYS_PER_YEAR) * 100.0


# ---------------------------------------------------------------------------
# 5. QLD 실제 수익률 vs QQQ 이론적 2배 수익률 괴리 (변동성 드래그)
# ---------------------------------------------------------------------------

def leveraged_gap(base_price: pd.Series, leveraged_price: pd.Series, leverage: float = 2.0) -> pd.DataFrame:
    """기초지수(QQQ) 대비 레버리지 상품(QLD)의 실제 누적수익률과
    '이론값(기초 일간수익률 × leverage 복리)'을 비교한 DataFrame을 반환한다.

    반환 컬럼:
        base_index        : 기초지수 정규화(100 시작) 누적수익률
        leveraged_actual   : 레버리지 상품 실제 정규화 누적수익률
        leveraged_theoretical : 기초 일간수익률×leverage를 매일 복리 재투자했다고 가정한 이론값
        gap_pct            : (실제 - 이론) / 이론 × 100, 음수면 이론보다 실제가 저조(드래그)
    """
    base = normalize_to_100(base_price)
    lev_actual = normalize_to_100(leveraged_price)

    # 공통 인덱스로 정렬
    idx = base.index.intersection(lev_actual.index)
    base = base.loc[idx]
    lev_actual = lev_actual.loc[idx]

    base_daily_ret = base.pct_change().fillna(0)
    theoretical_daily_ret = base_daily_ret * leverage
    lev_theoretical = 100.0 * (1 + theoretical_daily_ret).cumprod()

    gap_pct = (lev_actual - lev_theoretical) / lev_theoretical * 100.0

    return pd.DataFrame(
        {
            "base_index": base,
            "leveraged_actual": lev_actual,
            "leveraged_theoretical": lev_theoretical,
            "gap_pct": gap_pct,
        }
    )


# ---------------------------------------------------------------------------
# 6. STL 월별 분해 (보너스 A)
# ---------------------------------------------------------------------------

def monthly_return_series(price: pd.Series) -> pd.Series:
    """일별 가격 시리즈를 월말 기준 월간 수익률(%) 시리즈로 변환한다."""
    monthly_price = price.resample("ME").last()
    monthly_ret = monthly_price.pct_change().dropna() * 100.0
    return monthly_ret


def stl_monthly_decompose(price: pd.Series, period: int = 12):
    """statsmodels STL로 월간 수익률 시계열을 분해한다.

    Returns: statsmodels.tsa.seasonal.DecomposeResult
    (trend, seasonal, resid 속성을 가짐)

    주의: statsmodels 패키지가 설치되어 있어야 한다 (requirements.txt 참고).
    """
    from statsmodels.tsa.seasonal import STL

    monthly_ret = monthly_return_series(price)
    stl = STL(monthly_ret, period=period, robust=True)
    result = stl.fit()
    return monthly_ret, result


# ---------------------------------------------------------------------------
# 7. 베이스라인 예측 (보너스 B)
# ---------------------------------------------------------------------------

def train_test_split_tail(price: pd.Series, test_size: int = 20):
    """마지막 test_size 거래일을 테스트 구간으로 분리한다."""
    if len(price) <= test_size:
        raise ValueError("전체 데이터 길이가 test_size보다 작거나 같습니다.")
    train = price.iloc[:-test_size]
    test = price.iloc[-test_size:]
    return train, test


def baseline_naive(train: pd.Series, horizon: int) -> np.ndarray:
    """나이브 예측: 마지막 관측값을 horizon 기간 동안 그대로 유지."""
    last_value = train.iloc[-1]
    return np.full(horizon, last_value)


def baseline_moving_average(train: pd.Series, horizon: int, window: int = 20) -> np.ndarray:
    """단순 이동평균 예측: 학습구간 마지막 window일 평균을 horizon 기간 동안 유지."""
    ma_value = train.iloc[-window:].mean()
    return np.full(horizon, ma_value)


def baseline_linear_trend(train: pd.Series, horizon: int) -> np.ndarray:
    """선형추세 외삽: 학습구간 전체에 선형회귀를 적합해 horizon만큼 연장."""
    x = np.arange(len(train))
    y = train.values
    coeffs = np.polyfit(x, y, deg=1)  # [slope, intercept]
    future_x = np.arange(len(train), len(train) + horizon)
    return np.polyval(coeffs, future_x)


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """평균절대오차(Mean Absolute Error)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """평균절대백분율오차(%)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100.0)


# ---------------------------------------------------------------------------
# 9. 적립식(DCA) vs 거치식(Lump-sum) 투자 비교 (분석 주제 모음 - 주제 4)
# ---------------------------------------------------------------------------

def dca_vs_lumpsum(price: pd.Series, monthly_amount: float = 100.0) -> dict:
    """동일한 총 투자금 기준으로 적립식(매월 정액 투자)과 거치식(최초 1회 일시 투자)의
    최종 성과를 비교한다.

    방법:
      - 매월 첫 거래일 가격으로 `monthly_amount`만큼 매수했다고 가정 (적립식).
      - 적립식의 총 투자원금(=monthly_amount × 개월수)을 구간 시작일에 전액
        일시 투자했다고 가정 (거치식) — 두 전략의 총 투자금을 동일하게 맞춰
        공정하게 비교한다.

    Parameters
    ----------
    price : pd.Series
        일별 가격 시계열 (DatetimeIndex, 오름차순 정렬).
    monthly_amount : float
        매월 적립 투자 금액 (통화 단위는 임의, 비율 비교이므로 무관).

    Returns
    -------
    dict with keys:
        n_months, monthly_amount, total_invested,
        dca_final_value, dca_return_pct,
        lump_final_value, lump_return_pct,
        timeline (DataFrame: 누적투자원금 / 적립식_평가금액 / 거치식_평가금액, 월별)
    """
    price = price.dropna().sort_index()
    monthly_price = price.resample("MS").first().dropna()
    if len(monthly_price) < 2:
        raise ValueError("적립식 vs 거치식 비교에는 최소 2개월 이상의 데이터가 필요합니다.")

    shares_bought = monthly_amount / monthly_price
    cum_shares = shares_bought.cumsum()
    dca_value_path = cum_shares * monthly_price
    cum_invested_path = pd.Series(monthly_amount, index=monthly_price.index).cumsum()

    total_invested = float(cum_invested_path.iloc[-1])
    dca_final_value = float(dca_value_path.iloc[-1])
    dca_return_pct = (dca_final_value / total_invested - 1) * 100.0

    lump_shares = total_invested / monthly_price.iloc[0]
    lump_value_path = lump_shares * monthly_price
    lump_final_value = float(lump_value_path.iloc[-1])
    lump_return_pct = (lump_final_value / total_invested - 1) * 100.0

    timeline = pd.DataFrame(
        {
            "누적투자원금": cum_invested_path,
            "적립식_평가금액": dca_value_path,
            "거치식_평가금액": lump_value_path,
        }
    )

    return {
        "n_months": len(monthly_price),
        "monthly_amount": monthly_amount,
        "total_invested": total_invested,
        "dca_final_value": dca_final_value,
        "dca_return_pct": dca_return_pct,
        "lump_final_value": lump_final_value,
        "lump_return_pct": lump_return_pct,
        "timeline": timeline,
    }
