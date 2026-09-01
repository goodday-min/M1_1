"""
metrics.py 함수 검증 스크립트 (합성 데이터 기반 단위 테스트).

실제 SPY/QQQ/QLD 데이터가 아직 없는 상태에서도 함수 로직 자체의 정확성을
검증하기 위해, 알려진 성질을 갖는 합성(랜덤워크) 데이터로 각 함수를 테스트한다.
statsmodels가 설치되어 있지 않은 환경에서는 STL 테스트만 건너뛴다.

실행: python src/test_metrics.py
"""

import numpy as np
import pandas as pd

import metrics as m


def make_synthetic_price(n_days=5200, start="2006-06-21", daily_drift=0.0003, daily_vol=0.012, seed=42):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, periods=n_days)
    daily_ret = rng.normal(loc=daily_drift, scale=daily_vol, size=n_days)
    price = 100 * (1 + pd.Series(daily_ret, index=dates)).cumprod()
    return price


def test_normalize_to_100():
    price = make_synthetic_price()
    idx = m.normalize_to_100(price)
    assert abs(idx.iloc[0] - 100.0) < 1e-9, "정규화 시작값은 100이어야 함"
    print("OK: normalize_to_100")


def test_cagr_known_value():
    # 정확히 10년간 매년 10%씩 성장하는 시리즈 -> CAGR = 10%
    dates = pd.date_range("2016-01-01", periods=11, freq="YS")
    price = pd.Series([100 * (1.10 ** i) for i in range(11)], index=dates)
    result = m.cagr(price)
    assert abs(result - 0.10) < 0.01, f"CAGR 기대값 0.10, 실제 {result}"
    print(f"OK: cagr (계산값={result:.4f}, 기대값=0.10)")


def test_drawdown_and_mdd():
    dates = pd.bdate_range("2020-01-01", periods=10)
    # 100 -> 120(고점) -> 60(저점, -50% MDD) -> 130(회복)
    prices = [100, 110, 120, 100, 80, 60, 70, 90, 110, 130]
    price = pd.Series(prices, index=dates, dtype=float)
    dd = m.drawdown_curve(price)
    assert abs(dd.min() - (-50.0)) < 1e-9, f"MDD 기대값 -50.0, 실제 {dd.min()}"

    stats = m.max_drawdown_stats(price)
    assert abs(stats["mdd_pct"] - (-50.0)) < 1e-9
    assert stats["recovery_date"] is not None, "130에서 120 고점을 회복했어야 함"
    print(f"OK: drawdown_curve / max_drawdown_stats -> {stats}")


def test_rolling_volatility_reasonable_range():
    price = make_synthetic_price(daily_vol=0.01)
    vol = m.rolling_volatility(price, window=63).dropna()
    # 일 변동성 1% -> 연율화 대략 1%*sqrt(252) ~= 15.9%
    assert 5 < vol.mean() < 30, f"연율화 변동성이 비정상적 범위: {vol.mean()}"
    print(f"OK: rolling_volatility (평균={vol.mean():.2f}%, 이론적 기대값=~15.9%)")


def test_leveraged_gap_no_drift_matches_theory_closely():
    # 변동성이 0이면(추세만 있는 경우) 실제와 이론값이 거의 일치해야 함
    dates = pd.bdate_range("2010-01-01", periods=500)
    daily_ret = pd.Series(0.0005, index=dates)  # 무변동, 일정 수익률
    base = 100 * (1 + daily_ret).cumprod()
    leveraged = 100 * (1 + daily_ret * 2).cumprod()
    gap_df = m.leveraged_gap(base, leveraged, leverage=2.0)
    assert gap_df["gap_pct"].abs().max() < 1e-6, "무변동 구간에서는 실제=이론이어야 함"
    print("OK: leveraged_gap (무변동 구간 실제=이론 확인)")


def test_leveraged_gap_volatility_drag_negative():
    # 변동성이 있으면 레버리지 실제 수익률이 이론값보다 낮아야 함(변동성 드래그)
    rng = np.random.default_rng(1)
    dates = pd.bdate_range("2010-01-01", periods=1000)
    base_ret = rng.normal(0.0002, 0.02, size=1000)  # 고변동성, 약한 상승 추세
    base = pd.Series(100 * (1 + pd.Series(base_ret, index=dates)).cumprod())
    # 실제 레버리지 상품도 매일 2배 수익률로 리밸런싱된다고 가정(이론과 동일 룰) ->
    # 이 경우 gap은 0에 가까워야 하므로, 대신 "일부 비용 차감"을 추가 시뮬레이션
    leveraged_daily = base_ret * 2 - 0.0001  # 매일 운용보수 등으로 약간의 추가 손실 가정
    leveraged = pd.Series(100 * (1 + pd.Series(leveraged_daily, index=dates)).cumprod())
    gap_df = m.leveraged_gap(base, leveraged, leverage=2.0)
    assert gap_df["gap_pct"].iloc[-1] < 0, "비용을 반영하면 실제가 이론보다 낮아야 함"
    print(f"OK: leveraged_gap (비용 반영 시 최종 gap={gap_df['gap_pct'].iloc[-1]:.2f}%)")


def test_baseline_forecasts_shapes():
    price = make_synthetic_price(n_days=300)
    train, test = m.train_test_split_tail(price, test_size=20)
    horizon = len(test)

    naive = m.baseline_naive(train, horizon)
    ma_pred = m.baseline_moving_average(train, horizon, window=20)
    lin_pred = m.baseline_linear_trend(train, horizon)

    assert len(naive) == len(ma_pred) == len(lin_pred) == horizon
    assert np.all(naive == naive[0]), "나이브 예측은 상수여야 함"
    assert np.all(ma_pred == ma_pred[0]), "이동평균 예측은 상수여야 함"

    mae_naive = m.mae(test.values, naive)
    mae_ma = m.mae(test.values, ma_pred)
    mae_lin = m.mae(test.values, lin_pred)
    print(f"OK: baseline forecasts -> MAE naive={mae_naive:.3f}, MA={mae_ma:.3f}, linear={mae_lin:.3f}")


def test_moving_averages_and_cross():
    price = make_synthetic_price(n_days=600)
    ma_df = m.moving_averages(price, windows=(50, 200))
    assert "MA50" in ma_df.columns and "MA200" in ma_df.columns
    assert ma_df["MA200"].dropna().shape[0] > 0
    crossed = m.golden_dead_cross(ma_df)
    assert "golden_cross" in crossed.columns and "dead_cross" in crossed.columns
    print(f"OK: moving_averages / golden_dead_cross (골든크로스 {crossed['golden_cross'].sum()}회, "
          f"데드크로스 {crossed['dead_cross'].sum()}회)")


def test_monthly_return_series():
    price = make_synthetic_price(n_days=800)
    monthly = m.monthly_return_series(price)
    assert monthly.index.is_monotonic_increasing
    assert len(monthly) > 20
    print(f"OK: monthly_return_series ({len(monthly)}개월)")


def test_dca_vs_lumpsum_flat_price_zero_return():
    # 가격이 전혀 변하지 않으면 적립식·거치식 모두 수익률 0%여야 함
    dates = pd.bdate_range("2010-01-01", periods=800)
    price = pd.Series(100.0, index=dates)
    result = m.dca_vs_lumpsum(price, monthly_amount=100.0)
    assert abs(result["dca_return_pct"]) < 1e-6, "가격 불변 시 적립식 수익률은 0%여야 함"
    assert abs(result["lump_return_pct"]) < 1e-6, "가격 불변 시 거치식 수익률은 0%여야 함"
    print("OK: dca_vs_lumpsum (가격 불변 -> 수익률 0% 확인)")


def test_dca_vs_lumpsum_equal_total_invested():
    # 적립식 총투자금(월 100 x 개월수)과 거치식 총투자금이 정확히 동일해야 공정 비교
    price = make_synthetic_price(n_days=1500)
    result = m.dca_vs_lumpsum(price, monthly_amount=100.0)
    expected_total = 100.0 * result["n_months"]
    assert abs(result["total_invested"] - expected_total) < 1e-6, "총투자금이 monthly_amount x 개월수와 달라짐"
    print(f"OK: dca_vs_lumpsum (총투자금 동일성 확인, {result['n_months']}개월 x $100 = ${expected_total:,.0f})")


def test_dca_vs_lumpsum_lumpsum_wins_in_steady_uptrend():
    # 꾸준한 상승 추세(변동성 낮음)에서는 자금을 더 일찍 투입하는 거치식이 유리해야 함
    dates = pd.bdate_range("2010-01-01", periods=2500)
    daily_ret = pd.Series(0.0004, index=dates)  # 변동성 없는 꾸준한 상승
    price = 100 * (1 + daily_ret).cumprod()
    result = m.dca_vs_lumpsum(price, monthly_amount=100.0)
    assert result["lump_return_pct"] > result["dca_return_pct"], "꾸준한 상승장에서는 거치식 수익률이 더 높아야 함"
    print(f"OK: dca_vs_lumpsum (상승장에서 거치식 {result['lump_return_pct']:.1f}% > "
          f"적립식 {result['dca_return_pct']:.1f}%)")


def test_stl_if_available():
    try:
        import statsmodels  # noqa: F401
    except ImportError:
        print("SKIP: stl_monthly_decompose (statsmodels 미설치 - 실행 환경에서 별도 검증 필요)")
        return
    price = make_synthetic_price(n_days=3000)
    monthly_ret, result = m.stl_monthly_decompose(price)
    assert len(result.trend) == len(monthly_ret)
    assert len(result.seasonal) == len(monthly_ret)
    assert len(result.resid) == len(monthly_ret)
    print("OK: stl_monthly_decompose")


if __name__ == "__main__":
    tests = [
        test_normalize_to_100,
        test_cagr_known_value,
        test_drawdown_and_mdd,
        test_rolling_volatility_reasonable_range,
        test_leveraged_gap_no_drift_matches_theory_closely,
        test_leveraged_gap_volatility_drag_negative,
        test_baseline_forecasts_shapes,
        test_moving_averages_and_cross,
        test_monthly_return_series,
        test_dca_vs_lumpsum_flat_price_zero_return,
        test_dca_vs_lumpsum_equal_total_invested,
        test_dca_vs_lumpsum_lumpsum_wins_in_steady_uptrend,
        test_stl_if_available,
    ]
    failed = 0
    for t in tests:
        try:
            t()
        except AssertionError as e:
            failed += 1
            print(f"FAIL: {t.__name__} -> {e}")
        except Exception as e:
            failed += 1
            print(f"ERROR: {t.__name__} -> {type(e).__name__}: {e}")

    print("\n" + "=" * 50)
    if failed == 0:
        print(f"전체 {len(tests)}개 테스트 통과")
    else:
        print(f"{failed}/{len(tests)}개 테스트 실패")
        raise SystemExit(1)
