"""
데이터 수집 스크립트: SPY / QQQ / QLD 일별 가격 데이터 (Yahoo Finance)

- PRD 참고: claude/PRD.md
- 사용 라이브러리: yfinance (pip install yfinance)
- 가격 기준: Adj Close (배당 재투자·액면분할 반영, 총수익 기준 비교)
- 저장 위치: data/raw/{ticker}.csv (개별 전체 역사), data/common_period.csv (공통구간 정렬본)

실행 방법:
    pip install -r requirements.txt
    python data/fetch_data.py

주의:
- Yahoo Finance 데이터는 비상업적 개인/학습 목적으로만 사용합니다.
- 수집일 기준 스냅샷이며, 이후 재실행 시 최신 거래일까지 데이터가 갱신됩니다.
"""

import os
from datetime import datetime, timezone

import pandas as pd
import yfinance as yf

# ---------------------------------------------------------------------------
# 설정
# ---------------------------------------------------------------------------

TICKERS = {
    "SPY": "1993-01-22",  # 상장일 (참고용, 실제 다운로드는 yfinance 최대 범위로 시도)
    "QQQ": "1999-03-10",
    "QLD": "2006-06-21",
}

# 공통구간(핵심 비교): QLD 상장일 ~ 오늘
COMMON_START = "2006-06-21"

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(THIS_DIR, "raw")
os.makedirs(RAW_DIR, exist_ok=True)


def fetch_ticker(ticker: str) -> pd.DataFrame:
    """yfinance로 티커의 전체 역사 일별 데이터를 받아온다 (auto_adjust=False로
    Close/Adj Close를 모두 보존하고, 분석에는 Adj Close를 사용한다)."""
    print(f"[fetch] {ticker} 다운로드 중...")
    df = yf.download(
        ticker,
        start="1990-01-01",
        end=None,  # 오늘까지
        auto_adjust=False,
        actions=False,
        progress=False,
    )

    if df.empty:
        raise RuntimeError(
            f"{ticker} 데이터가 비어 있습니다. 네트워크 연결 또는 티커명을 확인하세요."
        )

    # yfinance가 MultiIndex 컬럼을 반환하는 경우(멀티 티커 등) 평탄화
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()
    df["Ticker"] = ticker
    return df


def main():
    all_frames = []

    for ticker in TICKERS:
        df = fetch_ticker(ticker)
        raw_path = os.path.join(RAW_DIR, f"{ticker.lower()}.csv")
        df.to_csv(raw_path, index=False)
        n_rows = len(df)
        date_min = df["Date"].min()
        date_max = df["Date"].max()
        print(f"  -> 저장: {raw_path} ({n_rows}행, {date_min.date()} ~ {date_max.date()})")
        all_frames.append(df)

    # ------------------------------------------------------------------
    # 공통구간(2006-06-21 ~ 오늘) 정렬: 3개 티커의 Adj Close를 wide 포맷으로
    # ------------------------------------------------------------------
    combined = pd.concat(all_frames, ignore_index=True)
    combined["Date"] = pd.to_datetime(combined["Date"])

    common = combined[combined["Date"] >= pd.Timestamp(COMMON_START)]
    wide = common.pivot(index="Date", columns="Ticker", values="Adj Close")
    wide = wide.sort_index()

    # 결측치 안내 (거래일 중 특정 티커만 누락된 경우 forward-fill은 분석 단계에서 처리)
    missing_summary = wide.isna().sum()
    print("\n[공통구간 결측치 현황]")
    print(missing_summary)

    common_path = os.path.join(THIS_DIR, "common_period.csv")
    wide.to_csv(common_path)
    print(f"\n[fetch] 공통구간 정렬 데이터 저장: {common_path}")
    print(f"  기간: {wide.index.min().date()} ~ {wide.index.max().date()} ({len(wide)}행)")

    # ------------------------------------------------------------------
    # 수집 메타데이터 기록 (출처/수집일/기간 명시 — 재현성 요건)
    # ------------------------------------------------------------------
    meta_path = os.path.join(THIS_DIR, "collection_metadata.txt")
    with open(meta_path, "w", encoding="utf-8") as f:
        f.write("데이터 수집 메타데이터\n")
        f.write("=" * 40 + "\n")
        f.write(f"수집 일시(UTC): {datetime.now(timezone.utc).isoformat()}\n")
        f.write("출처: Yahoo Finance (yfinance 라이브러리)\n")
        f.write("가격 기준: Adj Close (배당 재투자·액면분할 반영)\n")
        f.write(f"공통구간: {COMMON_START} ~ {wide.index.max().date()}\n")
        f.write("티커별 개별 수집 범위:\n")
        for frame in all_frames:
            t = frame["Ticker"].iloc[0]
            f.write(
                f"  - {t}: {frame['Date'].min().date()} ~ {frame['Date'].max().date()} "
                f"({len(frame)}행)\n"
            )
        f.write(
            "\n라이선스/주의: Yahoo Finance 데이터는 비상업적 개인/학습 목적 사용.\n"
        )
    print(f"[fetch] 메타데이터 저장: {meta_path}")


if __name__ == "__main__":
    main()
