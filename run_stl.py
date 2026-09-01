"""
QQQ 월별 수익률 STL 분해 (보너스 B-A) — 독립 실행 스크립트

fetch_data.py를 이미 실행하셨다면 같은 폴더(또는 그 하위 data/)에
common_period.csv가 있을 거예요. 이 스크립트를 그 파일과 같은 위치,
또는 data/common_period.csv 를 찾을 수 있는 위치에서 실행하세요.

실행:
    pip install statsmodels matplotlib   # statsmodels는 이미 설치하셨다고 하셨죠
    python run_stl.py

결과:
    - stl_decomposition.png (원본/추세/계절성/잔차 4단 차트)
    - 콘솔에 계절성/잔차 표준편차 비율 출력
"""

import os
import sys

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL

# common_period.csv 위치 후보 (있는 곳을 자동으로 찾음)
CANDIDATES = [
    "common_period.csv",
    "data/common_period.csv",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "common_period.csv"),
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "common_period.csv"),
]

path = next((p for p in CANDIDATES if os.path.exists(p)), None)
if path is None:
    print("common_period.csv를 찾을 수 없습니다. 이 스크립트와 같은 폴더에 두거나,")
    print("경로를 직접 지정해서 실행하세요: python run_stl.py 경로/common_period.csv")
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        sys.exit(1)

print(f"데이터 로드: {path}")
common = pd.read_csv(path, parse_dates=["Date"]).set_index("Date").sort_index()

# 한글 폰트 자동 감지 (없으면 경고만 출력하고 계속 진행)
import matplotlib.font_manager as fm
_candidates = ["AppleGothic", "Malgun Gothic", "NanumGothic", "NanumBarunGothic",
               "Noto Sans CJK KR", "Noto Sans KR", "Noto Sans CJK JP"]
_available = {f.name for f in fm.fontManager.ttflist}
_chosen = next((f for f in _candidates if f in _available), None)
if _chosen:
    plt.rcParams["font.family"] = _chosen
else:
    print("[경고] 한글 폰트를 찾지 못했습니다. 차트의 한글이 깨져 보일 수 있습니다.")
plt.rcParams["axes.unicode_minus"] = False

# QQQ 월간 수익률(%) 계산
qqq = common["QQQ"]
monthly_price = qqq.resample("ME").last()
monthly_ret = monthly_price.pct_change().dropna() * 100.0
print(f"월간 수익률 데이터 포인트: {len(monthly_ret)}개 "
      f"({monthly_ret.index.min().date()} ~ {monthly_ret.index.max().date()})")

# STL 분해
stl = STL(monthly_ret, period=12, robust=True)
result = stl.fit()

fig = result.plot()
fig.set_size_inches(11, 8)
fig.suptitle("QQQ 월별 수익률 STL 분해 (원본/추세/계절성/잔차)", y=1.02)
out_path = "stl_decomposition.png"
plt.savefig(out_path, dpi=120, bbox_inches="tight")
print(f"차트 저장: {out_path}")

seasonal_amp = result.seasonal.std()
resid_amp = result.resid.std()
ratio = seasonal_amp / resid_amp
print(f"\n계절성 성분 표준편차: {seasonal_amp:.4f}")
print(f"잔차 성분 표준편차: {resid_amp:.4f}")
print(f"계절성/잔차 비율: {ratio:.3f}")
if ratio < 0.5:
    print("-> 계절성이 잔차(노이즈) 대비 뚜렷하지 않음. '뚜렷한 계절성이 없다'는 것 자체가 유효한 결론.")
else:
    print("-> 계절성이 잔차 대비 상당히 뚜렷한 편. 특정 월 패턴이 존재할 가능성 시사.")

# 월별 평균 계절성 성분 (어느 달이 강세/약세인지 참고용)
seasonal_by_month = result.seasonal.groupby(result.seasonal.index.month).mean()
seasonal_by_month.index.name = "월"
print("\n월별 평균 계절성 성분(%):")
print(seasonal_by_month.round(3))
