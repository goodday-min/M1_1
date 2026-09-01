# SPY · QQQ · QLD 장기 수익률 비교 분석

시계열 데이터 분석 미션 — S&P500(SPY) · 나스닥100(QQQ) · 나스닥100 2배 레버리지(QLD) ETF의
장기 수익률·리스크 비교. 상세 배경/질문/방법론은 [`PRD.md`](PRD.md) 참고.

**이 분석은 "분석 주제 모음" 중 주제 1(장기 복리 수익률)·2(위기 구간 낙폭)·3(레버리지 변동성 손실)·
4(적립식 vs 거치식)를 통합해서 다루고, 5(금리 사이클)도 부분적으로 다룬다.**
자세한 대응표는 [`REPORT.md`](REPORT.md)의 "1-1. 분석 주제 모음 대응표" 참고.


## 폴더 구조

```
.
├── data/
│   ├── fetch_data.py        # yfinance로 SPY/QQQ/QLD 데이터 수집하는 스크립트
│   ├── raw/                 # (실행 후 생성) 티커별 전체 역사 원본 CSV
│   ├── common_period.csv    # (실행 후 생성) 공통구간(2006-06-21~) 정렬본
│   └── collection_metadata.txt  # (실행 후 생성) 수집일시/출처/기간 기록
├── src/
│   ├── metrics.py            # 정규화·CAGR·MDD·롤링변동성·STL·베이스라인예측 등 함수 모음
│   └── test_metrics.py       # metrics.py 단위 테스트 (합성 데이터 기반)
├── notebooks/
│   └── analysis.ipynb        # 데이터 로드~정제~분석~시각화~STL~베이스라인예측 전체 파이프라인
├── build_notebook.py         # analysis.ipynb를 코드로 생성하는 빌더 (nbformat 없이 동작)
├── run_stl.py                 # STL 분해만 독립적으로 실행하는 보조 스크립트
├── dashboard.py               # 보너스 A: Streamlit 인터랙티브 대시보드
├── REPORT.md                  # 분석 리포트 (주제/질문/데이터/시각화/인사이트/결론/AI 사용 로그)
├── figures/                    # analysis.ipynb 실행 시 생성되는 차트 PNG
├── requirements.txt
└── README.md
```

## 실행 방법

### 1. 의존성 설치

```bash
pip install -r requirements.txt
```

### 2. 데이터 수집

```bash
python data/fetch_data.py
```

- Yahoo Finance(`yfinance`)에서 SPY/QQQ/QLD의 Adj Close 등 일별 데이터를 받아
  `data/raw/{spy,qqq,qld}.csv`로 저장하고, 공통구간(QLD 상장일 2006-06-21 ~ 오늘)을
  정렬한 `data/common_period.csv`를 생성합니다.
- 인터넷 연결이 필요합니다. (참고: 이 코드는 Claude Cowork의 클라우드 작업 환경에서는
  네트워크 정책상 실행할 수 없어, 로컬 컴퓨터에서 실행하도록 준비되었습니다.)

### 3. 분석 노트북 실행

```bash
jupyter notebook notebooks/analysis.ipynb
```

`notebooks/analysis.ipynb`를 처음 셀부터 순서대로 실행하면 다음을 순서대로 수행합니다.

1. 데이터 로드 및 기본 정보 확인(기간/컬럼/결측치)
2. 데이터 정제 (forward-fill, 이상치 ±3표준편차 플래그)
3. 정규화 누적수익률(로그스케일), 이동평균(50/200일, 골든/데드크로스),
   낙폭(Drawdown) underwater 차트 및 MDD, 롤링 변동성, QLD 실제 vs QQQ×2 이론값 괴리,
   연도별 수익률 비교, 적립식(DCA) vs 거치식(Lump-sum) 비교(주제 4) — 총 7개 시각화 생성
   (`figures/` 폴더에 PNG 저장)
4. 보너스 B-A: QQQ 월별 수익률 STL 분해 (추세/계절성/잔차)
5. 보너스 B-B: 베이스라인 예측(나이브/이동평균/선형추세) 및 MAE 비교

### 4. 단위 테스트 (선택)

`src/metrics.py`의 계산 로직만 별도로 검증하려면:

```bash
cd src && python3 test_metrics.py
```

합성(랜덤워크) 데이터로 CAGR, MDD, 롤링 변동성, 레버리지 괴리, 적립식·거치식 비교,
베이스라인 예측 등 각 함수의 정확성을 확인합니다. (`statsmodels`가 없으면 STL 테스트만 건너뜁니다.)

## 대시보드 실행 (보너스 A)

```bash
pip install streamlit
streamlit run dashboard.py
```

- 사이드바에서 비교할 티커(SPY/QQQ/QLD), 기간 모드(공통구간 vs 개별 최대구간),
  날짜 범위를 선택할 수 있습니다.
- 선택 구간의 누적수익률/CAGR/연율화 변동성/MDD 통계표와 지표 카드,
  정규화 누적수익률(로그스케일) 차트, Drawdown underwater 차트를 보여줍니다.
- QQQ와 QLD를 모두 선택하면 "QLD vs QQQ×2 이론값 괴리" 차트를 펼쳐볼 수 있습니다.
- 사전 조건: `data/fetch_data.py` 실행으로 `data/raw/*.csv`, `data/common_period.csv`가
  이미 생성되어 있어야 합니다.

## 데이터 출처 및 라이선스

- 출처: [Yahoo Finance](https://finance.yahoo.com) (`yfinance` 파이썬 라이브러리 경유)
- 가격 기준: Adj Close (배당 재투자·액면분할 반영, 총수익 기준)
- 비상업적 개인/학습 목적으로만 사용

## 남은 작업

- [x] `data/fetch_data.py` 실행 후 실제 데이터로 `analysis.ipynb` 재실행
- [x] §5 인사이트 섹션을 실제 수치 기반 "관찰 → 해석" 구조로 완성
- [x] `REPORT.md` 작성 (분석 주제/질문/데이터 설명/시각화/인사이트/결론·한계/보너스 결과/AI 사용 로그)
- [x] 보너스 A: `dashboard.py` (Streamlit 대시보드) 구현 및 로직 검증 (4개 시나리오 모킹 테스트 통과)
- [ ] GitHub 저장소 정리 및 제출

## STL 분해 (보너스 B-A) 별도 실행

`analysis.ipynb`의 STL 셀은 `statsmodels`가 필요합니다. 이미 실행하셨다면 그대로 넘어가면 되고,
노트북 전체를 다시 돌리지 않고 이 부분만 확인하려면:

```bash
pip install statsmodels
python run_stl.py
```

`data/common_period.csv`를 자동으로 찾아 `stl_decomposition.png`를 생성합니다.
