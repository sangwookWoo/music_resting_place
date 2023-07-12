챗봇을 통해 AI와 공감 형태의 대화를 진행하고, 그 속에서 추출된 감성분석을 통해 감성과 유사한/감성과 반대의 음악을 추천해주는 웹사이트를 제작.
## 회고록
https://dataengineerstudy.tistory.com/79


## 기획 의도
기존 음악어플의 플레이리스트/순위 및 장르 추천에서 벗어나 사용자의 개인 감정에 기반한 새로운 음악 추천 모델 구현

## 데이터셋
### 챗봇
웰니스 대화 스크립트 :
https://aihub.or.kr/aihubdata/data/view.do?currMenu=120&topMenu=100&aihubDataSe=extrldata&dataSetSn=267

### 감성분석
감성말뭉치 : https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100 \
노년층 대상 감성 분류 모델(CSV 데이터) : https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100

### 음악 추천 알고리즘
멜론 음악 가사 데이터(셀레니움 기반 크롤링)
노래별 유튜브 링크 데이터(셀레니움 기반 크롤링)

### 팀구성 및 역할(TEAM 음악쉼표)
우상욱: 웹개발(streamlit 프레임워크 활용), 챗봇 개발, ML/DL 모델링, 자연어 데이터 가공(Okt 기반 불용어처리, 토크나이징)\
황도희: 챗봇 개발, 노래별 유튜브 링크 데이터 크롤링, ML 모델링, 자연어 데이터 가공(Okt 기반 품사추출, 토크나이징)\
민병창: DL 모델링, 자연어 데이터 가공(subword tokenizer 기반 토크나이징), 음악 추천 알고리즘 개발\
서영호: 자연어 데이터 가공 모델 -> 활용 데이터 적용 및 전처리, ML 모델링, ML/DL 모델 결과 비교분석\
신제우: DL 모델링(KoBERT), 노래 가사 데이터 크롤링 및 영문-> 한글 변환 자동화 프로그램 개발

## 기술스택(라이브러리 위주 기술)
### 챗봇
데이터 가공 : PANDAS, NUMPY, TORCH

### 모델링
SENTENCE TRANSFORMER(HUGGING FACE API)

### 감성분석
데이터 가공 : PANDAS, NUMPY, KONLPY, SENTENCEPIECE
모델링 : SKLEARN, CATBOOST, XGBOOST, LGBM , TENSORFLOW

### 음악 추천 알고리즘
NUMPY, TORCH

### 웹구현
STREAMLIT

### 데이터베이스(POSTGRESQL)
PSYCOPG2

