# Scikit-Learn Machine Learning Study (sklearn-est15th)

이 리포지토리는 Scikit-Learn을 활용한 머신러닝 학습 및 실습 코드를 담고 있습니다. 기본적인 데이터 전처리부터 모델 학습, 평가, 그리고 하이퍼파라미터 튜닝(Optuna)과 웹 배포까지 머신러닝의 전 과정을 단계별로 학습할 수 있도록 구성되어 있습니다.

## 📂 폴더 및 파일 구성

### 📚 정규 학습 커리큘럼 (Step-by-Step)
학습 단계에 맞춰 순서대로 진행할 수 있는 노트북 파일들입니다.

| 순서  | 파일명                           | 주제 및 설명                                               |
| :---: | :------------------------------- | :--------------------------------------------------------- |
|  01   | `1_sklearn_start.ipynb`          | Scikit-Learn 기초, 데이터 로드 및 워크플로우 소개          |
|  02   | `2_ModelSelection.ipynb`         | 데이터셋 분할(Train/Test/Val), 교차 검증(Cross Validation) |
|  03   | `3_SVM.ipynb`                    | Support Vector Machine (SVM) 알고리즘 이해 및 실습         |
|  04   | `4_sklearn_PreProcess.ipynb`     | 데이터 전처리 (Scaling, Encoding, Missing Values 등)       |
|  05   | `5_sklearn_classification.ipynb` | 다양한 분류 알고리즘 비교 및 실습                          |
|  06   | `6_classification_Optuna.ipynb`  | **Optuna**를 활용한 하이퍼파라미터 자동 튜닝 (분류)        |
|  07   | `7_Titanic.ipynb`                | **[Project]** 타이타닉 생존자 예측 실전 분석               |
|  08   | `8_polynominal_Feature.ipynb`    | 다항 회귀 (Polynomial Features) 및  Feature Engineering    |
|  09   | `9_LinearRegressionModel.ipynb`  | 선형 회귀(Linear Regression) 심화 및 규제(Ridge, Lasso)    |
|  10   | `10_ensemble.ipynb`              | 앙상블 학습 (Voting, Bagging, Random Forest, Boosting)     |
|  11   | `11_ensemble_Optuna.ipynb`       | 앙상블 모델의 성능 최적화 (Optuna)                         |
|  13   | `13_unsupervisedLearning.ipynb`  | 비지도 학습 (K-Means, PCA 차원 축소 등)                    |

### ➕ 심화 및 추가 실습 (Plus Series)
다양한 데이터셋을 활용한 추가 예제 및 심화 내용입니다.
- **Wine Analysis**: `Plus_1_sklearn_wine_classification.ipynb`, `Plus_2_Red_wine_quality_analysis.ipynb`, `Plus_4_sklearn_red_wine_quality.ipynb`
- **Digits & Others**: `Plus_3_sklearn_digits.ipynb`
- **Regression Advanced**: `Plus_5_LinearRegression_polyNorm.ipynb`, `Plus_6_LinearRegressionModel.ipynb`
- **Ensemble Advanced**: `Plus_7_ensemble.ipynb`

### 🏠 실전 프로젝트 예제
- **California House Price**: 캘리포니아 주택 가격 예측 (`california_house_price_prediction.ipynb` 시리즈)
- **Data Collection**: `Kaggle데이터셋_다운로드_NEW.ipynb`, `geocoding_kakao.py`

### 🌐 Web Integration (`webML/`)
학습된 머신러닝 모델을 웹 애플리케이션으로 배포하는 예제입니다.
- `web_app.py`: Streamlit 등을 활용한 모델 서빙 예제

### 🤖 AutoML (`AutoML/`)
자동화된 머신러닝(AutoML) 관련 연구 및 자료 폴더입니다.

## 🛠️ 사용 기술 (Tech Stack)
- **Language**: Python 3.x
- **Libraries**:
  - `scikit-learn`: 핵심 머신러닝 라이브러리
  - `pandas`, `numpy`: 데이터 처리 및 분석
  - `matplotlib`, `seaborn`: 데이터 시각화
  - `optuna`: 하이퍼파라미터 최적화
  - `folium`: 지도 시각화 (`folium_visualization_colored.ipynb`)

## 🚀 시작하기
1. **리포지토리 클론**
   ```bash
   git clone https://github.com/venture21/sklearn-est15th.git
   ```
2. **필요 패키지 설치**
   필요한 라이브러리를 설치합니다.
   ```bash
   pip install scikit-learn pandas numpy matplotlib seaborn optuna
   ```
3. **Google Antigravity 사용**
   Google Antigravity 툴을 활용하여 코드를 분석하고 실습을 진행합니다.
