import gradio as gr
import joblib
import pandas as pd
import numpy as np
import os

# 모델 로드
model_path = 'webML/titanic_voting_pipeline.pkl'

def train_and_save_pipeline():
    print("Training pipeline...")
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.impute import SimpleImputer
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier, VotingClassifier
    from sklearn.svm import SVC
    import seaborn as sns

    # 데이터 로드
    csv_path = 'data/titanic/train.csv'
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        print("Local data not found, loading from seaborn...")
        df = sns.load_dataset('titanic')

    df.columns = [c.lower() for c in df.columns]
    features = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']
    target = 'survived'
    
    X = df[features]
    y = df[target]

    # 파이프라인 구성
    numeric_features = ['pclass', 'age', 'sibsp', 'parch', 'fare']
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_features = ['sex', 'embarked']
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    voting_model = VotingClassifier(
        estimators=[
            ('lr', LogisticRegression(max_iter=1000)),
            ('knn', KNeighborsClassifier()),
            ('dt', DecisionTreeClassifier()),
            ('rf', RandomForestClassifier()),
            ('svc', SVC(probability=True))
        ],
        voting='soft'
    )

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', voting_model)
    ])

    pipeline.fit(X, y)
    
    os.makedirs('webML', exist_ok=True)
    joblib.dump(pipeline, model_path)
    print("Pipeline trained and saved.")
    return pipeline

try:
    pipeline = joblib.load(model_path)
except Exception:
    print("Pipeline not found, training new one...")
    pipeline = train_and_save_pipeline()

def predict_survival(pclass, sex, age, sibsp, parch, fare, embarked):
    if pipeline is None:
        return "Model not loaded."
    
    # 입력 데이터 프레임 생성 (전처리 없이 원본 데이터 형태 입력)
    input_df = pd.DataFrame({
        'pclass': [int(pclass)],
        'sex': [sex],
        'age': [age],
        'sibsp': [sibsp],
        'parch': [parch],
        'fare': [fare],
        'embarked': [embarked]
    })
    
    # 파이프라인을 통해 예측 (전처리 자동 수행)
    try:
        prediction = pipeline.predict(input_df)[0]
        probability = pipeline.predict_proba(input_df)[0][1]
    except Exception as e:
        return f"Prediction Error: {str(e)}"
    
    result = "생존 (Survived)" if prediction == 1 else "사망 (Did not survive)"
    return f"{result}\n생존 확률: {probability:.2%}"

# Gradio 인터페이스 정의
iface = gr.Interface(
    fn=predict_survival,
    inputs=[
        gr.Radio([1, 2, 3], label="객실 등급 (Pclass)", value=3),
        gr.Radio(['male', 'female'], label="성별 (Sex)", value='male'),
        gr.Slider(0, 100, step=1, label="나이 (Age)", value=30),
        gr.Slider(0, 8, step=1, label="형제/자매 수 (SibSp)", value=0),
        gr.Slider(0, 6, step=1, label="부모/자녀 수 (Parch)", value=0),
        gr.Number(label="요금 (Fare)", value=32.2),
        gr.Radio(['S', 'C', 'Q'], label="탑승 항구 (Embarked)", value='S')
    ],
    outputs="text",
    title="타이타닉 생존자 예측 서비스 (Pipeline)",
    description="Sklearn Pipeline을 적용한 모델입니다. 승객 정보를 입력하면 생존 여부를 예측해줍니다."
)

if __name__ == "__main__":
    iface.launch(server_name="0.0.0.0", share=False)
