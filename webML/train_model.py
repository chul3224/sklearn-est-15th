import pandas as pd
import os
import joblib

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
from sklearn.metrics import accuracy_score
import seaborn as sns

# 데이터 로드
csv_path = 'data/titanic/train.csv'
if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    print(f"Loaded data from {csv_path}")
else:
    print("Local data not found, loading from seaborn...")
    df = sns.load_dataset('titanic')

df.columns = [c.lower() for c in df.columns]
features = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']
target = 'survived'
    
df = df[features + [target]].copy()

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

full_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', voting_model)
])

# 학습
full_pipeline.fit(X, y)
print("Pipeline trained.")

# 저장
os.makedirs('webML', exist_ok=True)
joblib.dump(full_pipeline, 'webML/titanic_voting_pipeline.pkl')
print("Pipeline saved to webML/")
