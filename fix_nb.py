import json
import os

p = r'c:\Users\User\OneDrive\Desktop\Ai Bootcamp\github\datascience\scikit-learn\scikit-learn\Plus_7_ensemble_gemini.ipynb'

if not os.path.exists(p):
    print(f"File not found: {p}")
    exit(1)

with open(p, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Change 1: Uncomment install (Cell 2, Index 1)
nb['cells'][1]['source'] = [
    "# AutoGluon 설치 (필요한 경우 주석 해제 후 실행)\n",
    "!pip install autogluon"
]

# Change 2: Fix eval_metric (Cell 8, Index 7)
nb['cells'][7]['source'] = [
    "label = 'MedHouseVal'\n",
    "eval_metric = 'rmse'  # 회귀 문제 평가 지표 (mse 또는 rmse 사용)"
]

# Change 3: Add time_limit (Cell 10, Index 9)
nb['cells'][9]['source'] = [
    "predictor = TabularPredictor(label=label, eval_metric=eval_metric, path=save_path).fit(\n",
    "    train_data,\n",
    "    hyperparameters=hyperparameters,\n",
    "    num_bag_folds=num_bag_folds,\n",
    "    num_stack_levels=num_stack_levels,\n",
    "    time_limit=600,  # 최대 학습 시간 (초)\n",
    "    fit_weighted_ensemble=True  # Weighted Ensemble (Voting 유사 효과) 활성화\n",
    ")"
]

with open(p, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=4, ensure_ascii=False)

print("Modification complete.")
