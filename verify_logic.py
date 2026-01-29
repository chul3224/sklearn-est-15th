import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
try:
    from autogluon.tabular import TabularDataset, TabularPredictor
    print("AutoGluon import successful.")
except ImportError:
    print("AutoGluon not installed. This is expected if the user hasn't run the !pip install cell yet.")

# Load data
data = fetch_california_housing()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['MedHouseVal'] = data.target

# Split data (use a very small subset for fast verification)
train_data, test_data = train_test_split(df.iloc[:200], test_size=0.2, random_state=42)

print(f"Data verification: Train {train_data.shape}, Test {test_data.shape}")

# Logic check for TabularPredictor params
label = 'MedHouseVal'
eval_metric = 'rmse'
print(f"Params check: label={label}, eval_metric={eval_metric}")
