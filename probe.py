import pandas as pd
import seaborn as sns
import streamlit 
from sklearn.datasets import load_iris


iris = load_iris()
X=iris.data 
y= iris.target 
feature_names = iris.feature_names
target_names = iris.target_names
df = pd.DataFrame(data=X, columns=feature_names)
df['species'] = y
df['species_name'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

print("Features (first 5 rows):\n", df.head())
print("\nTarget Names:", target_names)



