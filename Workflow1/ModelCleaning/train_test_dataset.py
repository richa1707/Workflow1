import os
from pandas import read_csv
from joblib import dump
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

workspace = os.getenv('GITHUB_WORKSPACE')
model_cleaning_dir = os.path.join(workspace, 'ModelCleaning')
csv_file_path = os.path.join(model_cleaning_dir, 'cleaned_data.csv')

if os.path.exists(csv_file_path):
    print(f"File found: {csv_file_path}")
else:
    print(f"File not found at: {csv_file_path}")

df = read_csv(csv_file_path)

print(df.head())

X = df["Age"].values.reshape(-1, 1)
y = df["Salary"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

dump(model, "AgeSalaryModel.pkl")
print("Model saved as 'AgeSalaryModel.pkl'")

