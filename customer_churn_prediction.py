import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# 1. Load Dataset
df = pd.read_csv("Churn_Modelling.csv")

print("\nShape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nFirst 5 rows:\n", df.head())



# 2. Clean & Prepare Dataset
# Remove unnecessary columns
df = df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)
print("\nAfter Dropping Irrelevant Columns:\n", df.head())


# Check missing values
print("\nMissing Values:\n", df.isnull().sum())


# 3. Encode Categorical Features
# Label encode "Gender"
le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])

# One-Hot encode "Geography"
df = pd.get_dummies(df, columns=['Geography'], drop_first=True)

print("\nAfter Encoding:\n", df.head())


# 4. Train-Test Split
X = df.drop("Exited", axis=1)
y = df["Exited"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# 5. Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# 6. Train Model (Random Forest)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)


# 7. Evaluation
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


# 8. Feature Importance
importances = model.feature_importances_
features = X.columns

# Create DataFrame
feat_df = pd.DataFrame({
    "Feature": features,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

print("\nFeature Importance:\n", feat_df)


# Plot Feature Importance
plt.figure(figsize=(8,5))
sns.barplot(x="Importance", y="Feature", data=feat_df)
plt.title("Feature Importance for Churn Prediction")
plt.tight_layout()
plt.savefig("images/feature_importance.png")
plt.show()