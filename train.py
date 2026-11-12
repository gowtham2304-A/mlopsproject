
import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)


print("Loading dataset...")

data = pd.read_excel(
    "dataset/default of credit card clients.xls",
    header=1
)

print("Dataset loaded successfully!")
print("Dataset shape:", data.shape)

print("\nFirst five records:")
print(data.head())

print("\nColumn names:")
print(data.columns.tolist())


print("\nMissing values:")
print(data.isnull().sum().sum())

print("\nDuplicate records:")
print(data.duplicated().sum())

target = "default payment next month"

print("\nTarget distribution:")
print(data[target].value_counts())

data[target].value_counts().sort_index().plot(
    kind="bar",
    title="Credit Card Default Distribution"
)

plt.xlabel("Default (0 = No, 1 = Yes)")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.show()


data = data.drop_duplicates()

data = data.dropna(subset=[target])

X = data.drop(columns=[target, "ID"])
y = data[target].astype(int)

X = X.apply(pd.to_numeric, errors="coerce")

feature_names = X.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))


imputer = SimpleImputer(strategy="median")

X_train = pd.DataFrame(
    imputer.fit_transform(X_train),
    columns=feature_names
)

X_test = pd.DataFrame(
    imputer.transform(X_test),
    columns=feature_names
)


scaler = StandardScaler()

X_train = pd.DataFrame(
    scaler.fit_transform(X_train),
    columns=feature_names
)

X_test = pd.DataFrame(
    scaler.transform(X_test),
    columns=feature_names
)

print("\nPreprocessing completed!")


models = {
    "Decision Tree": DecisionTreeClassifier(
        random_state=42,
        class_weight="balanced"
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    ),

    "SVM": SVC(
        probability=True,
        random_state=42,
        class_weight="balanced"
    )
}


results = []

best_model = None
best_model_name = ""
best_f1 = -1

for name, model in models.items():

    print("\nTraining", name, "...")

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    precision = precision_score(
        y_test, predictions, zero_division=0
    )

    recall = recall_score(
        y_test, predictions, zero_division=0
    )

    f1 = f1_score(
        y_test, predictions, zero_division=0
    )

    probabilities = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, probabilities)

    print("\nModel:", name)
    print("Accuracy:", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1 Score:", round(f1, 4))
    print("ROC-AUC:", round(auc, 4))

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        predictions,
        zero_division=0
    ))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC-AUC": auc
    })

    if f1 > best_f1:
        best_f1 = f1
        best_model = model
        best_model_name = name


comparison = pd.DataFrame(results)

print("\nMODEL COMPARISON")
print(comparison.round(4).to_string(index=False))

print("\nBest Model:", best_model_name)
print("Best F1 Score:", round(best_f1, 4))


os.makedirs("models", exist_ok=True)

joblib.dump(
    {
        "model": best_model,
        "imputer": imputer,
        "scaler": scaler,
        "features": feature_names,
        "model_name": best_model_name
    },
    "models/best_model.pkl"
)

print("\nModel saved successfully!")
print("Saved at: models/best_model.pkl")

print("\nTRAINING COMPLETED!")
