# src/train.py
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns as snsplt

# -------------------------
# 1. Load dataset
# -------------------------
df = sns.load_dataset("titanic")  # seaborn built-in
# Select useful columns; seaborn titanic columns are lowercase
df = df[['survived','pclass','sex','age','sibsp','parch','fare','embarked']]

# -------------------------
# 2. Simple cleaning
# -------------------------
# Drop rows with missing target (none in seaborn), keep rows with missing features (imputed later)
print("Dataset shape:", df.shape)
print(df.head())

# -------------------------
# 3. Features & target
# -------------------------
X = df.drop('survived', axis=1)
y = df['survived']

# Define columns
numeric_features = ['age','sibsp','parch','fare']
categorical_features = ['pclass','sex','embarked']

# -------------------------
# 4. Preprocessing pipelines
# -------------------------
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('ohe', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', num_pipeline, numeric_features),
    ('cat', cat_pipeline, categorical_features)
])

# -------------------------
# 5. Train/test split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------
# 6. Model pipelines + hyperparams
# -------------------------
# Logistic Regression
pipe_lr = Pipeline([
    ('pre', preprocessor),
    ('clf', LogisticRegression(max_iter=1000))
])
lr_params = {
    'clf__C': [0.01, 0.1, 1, 10],
    'clf__penalty': ['l2'],
    'clf__solver': ['lbfgs']
}

# Random Forest
pipe_rf = Pipeline([
    ('pre', preprocessor),
    ('clf', RandomForestClassifier(random_state=42, n_jobs=-1))
])
rf_params = {
    'clf__n_estimators': [200, 400],
    'clf__max_depth': [4, 8, None],
    'clf__min_samples_leaf': [1, 2, 4]
}

# XGBoost
pipe_xgb = Pipeline([
    ('pre', preprocessor),
    ('clf', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42, n_jobs=1))
])
xgb_params = {
    'clf__n_estimators': [200, 400],
    'clf__max_depth': [3, 5],
    'clf__learning_rate': [0.01, 0.1],
    'clf__subsample': [0.7, 1.0],
    'clf__colsample_bytree': [0.7, 1.0]
}

# -------------------------
# 7. Helper to grid-search
# -------------------------
def run_grid(pipeline, params, X_tr, y_tr, name):
    print(f"\nRunning GridSearchCV for {name} ...")
    gs = GridSearchCV(pipeline, params, cv=5, scoring='roc_auc', n_jobs=-1, verbose=1)
    gs.fit(X_tr, y_tr)
    print(f"Best {name} params:", gs.best_params_)
    print(f"Best {name} CV ROC AUC: {gs.best_score_:.4f}")
    return gs.best_estimator_

# Run optimizations (you can comment/uncomment models you don't want to run)
best_lr = run_grid(pipe_lr, lr_params, X_train, y_train, "LogisticRegression")
best_rf = run_grid(pipe_rf, rf_params, X_train, y_train, "RandomForest")
best_xgb = run_grid(pipe_xgb, xgb_params, X_train, y_train, "XGBoost")

# -------------------------
# 8. Evaluate models on test set
# -------------------------
def evaluate(model, X_te, y_te, name):
    proba = model.predict_proba(X_te)[:,1]
    pred = (proba >= 0.5).astype(int)
    print(f"\n{name} Test ROC AUC: {roc_auc_score(y_te, proba):.4f}")
    print(classification_report(y_te, pred))
    cm = confusion_matrix(y_te, pred)
    snsplt.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"{name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

evaluate(best_lr, X_test, y_test, "LogisticRegression")
evaluate(best_rf, X_test, y_test, "RandomForest")
evaluate(best_xgb, X_test, y_test, "XGBoost")

# -------------------------
# 9. Pick best model by ROC AUC on test
# -------------------------
def roc_on_test(model):
    return roc_auc_score(y_test, model.predict_proba(X_test)[:,1])

scores = {
    'LogisticRegression': roc_on_test(best_lr),
    'RandomForest': roc_on_test(best_rf),
    'XGBoost': roc_on_test(best_xgb)
}
print("\nTest ROC AUC scores:", scores)

best_name = max(scores, key=scores.get)
best_model = {'LogisticRegression': best_lr, 'RandomForest': best_rf, 'XGBoost': best_xgb}[best_name]
print(f"\nSelected best model: {best_name} with ROC AUC = {scores[best_name]:.4f}")

# -------------------------
# 10. Save preprocessor and model artifacts (for deployment)
# -------------------------
# Save the fitted preprocessor and the full pipeline? We saved full pipeline inside best_model already,
# but to make a lightweight app we'll save preprocessor and classifier separately.
# Extract fitted preprocessor and classifier from pipeline
fitted_preprocessor = best_model.named_steps['pre']
fitted_clf = best_model.named_steps['clf']

joblib.dump(fitted_preprocessor, "../app/preprocessor.pkl")
joblib.dump(fitted_clf, "../app/model.pkl")

print("\nSaved preprocessor and model to app/ directory.")
