
# Titanic Survival Prediction – Fully Optimized Classification

Predict whether a passenger survived the Titanic disaster using a fully optimized machine learning pipeline. This project combines preprocessing, feature engineering, hyperparameter-tuned XGBoost, and deployment via Streamlit for interactive predictions.

---

## 🚀 Features

- **Data Preprocessing**: Handle missing values, scale numerical features, encode categorical features.  
- **Models**: Logistic Regression, Random Forest, XGBoost (hyperparameter tuned).  
- **Evaluation**: Accuracy, ROC-AUC, Confusion Matrix, Classification Report.  
- **Deployment**: Streamlit app for interactive survival prediction.

---

## 🛠️ Tech Stack

- **Python**  
- **Scikit-learn** (preprocessing, models)  
- **XGBoost** (optimized classifier)  
- **Pandas & NumPy** (data handling)  
- **Matplotlib & Seaborn** (visualizations)  
- **Streamlit** (interactive web app)  
- **Joblib** (save & load model/preprocessor)

---



---

## ⚡ Usage

1. Clone the repository:
```bash
git clone <your-github-repo-url>
cd titanic-classification
````

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the Streamlit app:

```bash
streamlit run streamlit_titanic.py
```

4. Input passenger details in the app and get predicted survival probability.

---

## 📈 Evaluation

* **Accuracy**: ~83–85%
* **ROC-AUC**: ~0.88–0.90
* Visualizations include confusion matrix and feature importance plots.

---

## 💡 Notes

* Hyperparameter tuning performed with GridSearchCV for optimal performance.
* Model, scaler, and preprocessing pipeline are saved for deployment.
* Streamlit app allows interactive predictions for user-provided inputs.

---

## 📚 References

* [Kaggle Titanic Dataset](https://www.kaggle.com/c/titanic)
* [Scikit-learn Documentation](https://scikit-learn.org/stable/)
* [XGBoost Documentation](https://xgboost.readthedocs.io/en/stable/)
* [Streamlit Documentation](https://docs.streamlit.io/)

```

---


