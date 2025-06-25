# 🧠 Sentiment Analysis 

This is a full-stack **Machine Learning web application** for predicting the **sentiment** of input text — either **Positive** or **Negative**.

The app is powered by **XGBoost**, **NLTK**, and **CountVectorizer**, and provides:

- 📄 Bulk prediction via CSV upload
- 💬 Real-time single sentence prediction
- 📊 Sentiment distribution visualization
- 📥 Option to download results as a CSV

> **Frontend:** Streamlit  
> **Backend:** Flask  
> **Model:** XGBoost Classifier

---

## 🧠 Model Information

- **Model Type:** XGBoost Classifier
- **Text Preprocessing:** NLTK Stopwords, Porter Stemmer
- **Vectorization:** CountVectorizer
- **Feature Scaling:** StandardScaler

### 🔐 Model Artifacts:

- `Models/model_xgb.pkl` – Trained sentiment classification model
- `Models/scaler.pkl` – Feature scaler used during training
- `Models/countVectorizer.pkl` – Vectorizer used for tokenizing text

---

## 📁 Project Structure

```
sentiment-analysis-app/
├── Models/                       # Trained model and preprocessing tools
│   ├── model_xgb.pkl             # Sentiment classification model
│   ├── scaler.pkl                # StandardScaler object
│   └── countVectorizer.pkl       # CountVectorizer object
│
├── templates/
│   └── landing.html              # Optional HTML form (Flask frontend)
│
├── main.py                       # Streamlit UI for frontend interaction
├── api.py                        # Flask backend API for predictions
```

