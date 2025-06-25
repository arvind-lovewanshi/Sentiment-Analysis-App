from flask import Flask, request, jsonify, send_file, render_template
import re
from io import BytesIO
import pandas as pd
import pickle
import base64
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import matplotlib.pyplot as plt

STOPWORDS = set(stopwords.words("english"))

app = Flask(__name__)

# Home page with HTML form
@app.route("/", methods=["GET"])
def home():
    return render_template("landing.html")


# Route used by Streamlit/JSON or bulk file upload
@app.route("/predict", methods=["POST"])
def predict():
    predictor = pickle.load(open(r"Models/model_xgb.pkl", "rb"))
    scaler = pickle.load(open(r"Models/scaler.pkl", "rb"))
    cv = pickle.load(open(r"Models/countVectorizer.pkl", "rb"))

    try:
        if "file" in request.files:
            file = request.files["file"]
            data = pd.read_csv(file)
            predictions, graph = bulk_prediction(predictor, scaler, cv, data)

            response = send_file(
                predictions,
                mimetype="text/csv",
                as_attachment=True,
                download_name="Predictions.csv",
            )

            response.headers["X-Graph-Exists"] = "true"
            response.headers["X-Graph-Data"] = base64.b64encode(graph.getbuffer()).decode("ascii")
            return response

        elif request.is_json:
            content = request.get_json()
            if "text" in content:
                text_input = content["text"]
                predicted_sentiment = single_prediction(predictor, scaler, cv, text_input)
                return jsonify({"prediction": predicted_sentiment})
            else:
                return jsonify({"error": "Missing 'text' field"}), 400

        else:
            return jsonify({"error": "Invalid input format"}), 400

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# New route to support HTML form
@app.route("/predict-text", methods=["POST"])
def predict_text():
    try:
        text_input = request.form["text"]
        predictor = pickle.load(open(r"Models/model_xgb.pkl", "rb"))
        scaler = pickle.load(open(r"Models/scaler.pkl", "rb"))
        cv = pickle.load(open(r"Models/countVectorizer.pkl", "rb"))

        prediction = single_prediction(predictor, scaler, cv, text_input)
        return render_template("landing.html", prediction=prediction)

    except Exception as e:
        return render_template("landing.html", prediction=f"Error: {str(e)}")


def single_prediction(predictor, scaler, cv, text_input):
    corpus = []
    stemmer = PorterStemmer()
    review = re.sub("[^a-zA-Z]", " ", text_input)
    review = review.lower().split()
    review = [stemmer.stem(word) for word in review if word not in STOPWORDS]
    review = " ".join(review)
    corpus.append(review)

    X_prediction = cv.transform(corpus).toarray()
    X_prediction_scl = scaler.transform(X_prediction)
    y_predictions = predictor.predict_proba(X_prediction_scl)
    y_predictions = y_predictions.argmax(axis=1)[0]

    return "Positive" if y_predictions == 1 else "Negative"


def bulk_prediction(predictor, scaler, cv, data):
    corpus = []
    stemmer = PorterStemmer()
    for i in range(0, data.shape[0]):
        review = re.sub("[^a-zA-Z]", " ", data.iloc[i]["Sentence"])
        review = review.lower().split()
        review = [stemmer.stem(word) for word in review if word not in STOPWORDS]
        review = " ".join(review)
        corpus.append(review)

    X_prediction = cv.transform(corpus).toarray()
    X_prediction_scl = scaler.transform(X_prediction)
    y_predictions = predictor.predict_proba(X_prediction_scl)
    y_predictions = y_predictions.argmax(axis=1)
    y_predictions = list(map(sentiment_mapping, y_predictions))

    data["Predicted sentiment"] = y_predictions
    predictions_csv = BytesIO()
    data.to_csv(predictions_csv, index=False)
    predictions_csv.seek(0)

    graph = get_distribution_graph(data)
    return predictions_csv, graph


def get_distribution_graph(data):
    fig = plt.figure(figsize=(5, 5))
    colors = ("green", "red")
    wp = {"linewidth": 1, "edgecolor": "black"}
    tags = data["Predicted sentiment"].value_counts()
    explode = (0.01, 0.01)

    tags.plot(
        kind="pie",
        autopct="%1.1f%%",
        shadow=True,
        colors=colors,
        startangle=90,
        wedgeprops=wp,
        explode=explode,
        title="Sentiment Distribution",
        xlabel="",
        ylabel="",
    )

    graph = BytesIO()
    plt.savefig(graph, format="png")
    plt.close()

    return graph


def sentiment_mapping(x):
    return "Positive" if x == 1 else "Negative"


if __name__ == "__main__":
    app.run(port=5000, debug=True)
