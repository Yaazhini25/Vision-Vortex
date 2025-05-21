from flask import Flask, request, jsonify
import joblib
import os

app = Flask(__name__)

model = joblib.load("model/fake_news_model.joblib")
vectorizer = joblib.load("model/tfidf_vectorizer.joblib")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "Text is required"}), 400

    vectorized = vectorizer.transform([text])
    prediction = model.predict(vectorized)[0]
    prob = model.predict_proba(vectorized)[0][prediction]

    return jsonify({
        "label": "Fake" if prediction else "Real",
        "confidence": round(prob, 2)
    })

if __name__ == "__main__":
    app.run(debug=True)
