from flask import Flask, request, jsonify, render_template
import joblib
import os

app = Flask(__name__)

# Load your model and vectorizer
model = joblib.load("model/fake_news_model.joblib")
vectorizer = joblib.load("model/tfidf_vectorizer.joblib")

# Serve the index.html front-end
@app.route("/")
def index():
    return render_template("index.html")  # Make sure this file exists in templates/

# Prediction API endpoint
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
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
