import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import os

# Create model directory if it doesn't exist
os.makedirs("model", exist_ok=True)

# Load dataset
df = pd.read_csv("processed_news.csv", on_bad_lines='skip', engine='python')
df = df.dropna(subset=["Text", "label"])
df['label'] = df['label'].map({"Real": 0, "Fake": 1})

# Prepare data
X = df['Text']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# TF-IDF and model
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = LogisticRegression()
model.fit(X_train_vec, y_train)

# Evaluate
pred = model.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, pred))

# Save model and vectorizer
joblib.dump(model, "model/fake_news_model.joblib")
joblib.dump(vectorizer, "model/tfidf_vectorizer.joblib")
