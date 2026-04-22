import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

print("Step 1: Loading data...")

fake = pd.read_csv("Fake.csv", encoding="utf-8", on_bad_lines="skip")
true = pd.read_csv("True.csv", encoding="utf-8", on_bad_lines="skip")

print("Step 2: Adding labels...")

fake["label"] = 0
true["label"] = 1

data = pd.concat([fake, true])
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

data = data[["text", "label"]]

print("Step 3: Splitting data...")

X = data["text"]
y = data["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print("Step 4: Vectorizing...")

vectorizer = TfidfVectorizer(stop_words="english")
X_train_vec = vectorizer.fit_transform(X_train)

print("Step 5: Training model...")

model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

print("Step 6: Saving model...")

pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("✅ DONE — files saved!")