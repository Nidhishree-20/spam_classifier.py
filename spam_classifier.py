import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Sample dataset
data = {
    "message": [
        "Congratulations you won a free lottery",
        "Call me when you reach home",
        "Win money now click this link",
        "Let's meet for lunch tomorrow",
        "Free entry in a prize contest",
        "Are you coming to class today"
    ],
    "label": ["spam", "ham", "spam", "ham", "spam", "ham"]
}

df = pd.DataFrame(data)

# Convert text to numbers
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df["message"])

y = df["label"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Test accuracy
predictions = model.predict(X_test)
print("Model Accuracy:", accuracy_score(y_test, predictions))

# Test with new message
new_message = ["Congratulations you have won a prize"]
new_vector = vectorizer.transform(new_message)

prediction = model.predict(new_vector)

print("Message:", new_message[0])
print("Prediction:", prediction[0])