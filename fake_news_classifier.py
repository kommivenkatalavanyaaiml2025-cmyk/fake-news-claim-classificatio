
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Sample dataset for demonstration
news_data = [
    "Government announces new education policy",
    "Scientists discover a new planet",
    "Health department releases vaccination guidelines",
    "Breaking news celebrity turns into an alien overnight",
    "Miracle drink cures every disease instantly",
    "Click here to win millions of dollars guaranteed",
    "Local school announces examination schedule",
    "Weather department issues heavy rain warning",
    "Famous actor secretly becomes the president of Mars",
    "Free money available to everyone with no conditions"
]

labels = [
    "REAL", "REAL", "REAL", "FAKE", "FAKE",
    "FAKE", "REAL", "REAL", "FAKE", "FAKE"
]

# Create a machine learning pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        preprocessor=lambda text: re.sub(r"\s+", " ", text.strip())
    )),
    ("classifier", LogisticRegression(max_iter=1000))
])

# Train the model
model.fit(news_data, labels)

print("=== Fake News Claim Classification ===")
print("Type a news claim to classify.")
print("Type 'exit' to stop.\n")

while True:
    claim = input("Enter news claim: ").strip()

    if claim.lower() == "exit":
        print("Program ended.")
        break

    if not claim:
        print("Please enter a news claim.")
        continue

    prediction = model.predict([claim])[0]
    print("Prediction:", prediction)
    print()
