import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Setup
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess(text):
    text = re.sub('[^a-zA-Z]', ' ', text).lower()
    words = text.split()
    return ' '.join([stemmer.stem(w) for w in words if w not in stop_words])

# Load & prepare data
df = pd.read_csv('C:\\Users\\ANKIT\\OneDrive\\Desktop\\INFOBYTE\\intrn\\data\\spam.csv', encoding='latin-1')
df = df[['v1', 'v2']].rename(columns={'v1': 'label', 'v2': 'message'})
df['label'] = df['label'].map({'ham': 0, 'spam': 1})
df['message'] = df['message'].apply(preprocess)

# Split
X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.2, random_state=42)

# Pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LogisticRegression(solver='liblinear'))
])

# Train & Evaluate
pipeline.fit(X_train, y_train)
preds = pipeline.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, preds):.4f}")

# Predict new message
def predict_spam(msg):
    cleaned = preprocess(msg)
    pred = pipeline.predict([cleaned])[0]
    return "Spam" if pred else "Ham"

# Test
print(predict_spam("Claim your FREE prize now!"))
print(predict_spam("Hey, are we still on for tomorrow?"))
