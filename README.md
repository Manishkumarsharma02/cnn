import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Download NLTK data
nltk.download('stopwords')

# Load the dataset
def load_data(filepath):
    data = pd.read_csv(filepath, delimiter='\t', header=None, names=['label', 'message'])
    return data

# Text preprocessing
def preprocess_text(text):
    # Remove non-alphabetic characters
    text = re.sub('[^a-zA-Z]', ' ', text)
    # Convert to lowercase
    text = text.lower()
    # Tokenize
    tokens = text.split()
    # Remove stopwords and apply stemming
    ps = PorterStemmer()
    tokens = [ps.stem(word) for word in tokens if not word in set(stopwords.words('english'))]
    return ' '.join(tokens)

def extract_features(messages):
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(messages).toarray()
    return X, vectorizer



def train_and_evaluate(X_train, X_test, y_train, y_test):
    # Naive Bayes
    nb = MultinomialNB()
    nb.fit(X_train, y_train)
    y_pred_nb = nb.predict(X_test)
    print("Naive Bayes Classifier:")
    print("Accuracy:", accuracy_score(y_test, y_pred_nb))
    print("Classification Report:\n", classification_report(y_test, y_pred_nb))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_nb))

  svm = SVC(kernel='linear')
    svm.fit(X_train, y_train)
    y_pred_svm = svm.predict(X_test)
    print("\nSupport Vector Machine Classifier:")
    print("Accuracy:", accuracy_score(y_test, y_pred_svm))
    print("Classification Report:\n", classification_report(y_test, y_pred_svm))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_svm))

def main():
    # Load dataset
    filepath = 'SMSSpamCollection.tsv'  # Update this path to your dataset file
    data = load_data(filepath)

  data['message'] = data['message'].apply(preprocess_text)
    
  X, vectorizer = extract_features(data['message'])
  y = data['label'].map({'ham': 0, 'spam': 1}).astype(int)
    
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
  train_and_evaluate(X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    main()
