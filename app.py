from flask import Flask, render_template, request, jsonify
from docx import Document
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier  # Import Random Forest Classifier
from sklearn.tree import DecisionTreeClassifier  # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from datetime import datetime
import json

app = Flask(__name__)

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load the Word document
doc = Document("new_one_27date.docx")

# Extract key-value pairs from the document
def extract_key_value_pairs(doc):
    key_value_pairs = {}
    for paragraph in doc.paragraphs:
        if ":" in paragraph.text:
            key, value = paragraph.text.split(":")
            key = key.strip()
            value = value.strip()
            key_value_pairs[key] = value
    return key_value_pairs

key_value_pairs = extract_key_value_pairs(doc)

# Preprocess text
def preprocess_text(text):
    # Tokenization
    tokens = word_tokenize(text)
    # Convert to lowercase and lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word.lower()) for word in tokens]
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    clean_tokens = [word for word in tokens if word not in stop_words]
    # Reconstruct text from tokens
    clean_text = ' '.join(clean_tokens)
    return clean_text

# Preprocess data
def preprocess_data(key_value_pairs):
    cleaned_data = {}
    for question, answer in key_value_pairs.items():
        cleaned_question = preprocess_text(question)
        cleaned_answer = preprocess_text(answer)
        cleaned_data[cleaned_question] = cleaned_answer
    return cleaned_data

clean_data = preprocess_data(key_value_pairs)

# Create TF-IDF vectorizer
vectorizer = TfidfVectorizer()
# Fit and transform the preprocessed data
corpus = list(clean_data.keys())
X = vectorizer.fit_transform(corpus)

# Assign labels to each intent
labels = ['intent'] * len(corpus)  # Replace 'intent' with actual intents

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# Train Random Forest classifier
rf_classifier = RandomForestClassifier()  # Initialize Random Forest Classifier
rf_classifier.fit(X_train, y_train)

# Train Decision Tree classifier
dt_classifier = DecisionTreeClassifier()  # Initialize Decision Tree Classifier
dt_classifier.fit(X_train, y_train)

# Test the models
y_pred_rf = rf_classifier.predict(X_test)
y_pred_dt = dt_classifier.predict(X_test)

# Calculate accuracy for Random Forest classifier
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print("Random Forest Classifier Accuracy:", accuracy_rf)

# Calculate accuracy for Decision Tree classifier
accuracy_dt = accuracy_score(y_test, y_pred_dt)
print("Decision Tree Classifier Accuracy:", accuracy_dt)

# Get response
def get_response(user_query):
    # Preprocess user query
    cleaned_query = preprocess_text(user_query)
    # If no named entities are found, continue with TF-IDF similarity-based response
    query_vector = vectorizer.transform([cleaned_query])
    similarities = cosine_similarity(query_vector, X)
    most_similar_index = np.argmax(similarities)
    most_similar_question = corpus[most_similar_index]
    return clean_data.get(most_similar_question, "")

# Save response to JSON
def save_response_to_json(user_query, response):
    # Define the filename for the JSON file
    json_filename = "chatbot_responses.json"
    # Create a dictionary to represent the user query and response
    data = {
        "timestamp": str(datetime.now()),
        "user_query": user_query,
        "response": response
    }
    # Append the data to the JSON file
    with open(json_filename, 'a') as file:
        file.write(json.dumps(data) + '\n')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_query = request.json['user_query']
    response = get_response(user_query)
    save_response_to_json(user_query, response)
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)
