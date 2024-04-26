from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.stem import WordNetLemmatizer
#import nltk
#nltk.download('wordnet')
from nltk.corpus import stopwords
import math

app = Flask(__name__)

# Load the dataset and preprocess it
df = pd.read_csv("Artificial_Intellifence_Research_Papers.csv")
stop_words = stopwords.words("english")
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = " ".join([word.lower() for word in text.split() if word not in stop_words])
    text = " ".join([lemmatizer.lemmatize(word) for word in text.split()])
    return text

df["abstract_processed"] = df["Abstract"].apply(preprocess_text)
df["combined_features"] = df["abstract_processed"]

# Initialize TF-IDF vectorizer
vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
tfidf_matrix = vectorizer.fit_transform(df["combined_features"])

# Define route for homepage
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        user_query = request.form["query"]
        recommendations = recommend_papers(user_query)
        return render_template("results.html", recommendations=recommendations, query=user_query)
    return render_template("index.html")

# Function to recommend papers based on user query
def recommend_papers(user_query, top_n=20):
    query_vector = vectorizer.transform([user_query])
    similarities = cosine_similarity(query_vector, tfidf_matrix)
    similar_documents_indices = similarities.argsort()[0, :(-top_n - 1):-1]
    recommended_papers = df.iloc[similar_documents_indices]
    return recommended_papers

if __name__ == "__main__":
    app.run(debug=True)