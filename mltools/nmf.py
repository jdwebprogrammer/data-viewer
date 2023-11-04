import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
import argparse

class NMFTopicModel:
    def __init__(self, n_topics=5, max_features=1000):
        self.n_topics = n_topics
        self.max_features = max_features
        self.vectorizer = TfidfVectorizer(max_features=self.max_features, stop_words='english')
        self.nmf_model = NMF(n_components=self.n_topics, random_state=42)

    def fit_transform(self, documents):
        # Vectorize the text data
        document_term_matrix = self.vectorizer.fit_transform(documents)
        
        # Fit the NMF model
        self.nmf_model.fit(document_term_matrix)
        
        return document_term_matrix, self.nmf_model

    def display_topics(self, feature_names, n_top_words=10):
        for topic_idx, topic in enumerate(self.nmf_model.components_):
            print(f"Topic {topic_idx + 1}:")
            print(" ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]))
            print()

def main():
    parser = argparse.ArgumentParser(description='NMF Topic Modeling')
    parser.add_argument('--n_topics', type=int, default=5, help='Number of topics (default is 5)')
    parser.add_argument('--max_features', type=int, default=1000, help='Maximum number of features (default is 1000)')
    args = parser.parse_args()

    documents = ["Machine learning is a subfield of artificial intelligence.",
                "Natural language processing is used in text analysis.",
                "Deep learning models require large datasets.",
                "Topic modeling helps discover hidden themes in text data."]
    nmf_topic_model = NMFTopicModel(n_topics=args.n_topics, max_features=args.max_features)
    document_term_matrix, nmf_model = nmf_topic_model.fit_transform(documents)
    feature_names = nmf_topic_model.vectorizer.get_feature_names_out()
    nmf_topic_model.display_topics(feature_names)

if __name__ == "__main__":
    main()