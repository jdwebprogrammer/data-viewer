import gensim.downloader as api
from gensim.models import Word2Vec
import numpy as np
import argparse

class WordEmbeddings:
    def __init__(self, embedding_model_name='word2vec-google-news-300'):
        self.embedding_model_name = embedding_model_name
        self.embedding_model = None
        self.embedding_dim = None
        self.load_embedding_model()

    def load_embedding_model(self):
        try:
            self.embedding_model = api.load(self.embedding_model_name)
            self.embedding_dim = self.embedding_model.vector_size
        except Exception as e:
            raise Exception(f"Error loading the embedding model: {e}")

    def get_embedding(self, word):
        if self.embedding_model is not None:
            if word in self.embedding_model.wv:
                return self.embedding_model.wv[word]
            else:
                return np.zeros(self.embedding_dim)  # Return zeros for out-of-vocabulary words
        else:
            raise ValueError("Embedding model has not been loaded yet.")

    def get_similarity(self, word1, word2):
        if self.embedding_model is not None:
            return self.embedding_model.wv.similarity(word1, word2)
        else:
            raise ValueError("Embedding model has not been loaded yet.")

"""
def main():
    parser = argparse.ArgumentParser(description='Word Embeddings')
    parser.add_argument('--embedding_model_name', type=str, default='word2vec-google-news-300', help='Embedding model name (default is word2vec-google-news-300)')
    parser.add_argument('--word', type=str, default='king', help='Word for which to get embedding')
    parser.add_argument('--word1', type=str, default='king', help='First word for similarity calculation')
    parser.add_argument('--word2', type=str, default='queen', help='Second word for similarity calculation')
    args = parser.parse_args()

    word_embeddings = WordEmbeddings()

    # Get the word embedding for a word
    embedding = word_embeddings.get_embedding(args.word)

    # Calculate similarity between two words
    similarity = word_embeddings.get_similarity(args.word1, args.word2)

    print(f"Embedding for '{args.word}':\n{embedding}")
    print(f"Similarity between '{args.word1}' and '{args.word2}': {similarity:.2f}")

if __name__ == "__main__":
    main()
"""