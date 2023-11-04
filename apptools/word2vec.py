from gensim.models import Word2Vec
import nltk

# Sample code data
code_data = [
    "Welcome to the game!"
]

# Preprocess and tokenize the code
def preprocess_code(code):
    # Tokenize the code using NLTK (you can use other tokenizers)
    tokens = nltk.word_tokenize(code)
    return tokens

tokenized_code_data = [preprocess_code(code) for code in code_data]

# Train Word2Vec model
model = Word2Vec(tokenized_code_data, vector_size=3, window=5, min_count=1, sg=0)

# Save the trained Word2Vec model
model.save("code2vec.model")

# Retrieve word embeddings
for wordindex in range(len(model.wv) ):
    
    print(f"Embedding for '{wordindex}': {model.wv} - {model.wv[wordindex]}")