
import spacy


nlp = spacy.load("en_core_web_sm")

def extract_keywords(input_text):
    doc = nlp(input_text)
    keywords = [token.text for token in doc if token.pos_ == "NOUN"]
    keyword_string = " ".join(keywords)
    return keyword_string



input_text = ""
keywords = extract_keywords(input_text)
print(keywords)