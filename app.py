import nltk
import random
import string
import warnings
warnings.filterwarnings('ignore')

f = open("information.txt","r")
raw = f.read()
raw = raw.lower()

sent_tokens = nltk.sent_tokenize(raw) #converts to list of sentences
word_tokens = nltk.word_tokenize(raw) #converts to list of words

sentToken = sent_tokens[:4]
print(sentToken)
wordToken = word_tokens[:4]
print(wordToken)

# Preprocessing
lemmer = nltk.stem.WordNetLemmatizer()
# loved to love

def LemToken(tokens):
    return [lemmer.lemmatize(token) for token in tokens]
remove_punct_dict = dict((ord(punct),None) for punct in string.punctuation)

def LemNormalize(text):
    return LemToken(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

greeting_inputs = [
    "Hello",
    "Hi there",
    "Hey",
    "Hi",
    "Hello there",
    "Hey there"
]

greeting_responses = [
    "Hello! Explore my career highlights and projects.",
    "Hi there! Learn about my professional journey.",
    "Hey! Discover my career and projects.",
    "Hi! Explore my portfolio and career.",
    "Hello! Learn about my work and achievements."
]

def greeting(sentence):
    for word in sentence.split():
        if word.lower() in greeting_inputs:
            return random.choice(greeting_responses)
        
# Vectorizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# When it comes to compare document magnitude doesnt play import role we use cosine_similarity

def response(user_response):
    chatbot_response = ''
    sent_tokens.append(user_response)
    tfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words="english")