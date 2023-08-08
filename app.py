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

def LemToken(tokens):
    return [lemmer.lemmatize(token) for token in tokens]
remove_punct_dict = dict((ord(punct),None) for punct in string.punctuation)

def LemNormalize(text):
    return LemToken(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

greeting_inputs = [
    "Hello!",
    "Hi there!",
    "Hey!",
    "Hi!",
    "Hello there!",
    "Hey there!"
]

greeting_responses = [
    
]