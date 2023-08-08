import nltk
import random
import string
import warnings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# When it comes to compare document magnitude doesnt play import role we use cosine_similarity

warnings.filterwarnings('ignore')

f = open('information.txt',"r")
raw = f.read()
raw = raw.lower()

sent_tokens = nltk.sent_tokenize(raw) #converts to list of sentences
word_tokens = nltk.word_tokenize(raw) #converts to list of words

sentToken = sent_tokens[:4]
# print(sentToken)
wordToken = word_tokens[:4]
# print(wordToken)
# Preprocessing
lemmer = nltk.stem.WordNetLemmatizer()
# loved to love

def LemToken(tokens):
    return [lemmer.lemmatize(token) for token in tokens]
remove_punct_dict = dict((ord(punct),None) for punct in string.punctuation)

def LemNormalize(text):
    return LemToken(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))
# Grettings
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
chatbot_unaware = [
    "Hello! I'm sorry, but I can't answer that.",
    "Hi there! Unfortunately, I can't provide an answer to that.",
    "Hey! I'm sorry, I can't respond to that.",
    "Hi! I apologize, but I can't answer that question.",
    "Hello! Regrettably, I can't provide an answer to that."
]


def response(user_response):
    chatbot_response = ''  # Initialize an empty string to store the chatbot's response.
    sent_tokens.append(user_response)  # Add the user's response to the list of sentence tokens.
    tfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words="english")
    tfidf = tfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx = vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]

    if req_tfidf == 0:
        chatbot_response = chatbot_response + random.choice(chatbot_unaware)
        return chatbot_response
    else:
        chatbot_response = chatbot_response + sent_tokens[idx]
        return chatbot_response

if __name__=="__main__":
    flag = True
    print("Hello! I am your Portfolio Companion Chatbot created by Rishav.")
    while(flag==True):
        user_response = input()
        user_response = user_response.lower()
        if(user_response !='bye'):
            if user_response == 'thanks' or user_response == 'thank you':
                flag = False
                print("Bot: You' re welcome!")
            else:
                if greeting(user_response)!=None:
                    print("Bot: " + greeting(user_response))
                else:
                    # sent_tokens.append(user_response)
                    print("Bot:",end='')
                    print(response(user_response))
                    sent_tokens.remove(user_response)
        else:
            flag = False
            print("Bot: Bye! Have a nice day")
