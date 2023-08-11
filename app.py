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
    "hello",
    "hi there",
    "hey",
    "hi",
    "hello there",
    "hey there"
    
]

greeting_responses = [
    "Hello! Explore my career highlights and projects.",
    "Hi there! Learn about my professional journey.",
    "Hey! Discover my career and projects.",
    "Hi! Explore my portfolio and career.",
    "Hello! Learn about my work and achievements."
]
ending_response = [
    "Goodbye! Feel free to reach out whenever you want to chat again.",
    "It's been great talking to you! Until next time.",
    "Farewell for now! Remember, I'm here whenever you need me.",
    "Take care and see you later!",
    "Bye-bye! Don't hesitate to return if you have more questions.",
    "Signing off for now. Have a wonderful day!",
    "Catch you on the flip side! Remember, I'm just a message away.",
    "Adios! Feel free to drop by anytime you want to chat.",
    "So long! Looking forward to our next conversation.",
    "Until we chat again, goodbye and stay awesome!"
]
user_asking = [
    "how are you",
    "how's it going",
    "what's up",
    "how's everything",
    "are you doing well",
    "how are things on your end",
    "are you ready to help",
    "are you available",
    "how are you today"
]

goodbye_keywords = ["bye", "goodbye", "see you", "farewell", "adios", "au revoir", "ciao"]

chatbot_unaware = [
    "I'm sorry, but I can't answer that.",
    "Unfortunately, I can't provide an answer to that.",
    "I'm sorry, I can't respond to that.",
    "I apologize, but I can't answer that question.",
    "Regrettably, I can't provide an answer to that."
]
chatbot_thanks_response = [
    "You're welcome! Feel free to ask more questions.",
    "Glad to assist! If you need more info, ask away.",
    "Happy to help. Have more queries? Just ask.",
    "No problem! Reach out if you want more insights.",
    "My pleasure! Any more questions? Let me know.",
    "You're welcome! More info available on request.",
    "Anytime! Need further assistance? Just ask.",
    "Glad I could assist! Keep the questions coming.",
    "No worries! Want to know more? Just ask."
]

asking_response = [
    "I'm here and ready to assist you!",
    "I'm just a chatbot, but I'm here to help!",
    "I'm functioning as expected and here to provide information.",
    "I'm a machine, so I don't have feelings, but I'm at your service!",
    "I'm here, waiting to answer your questions!",
    "I'm here and fully operational to help with any inquiries.",
    "I don't have feelings, but I'm here and eager to assist you.",
    "I'm here to provide information and support!",
    "I'm a chatbot, so I'm always ready to help you out!",
    "I don't experience emotions, but I'm available to help you.",
    "I'm here, powered up, and ready to answer your queries!",
    "I'm doing well in terms of functionality. How can I assist you today?",
    "I'm here to help, so feel free to ask me anything you need.",
    "I'm ready and equipped to provide you with information.",
    "I'm just a program, but I'm operational and ready to assist you!"
]

def greeting(sentences):
    for i in sentences.split():
        if i.lower() in greeting_inputs:
            return random.choice(greeting_responses)
def ending(sentences):
    for i in sentences.split():
        if i.lower() in goodbye_keywords:
            return "bye"
def asking(sentences):
    for i in sentences.split():
        if i.lower() in user_asking:
            return random.choice(asking_response)

# Vectorizer



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

    if (req_tfidf == 0):
        chatbot_response = chatbot_response + random.choice(chatbot_unaware)
        return chatbot_response
    else:
        chatbot_response = chatbot_response + sent_tokens[idx]
        return chatbot_response

if __name__ == "__main__":
    flag = True
    print("Hello! I am your Portfolio Companion Chatbot created by Rishav.")
    
    while flag:
        user_response = input()
        user_response = user_response.lower()
        
        if ending(user_response) != "bye":
            if user_response == 'thanks' or user_response == 'thank you':
                flag = False
                print(random.choice(chatbot_thanks_response))
                break
            elif asking(user_response) is not None:
                print("Bot: " + asking(user_response))
            else:
                if greeting(user_response) is not None:
                    print("Bot: " + greeting(user_response))
                else:
                    print("Bot:", end='')
                    print(response(user_response))
                    sent_tokens.remove(user_response)
        else:
            flag = False
            print(random.choice(ending_response))
