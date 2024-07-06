import numpy as np
import nltk
import random
import string

# Open and read the file
f = open('chatbot.txt', errors='ignore')
raw_doc = f.read().lower()  # Convert text to lowercase

# Download necessary NLTK data files
nltk.download('punkt')
nltk.download('wordnet')

# Tokenize the raw_doc into sentences and words
sent_tokens = nltk.sent_tokenize(raw_doc)
word_tokens = nltk.word_tokenize(raw_doc)

# Initialize the lemmatizer
lemmer = nltk.stem.WordNetLemmatizer()

# Define a function to lemmatize tokens
def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]

# Create a translation table to remove punctuation
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)

# Define a function to normalize text
def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

# Define greeting inputs and responses
GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up", "hey")
GREETING_RESPONSES = ["hi", "hey", "*nods*", "hi there", "hello", "I am glad! You are talking to me"]

# Define a function to respond to greetings
def greeting(sentence):
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Define a function to generate a response
def response(user_response):
    chatbot_response = ''
    sent_tokens.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english', token_pattern=None)
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf[:-1])  # Compare user input only with existing sentences
    idx = vals.argsort()[0][-1]  # Get the index of the most similar sentence
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-1]
    if req_tfidf == 0:
        chatbot_response = chatbot_response + "I am sorry! I don't understand you"
    else:
        chatbot_response = chatbot_response + sent_tokens[idx]
    sent_tokens.pop()  # Remove the user response added for processing
    return chatbot_response

# Initialize the chatbot
flag = True
print("Chatbot: My name is Chatbot. I will answer your queries. If you want to exit, type 'bye'!")

# Run the chatbot loop
while flag:
    user_response = input().lower()
    if user_response != 'bye':
        if user_response in ['thanks', 'thank you']:
            flag = False
            print("Chatbot: You are welcome..")
        else:
            if greeting(user_response) is not None:
                print("Chatbot: " + greeting(user_response))
            else:
                print("Chatbot: " + response(user_response))
    else:
        flag = False
        print("Chatbot: Bye! Take care..")
