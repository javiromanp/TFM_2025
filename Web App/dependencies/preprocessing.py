# dependencies/preprocessing.py

import re

def replace_users(tweet):
    # Reemplaza las menciones de usuarios en un tweet con '@user'
    tweet = re.sub(r'(RT\s@[A-Za-z]+[A-Za-z0-9-_]+)', '@user', tweet) 
    tweet = re.sub(r'(@[A-Za-z0-9-_]+[A-Za-z0-9-_]+)', '@user', tweet)
    tweet = re.sub(r'\[user\]', '@user', tweet)
    return tweet

def preprocess_text(text):
    # Convierte el texto a minúsculas y reemplaza las menciones de usuarios
    text = text.lower()
    text = replace_users(text)
    return text
