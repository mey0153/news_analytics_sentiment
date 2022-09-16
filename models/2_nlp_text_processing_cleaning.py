#%% Raw text preprocessing script
# Workflow
# Convert all data to string type
# Convert all text to lowercase
# Count number of words before preprocessing
# Remove html tags
# Remove URLs
# Remove accented characters
# Remove punctuations
# Remove irrelevant characters
# Remove extra whitespaces
# Count number of words after preprocessing
# Additional steps
# Expand word contractions
#%% Script description
""" This script consists of a collection of functions used for pre-processing raw text data and example application
    Functions:
    remove_html_tags_func()
    remove_url_func()
    remove_accented_chars_func()
    remove_punctuation_func()
    remove_irr_char_func()
    remove_extra_whitespaces_func()
    word_count_func()
    expand_contractions() - using two different packages
"""
#%% Import required libraries
# NLTK module
import nltk   # nltk module
nltk.download('punkt')    # download tokenizers
nltk.download('stopwords')    # download stopwords
nltk.download('wordnet')    # download wordnet
nltk.download('averaged_perceptron_tagger')    # download perceptron tagger
nltk.download('maxent_ne_chunker')    # download chunker

from nltk.tokenize import sent_tokenize, word_tokenize    # tokenizers
from nltk.corpus import stopwords    # stopwords
from nltk.corpus import wordnet
from nltk import pos_tag
from nltk import ne_chunk

# Import Stemmers
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

from nltk.probability import FreqDist
import re    # Regular expressions

# Import Pandas, Numpy, pickle and os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle as pk
import os    # For reading the titles of files
# Import warning and ignore
import warnings
warnings.filterwarnings("ignore")
# Import beautiful soup
from bs4 import BeautifulSoup as bs

import unicodedata
from wordcloud import WordCloud

#%% Functions
# Function for removing html tags

def remove_html_tags_func(text):
    '''
    Removes HTML-Tags from a string, if present
    
    Args:
        text (str): String to which the function is to be applied, string
    
    Returns:
        Clean string without HTML-Tags
    ''' 
    return bs(text, 'html.parser').get_text()

# Function for removing URLs
def remove_url_func(text):
    '''
    Removes URL addresses from a string, if present
    
    Args:
        text (str): String to which the function is to be applied, string
    
    Returns:
        Clean string without URL addresses
    ''' 
    return re.sub(r'https?://\S+|www\.\S+', '', text)

# Function for removing accented characters
def remove_accented_chars_func(text):
    '''
    Removes all accented characters from a string, if present
    
    Args:
        text (str): String to which the function is to be applied, string
    
    Returns:
        Clean string without accented characters
    '''
    return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')

# Function for removing punctuations
def remove_punctuation_func(text):
    '''
    Removes all punctuation from a string, if present
    
    Args:
        text (str): String to which the function is to be applied, string
    
    Returns:
        Clean string without punctuations
    '''
    return re.sub(r'[^a-zA-Z0-9]', ' ', text)

# Function for removing irrelevant characters
def remove_irr_char_func(text):
    '''
    Removes all irrelevant characters (numbers and punctuation) from a string, if present
    
    Args:
        text (str): String to which the function is to be applied, string
    
    Returns:
        Clean string without irrelevant characters
    '''
    return re.sub(r'[^a-zA-Z]', ' ', text)

# Function for removing extra white spaces
def remove_extra_whitespaces_func(text):
    '''
    Removes extra whitespaces from a string, if present
    
    Args:
        text (str): String to which the function is to be applied, string
    
    Returns:
        Clean string without extra whitespaces
    ''' 
    return re.sub(r'^\s*|\s\s*', ' ', text).strip()

# Function for counting words
def word_count_func(text):
    '''
    Counts words within a string
    
    Args:
        text (str): String to which the function is to be applied, string
    
    Returns:
        Number of words within a string, integer
    ''' 
    return len(text.split())

# Function for expanding contractions
from contractions import CONTRACTION_MAP 
import re 

def expand_contractions(text, map=CONTRACTION_MAP):
    pattern = re.compile('({})'.format('|'.join(map.keys())), flags=re.IGNORECASE|re.DOTALL)
    def get_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded = map.get(match) if map.get(match) else map.get(match.lower())
        expanded = first_char+expanded[1:]
        return expanded     
    new_text = pattern.sub(get_match, text)
    new_text = re.sub("'", "", new_text)
    return new_text

# Another function for expanding contractions
from pycontractions import Contractions
cont = Contractions(kv_model=model)
cont.load_models()# 

def expand_contractions(text):
    text = list(cont.expand_texts([text], precise=True))[0]
    return text
#%% Application example
# Text Cleaning - Application to the DataFrame
df['Clean_Reviews'] = df['Reviews'].str.lower()
df['Clean_Reviews'] = df['Clean_Reviews'].apply(remove_html_tags_func)
df['Clean_Reviews'] = df['Clean_Reviews'].apply(remove_url_func)
df['Clean_Reviews'] = df['Clean_Reviews'].apply(remove_accented_chars_func)
df['Clean_Reviews'] = df['Clean_Reviews'].apply(remove_punctuation_func)
df['Clean_Reviews'] = df['Clean_Reviews'].apply(remove_irr_char_func)
df['Clean_Reviews'] = df['Clean_Reviews'].apply(remove_extra_whitespaces_func)

# View dataframs
df.head()
#%% Count words
# Word count
df['Word_Count'] = df['Clean_Reviews'].apply(word_count_func)    # created a new column
df[['Clean_Reviews', 'Word_Count']].head()

#%% Assign clean_text
clean_text = messy_text_whitespace_removed    # only in this particular example

# Write the file as a pickle file
pk.dump(clean_text, open(r'D:\Project\text_mining\text_mining\data\interim\clean_text.pkl', 'wb'))
# Write data to csv file
df.to_csv(r'D:\Project\text_mining\text_mining\data\interim\Amazon_Unlocked_Mobile_small_Part_I.csv', index = False)