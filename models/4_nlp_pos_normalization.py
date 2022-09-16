#%% Raw text processing - POS Tagging and Text Normalization - Stemming and Lemmatization
# Workflow
# Preprocess raw text data using functions provided in the 2_nlp_processing_cleaning.py script
# Tokenize and remove stopwords by using functions provided in 3_nlp_tokenize_stopwords.py script
# Word count is taken after processing to quantify the impact.
# Transform text data in string type - to ensure all data is string
# Apply tokenization and stopword removal function
# Apply lemmatization or stemming
# Save output 
# 
#%% Script description
""" This script consists of a collection of functions used for POS Tagging and Text normalization - stemming and lemmatization - as well as example applications
    Functions:
    word_count_func()
    norm_stemming_func()
    norm_lemm_func()
    norm_lemm_v_func()
    norm_lemm_a_func()
    get_wordnet_pos_func()
    norm_lemm_POS_tag_func()
    norm_lemm_v_a_func()
"""
#%% Import required libraries
import pandas as pd
import numpy as np

import pickle as pk

import warnings
warnings.filterwarnings("ignore")


from bs4 import BeautifulSoup
import unicodedata
import re

from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize

from nltk.corpus import stopwords


from nltk.corpus import wordnet
from nltk import pos_tag
from nltk import ne_chunk

from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

from nltk.probability import FreqDist
import matplotlib.pyplot as plt
from wordcloud import WordCloud

#%% Functions
# Word count function
def word_count_func(text):
    '''
    Counts words within a string
    
    Args:
        text (str): String to which the function is to be applied, string
    
    Returns:
        Number of words within a string, integer
    ''' 
    return len(text.split())

# Function for applying stemmer - using PorterStemmer()

def norm_stemming_func(text):
    '''
    Stemming tokens from string
    
    Step 1: Use word_tokenize() to get tokens from string
    Step 2: Use PorterStemmer() to stem the created tokens
    
    Args:
        text (str): String to which the functions are to be applied, string
    
    Returns:
        String with stemmed words
    '''  
    words = word_tokenize(text)
    text = ' '.join([PorterStemmer().stem(word) for word in words])
    return text

# Function for applying lemmatizer - using WordNetLemmatizer()

def norm_lemm_func(text):
    '''
    Lemmatize tokens from string
    
    Step 1: Use word_tokenize() to get tokens from string
    Step 2: Use WordNetLemmatizer() to lemmatize the created tokens
    
    Args:
        text (str): String to which the functions are to be applied, string
    
    Returns:
        String with lemmatized words
    '''  
    words = word_tokenize(text)
    text = ' '.join([WordNetLemmatizer().lemmatize(word) for word in words])
    return text

# Function for applying lemmatization on verbs - using WordNetLemmatizer()

def norm_lemm_v_func(text):
    '''
    Lemmatize tokens from string 
    
    Step 1: Use word_tokenize() to get tokens from string
    Step 2: Use WordNetLemmatizer() to lemmatize the created tokens
            POS tag is set to 'v' for verb
    
    Args:
        text (str): String to which the functions are to be applied, string
    
    Returns:
        String with lemmatized words
    '''  
    words = word_tokenize(text)
    text = ' '.join([WordNetLemmatizer().lemmatize(word, pos='v') for word in words])
    return text

# Function for applying lemmatization on adjectives - using WordNetLemmatizer()

def norm_lemm_a_func(text):
    '''
    Lemmatize tokens from string
    
    Step 1: Use word_tokenize() to get tokens from string
    Step 2: Use WordNetLemmatizer() to lemmatize the created tokens
            POS tag is set to 'a' for adjective
    
    Args:
        text (str): String to which the functions are to be applied, string
    
    Returns:
        String with lemmatized words
    ''' 
    words = word_tokenize(text)
    text = ' '.join([WordNetLemmatizer().lemmatize(word, pos='a') for word in words])
    return text

# Function for mapping POS tag of word to wordnet

def get_wordnet_pos_func(word):
    '''
    Maps the respective POS tag of a word to the format accepted by the lemmatizer of wordnet
    
    Args:
        word (str): Word to which the function is to be applied, string
    
    Returns:
        POS tag, readable for the lemmatizer of wordnet
    '''     
    tag = pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)

# Function for applying lemmatization to strings - tokenize + lemmatize

def norm_lemm_POS_tag_func(text):
    '''
    Lemmatize tokens from string
    
    Step 1: Use word_tokenize() to get tokens from string
    Step 2: Use WordNetLemmatizer() to lemmatize the created tokens
            POS tag is determined with the help of function get_wordnet_pos()
    
    Args:
        text (str): String to which the functions are to be applied, string
    
    Returns:
        String with lemmatized words
    ''' 
    words = word_tokenize(text)
    text = ' '.join([WordNetLemmatizer().lemmatize(word, get_wordnet_pos_func(word)) for word in words])
    return text

# Function for applying lemmatization on verbs and adjectives in a string - tokenize, lemmatize verbs, tokenize, lemmatize adjectives

def norm_lemm_v_a_func(text):
    '''
    Lemmatize tokens from string
    
    Step 1: Use word_tokenize() to get tokens from string
    Step 2: Use WordNetLemmatizer() with POS tag 'v' to lemmatize the created tokens
    Step 3: Use word_tokenize() to get tokens from generated string        
    Step 4: Use WordNetLemmatizer() with POS tag 'a' to lemmatize the created tokens
    
    Args:
        text (str): String to which the functions are to be applied, string
    
    Returns:
        String with lemmatized words
    '''
    words1 = word_tokenize(text)
    text1 = ' '.join([WordNetLemmatizer().lemmatize(word, pos='v') for word in words1])
    words2 = word_tokenize(text1)
    text2 = ' '.join([WordNetLemmatizer().lemmatize(word, pos='a') for word in words2])
    return text2


#%% Change display setting
pd.set_option('display.max_colwidth', 30)    # Changed column width
# pd.set_option('display.max_colwidth', 50)    # to reset
#%% Read dataset

# Import example text
clean_text_wo_stop_words = pk.load(open(r'D:\Project\text_mining\text_mining\data\interim\clean_text_wo_stop_words.pkl','rb'))
print(clean_text_wo_stop_words)

file = r'D:\Project\text_mining\text_mining\data\interim\Amazon_Unlocked_Mobile_small_Part_II.csv'

# open file and read all lines
with open(file, 'r'):
    df = pd.read_csv(file)
# Check import
print(df.head())

# Ensure text is str type
df['Reviews_wo_Stop_Words'] = df['Reviews_wo_Stop_Words'].astype(str)

#%% POS Tagging example
# Define a sentence
pos_ner_text = "Bill Gates founded Microsoft Corp. together with Paul Allen in 1975."
print(pos_ner_text)
# Get POS Tags
POS_tag = pos_tag(word_tokenize(pos_ner_text))
print(POS_tag)

#%% Named Entity Recognition (NER) example
# Create NER chunks
NER_tree = ne_chunk(pos_tag(word_tokenize(pos_ner_text)))
print(NER_tree)

#%% Example of PorterStemmer() and WordNetLemmatizer()
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

print("Original Word: 'studies' ")
print()
print('With Stemming: ' + str(stemmer.stem("studies")))
print('with Lemmatization: ' + str(lemmatizer.lemmatize("studies")))

#%% Example text for normalisation
# Define text for normalization
text_for_normalization = "\
I saw an amazing thing and ran. \
It took too long. \
We are eating and swimming. \
I want better dog.\
"
print(text_for_normalization)

#%% Apply norm stemming function to text
print(norm_stemming_func(text_for_normalization))

#%% Apply lemmatizer on text
print(norm_lemm_func(text_for_normalization))
print(text_for_normalization)
# Compare with stemmer
print(norm_stemming_func(text_for_normalization))

#%% Wordnet Lemmatizer on specific POS tag -verb
print(norm_lemm_v_func(text_for_normalization))

#%% Wordnet Lemmatizer on specific POS tag -adjective
print(norm_lemm_a_func(text_for_normalization))

#%% Applying POS to wordnet mapper function to sample text

print('POS tag for the word "dog": ' + str(get_wordnet_pos_func("dog")))
print('POS tag for the word "going": ' + str(get_wordnet_pos_func("going")))
print('POS tag for the word "good": ' + str(get_wordnet_pos_func("good")))

#%% Applying POS taggin and lemmatization function to sample text
print(norm_lemm_POS_tag_func(text_for_normalization))

#%% Apply two lemmatization algorithms with different specific POS tags one after the other:
text_for_norm_v_lemmatized = norm_lemm_v_func(text_for_normalization)
text_for_norm_n_lemmatized = norm_lemm_a_func(text_for_norm_v_lemmatized)
print(text_for_norm_n_lemmatized)

#%% Application of POS and normalization to sample string from dataset
# Sample text
print(clean_text_wo_stop_words)

#%% Apply lemmatization of adjectives
clean_text_lemmatized_v_a = norm_lemm_v_a_func(clean_text_wo_stop_words)
print(clean_text_lemmatized_v_a)

#%% Apply POS tagging and lemmatization function
clean_text_lemmatized_pos_tag = norm_lemm_POS_tag_func(clean_text_wo_stop_words)
print(clean_text_lemmatized_pos_tag)

#%%========================Application to dataframe===================
# Dataframe overview
print(df.head(3).T)

#%% Lemmatize text and create new column - Reviews_lemmatized and then count words
# Lemmatize text
df['Reviews_lemmatized'] = df['Reviews_wo_Stop_Words'].apply(norm_lemm_v_a_func)
# Count words
df['Word_Count_lemmatized_Reviews'] = df['Reviews_lemmatized'].apply(word_count_func)
# Overview
print(df.head(3).T)

#%% Save Output

# Save example text
pk.dump(clean_text_lemmatized_v_a, open(r'D:\Project\text_mining\text_mining\data\interim\clean_text_lemmatized_v_a.pkl', 'wb'))
# Save dataframe
df.to_csv(r'D:\Project\text_mining\text_mining\data\interim\Amazon_Unlocked_Mobile_small_Part_III.csv', index = False)
