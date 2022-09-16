#%% Raw text preprocessing script
# Workflow
# Preprocess raw text data using functions provided in the 2_nlp_processing_cleaning.py script
# Save output 
# Tokenize and remove stopwords by using functions provided in 3_nlp_tokenize_stopwords.py script
# Save output 
# Word count is taken after processing to quantify the impact.
# Save output 
# Transform text data in string type - to ensure all data is string
# Save output 
# Apply tokenization and stopword removal function
# Save output 
# Apply normalization - lemmatization or stemming
# Save output 
# Remove single character words
# Save output 
# Create and explore text corpus - most/least common words and their visualizations
# Save output 
# Remove single or multiple words
# Save output 


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
    remove_english_stopwords_func()
    norm_stemming_func()
    norm_lemm_func()
    norm_lemm_v_func()
    norm_lemm_a_func()
    get_wordnet_pos_func()
    norm_lemm_POS_tag_func()
    norm_lemm_v_a_func()
    remove_single_char_func()
    token_and_unique_word_count_func()
    most_common_word_func()
    least_common_word_func()
    # single_word_remove_func()
    # multiple_word_remove_func()
    # most_freq_word_func()
    # most_rare_word_func()       
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

# Function for expanding contractions
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

# Remove stopwords function
def remove_english_stopwords_func(text):

    '''
    Removes Stop Words (also capitalized) from a string, if present
    
    Args:
        text (str): String to which the function is to be applied, string
    
    Returns:
        Clean string without Stop Words
    ''' 
    # check in lowercase 
    t = [token for token in text if token.lower() not in stopwords.words("english")]
    text = ' '.join(t)    
    return text

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

# Remove single character from text function
def remove_single_char_func(text, threshold=1):

    '''
    Removes single characters from string, if present
    
    Step 1: Use word_tokenize() to get tokens from string
    Step 2: Removes words whose length falls below the threshold (by default = 1)
    
    Args:
        text (str): String to which the functions are to be applied, string
    
    Returns:
        String with removed words whose length was below the threshold (by default = 1)
    ''' 
    threshold = threshold
    
    words = word_tokenize(text)
    text = ' '.join([word for word in words if len(word) > threshold])
    return text

# Function to count number of words and number of unique words
def token_and_unique_word_count_func(text):
    '''
    Outputs the number of words and unique words
    
    Step 1: Use word_tokenize() to get tokens from string
    Step 2: Uses the FreqDist function to count unique words
    
    Args:
        text (str): String to which the functions are to be applied, string
    
    Prints:
        Number of existing tokens and number of unique words
    ''' 
    words = word_tokenize(text)
    fdist = FreqDist(words) 
    
    print('Number of tokens: ' + str(len(words))) 
    print('Number of unique words: ' + str(len(fdist)))

# Function for counting the most common word - default = 25
def most_common_word_func(text, n_words=25):

    '''
    Returns a DataFrame with the most commonly used words from a text with their frequencies
    
    Step 1: Use word_tokenize() to get tokens from string
    Step 2: Uses the FreqDist function to determine the word frequency
    
    Args:
        text (str): String to which the functions are to be applied, string
    
    Returns:
        A DataFrame with the most commonly occurring words (by default = 25) with their frequencies
    ''' 
    words = word_tokenize(text)
    fdist = FreqDist(words) 
    n_words = n_words
    
    df_fdist = pd.DataFrame({'Word': fdist.keys(),
                             'Frequency': fdist.values()})
    df_fdist = df_fdist.sort_values(by='Frequency', ascending=False).head(n_words)
    
    return df_fdist

# Function for counting the least common word - default = 25
def least_common_word_func(text, n_words=25):
    '''
    Returns a DataFrame with the least commonly used words from a text with their frequencies
    
    Step 1: Use word_tokenize() to get tokens from string
    Step 2: Uses the FreqDist function to determine the word frequency
    
    Args:
        text (str): String to which the functions are to be applied, string
    
    Returns:
        A DataFrame with the least commonly occurring words (by default = 25) with their frequencies
    ''' 
    words = word_tokenize(text)
    fdist = FreqDist(words) 
    
    n_words = n_words
    
    df_fdist = pd.DataFrame({'Word': fdist.keys(),
                             'Frequency': fdist.values()})
    df_fdist = df_fdist.sort_values(by='Frequency', ascending=False).tail(n_words)
    
    return df_fdist

# Single word removal function
def single_word_remove_func(text, word_2_remove):
    '''
    Removes a specific word from string, if present
    
    Step 1: Use word_tokenize() to get tokens from string
    Step 2: Removes the defined word from the created tokens
    
    Args:
        text (str): String to which the functions are to be applied, string
        word_2_remove (str): Word to be removed from the text, string
    
    Returns:
        String with removed words
    '''    
    word_to_remove = word_2_remove
    
    words = word_tokenize(text)
    text = ' '.join([word for word in words if word != word_to_remove])
    return text

# Multiple word removal function
def multiple_word_remove_func(text, words_2_remove_list):
    '''
    Removes certain words from string, if present
    
    Step 1: Use word_tokenize() to get tokens from string
    Step 2: Removes the defined words from the created tokens
    
    Args:
        text (str): String to which the functions are to be applied, string
        words_2_remove_list (list): Words to be removed from the text, list of strings
    
    Returns:
        String with removed words
    '''     
    words_to_remove_list = words_2_remove_list
    
    words = word_tokenize(text)
    text = ' '.join([word for word in words if word not in words_to_remove_list])
    return text

# Most frequent words functions
def most_freq_word_func(text, n_words=5):
    '''
    Returns the most frequently used words from a text
    
    Step 1: Use word_tokenize() to get tokens from string
    Step 2: Uses the FreqDist function to determine the word frequency
    
    Args:
        text (str): String to which the functions are to be applied, string
    
    Returns:
        List of the most frequently occurring words (by default = 5)
    ''' 
    words = word_tokenize(text)
    fdist = FreqDist(words) 
    
    df_fdist = pd.DataFrame({'Word': fdist.keys(),
                             'Frequency': fdist.values()})
    df_fdist = df_fdist.sort_values(by='Frequency', ascending=False)
    
    n_words = n_words
    most_freq_words_list = list(df_fdist['Word'][0:n_words])
    
    return most_freq_words_list

# Most rare word function
def most_rare_word_func(text, n_words=5):

    '''
    Returns the most rarely used words from a text
    
    Step 1: Use word_tokenize() to get tokens from string
    Step 2: Uses the FreqDist function to determine the word frequency
    
    Args:
        text (str): String to which the functions are to be applied, string
    
    Returns:
        List of the most rarely occurring words (by default = 5)
    ''' 
    words = word_tokenize(text)
    fdist = FreqDist(words) 
    
    df_fdist = pd.DataFrame({'Word': fdist.keys(),
                             'Frequency': fdist.values()})
    df_fdist = df_fdist.sort_values(by='Frequency', ascending=False)
    
    n_words = n_words
    most_rare_words_list = list(df_fdist['Word'][-n_words:])
    
    return most_rare_words_list
    