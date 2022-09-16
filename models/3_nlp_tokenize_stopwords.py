#%% Raw text processing - tokenization and stopword removal script
# Workflow
# Preprocess raw text data using functions provided in the 2_nlp_processing_cleaning.py script
# This includes: all text is converted to lowercase, word count taken, removal of unwanted 
# text suchas html tags, URLs, accented characters, punctuation, irrelevant characters, and whitespaces. 
# Word count is taken after processing to quantify the impact.
# Transform text data in string type - to ensure all data is string
# Apply tokenization
# Apply stopword removal function
# Save output 
# 
#%% Script description
""" This script consists of a collection of functions used for tokenizing text and removing stopwords as well as example applications
    Functions:
    remove_english_stopwords_func()
    
"""
#%% Import required libraries
# Import modules
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
# Function for removing stopwords in English

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
#%% Application example
# Import file
file = r'D:\Project\text_mining\text_mining\data\interim\Amazon_Unlocked_Mobile_small_Part_I.csv'
# open file and read all lines
with open(file,'r'):
    df = pd.read_csv(file)
# Check import
print(df.head())
#%% Transform datatypes
df['Clean_Reviews'] = df['Clean_Reviews'].astype(str)
# Load pickle file
clean_text = pk.load(open("clean_text.pkl",'rb'))
clean_text    
#%% Tokenization
# Text for tokenization
text_for_tokenization = \
"Hi my name is Michael. \
I am an enthusiastic Data Scientist. \
Currently I am working on a post about NLP, more specifically about the Pre-Processing Steps."
# Print
print(text_for_tokenization)
# print(clean_text)
#%% Apply word tokenize
words = word_tokenize(text_for_tokenization)
print(words)
print(type(words))
print('Number of words found: ' + str(len(words)))

#%% Apply sentence tokenize
sentences = sent_tokenize(text_for_tokenization)
print(sentences)
print(type(sentences))
print('Number of sentences found: ' + str(len(sentences)))

# Printing each sentence
for each_sentence in sentences:
    n_words=word_tokenize(each_sentence)
    print(each_sentence)   

# Print number of words in each sentence
for each_sentence in sentences:
    n_words=word_tokenize(each_sentence)
    print(len(n_words))
#%% Apply word tokenize to clean_text
# # Apply word tokenize
tokens_clean_text = word_tokenize(clean_text)
print(tokens_clean_text)
print('Number of tokens found: ' + str(len(tokens_clean_text))) 


#%%============ Stopword removal=============
# View default stopwords
print(stopwords.words("english"))

# Add additional words to default stopword list


#%% Small example
# Define text for stopword removal
text_for_stop_words = "Hi my name is Michael. I am an enthusiastic Data Scientist."
print(text_for_stop_words)

# Tokenize
tokens_text_for_stop_words = word_tokenize(text_for_stop_words)
print(tokens_text_for_stop_words)

# Apply stopword removal function
remove_english_stopwords_func(tokens_text_for_stop_words)

#%% Example 2
# Check tokenized text
print(tokens_clean_text)

# Count number of tokens
print('Number of tokens found: ' + str(len(tokens_clean_text)))

# Apply stopword removal and display the stopwords
stop_words_within_tokens_clean_text = [w for w in tokens_clean_text if w in stopwords.words("english")]
print()
print('These Stop Words were found in our example string:')
print()
print(stop_words_within_tokens_clean_text)
print()
print('Number of Stop Words found: ' + str(len(stop_words_within_tokens_clean_text)))

# Apply stopword removal and display the cleaned text after stopwords are removed
clean_text_wo_stop_words = [w for w in tokens_clean_text if w not in stopwords.words("english")]

print()
print('These words would remain after Stop Words removal:')
print()
print(clean_text_wo_stop_words)
print()
print('Number of remaining words: ' + str(len(clean_text_wo_stop_words)))

# Apply stopword removal to example text
clean_text_wo_stop_words = remove_english_stopwords_func(tokens_clean_text)
print(clean_text_wo_stop_words)
# Count the number of words
print('Number of words: ' + str(word_count_func(clean_text_wo_stop_words)))


#%% ================Application on dataframe: Tokenization==================

# View dataframe
print(df.head())
# Set viewing options - change column width from 50 to 30
pd.set_option('display.max_colwidth', 30)
# Reset viewing options to default - change column width from 30 to 50
# pd.set_option('display.max_colwidth', 50)

# Apply word tokenization to preprocessed text column and save tokens to new column - Reviews_Tokenized
df['Reviews_Tokenized'] = df['Clean_Reviews'].apply(word_tokenize)

# Count the number of tokens in each row and create a word count column Token_Count
df['Token_Count'] = df['Reviews_Tokenized'].str.len()

# Review changes in word count
print('Average of words counted: ' + str(df['Word_Count'].mean()))
print('Average of tokens counted: ' + str(df['Token_Count'].mean()))

#%%============ Optional - Investigate change in word count=============

# # Create a subset consisting of cleaned reviews, tokenized reviews and their respective counts
# df_subset = df[['Clean_Reviews', 'Word_Count', 'Reviews_Tokenized', 'Token_Count']]

# # Find the difference between token count and word count for each row
# df_subset['Diff'] = df_subset['Token_Count'] - df_subset['Word_Count']

# # From the above subset create subset of rows with non-zero difference
# df_subset = df_subset[(df_subset["Diff"] != 0)]
# # Sort rows in descending order
# df_subset.sort_values(by='Diff', ascending=False)
#%%============= Application to dataframe continued: Stopword removal=============
# Dataframe
print(df.head())

# Create column for stopword removed text
df['Reviews_wo_Stop_Words'] = df['Reviews_Tokenized'].apply(remove_english_stopwords_func)

# Creating a count column for counting text after stopword removal
df['Word_Count_wo_Stop_Words'] = df['Reviews_wo_Stop_Words'].apply(word_count_func)

df.head().T

#%% Save output

# Save example text
pk.dump(clean_text_wo_stop_words, open(r'D:\Project\text_mining\text_mining\data\interim\clean_text_wo_stop_words.pkl', 'wb'))
# Save dataframe
df.to_csv(r'D:\Project\text_mining\text_mining\data\interim\Amazon_Unlocked_Mobile_small_Part_II.csv', index = False)
