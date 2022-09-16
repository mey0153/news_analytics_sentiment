#%% Raw text processing - POS Tagging and Text Normalization - Stemming and Lemmatization
# Workflow
# Preprocess raw text data using functions provided in the 2_nlp_processing_cleaning.py script
# Tokenize and remove stopwords by using functions provided in 3_nlp_tokenize_stopwords.py script
# Word count is taken after processing to quantify the impact.
# Transform text data in string type - to ensure all data is string
# Apply tokenization and stopword removal function
# Apply normalization - lemmatization or stemming
# Remove single character words
# Save output 
#  
#%% Script description
""" This script consists of a collection of functions used for removing single or double characters from string as well as example applications
    Functions:
    word_count_func()
    remove_single_char_func()    
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
#%% Change display setting
pd.set_option('display.max_colwidth', 30)    # Changed column width
# pd.set_option('display.max_colwidth', 50)    # to reset    

#%% Import dataset
df = pd.read_csv(r'D:\Project\text_mining\text_mining\data\interim\Amazon_Unlocked_Mobile_small_Part_III.csv')
print(df.head(3))
# Convert to string
df['Reviews_lemmatized'] = df['Reviews_lemmatized'].astype(str)

# Open sample row from dataset file
clean_text_lemmatized_v_a = pk.load(open(r'D:\Project\text_mining\text_mining\data\interim\clean_text_lemmatized_v_a.pkl','rb'))
print(clean_text_lemmatized_v_a)

#%%Removing single character Example 1
text_for_remove_single_char = \
"This is an example string with numbers like 5 or 10 and single characters like a, b and c."
print(text_for_remove_single_char)

# Apply single character removal function
print(remove_single_char_func(text_for_remove_single_char))

# Apply character removal with threshold = 2 , i.e., two characters removed
print(remove_single_char_func(text_for_remove_single_char, threshold=2))

#%% Application to sample string
# Sample string
print(clean_text_lemmatized_v_a)

# Apply single character removal function
clean_text_wo_single_char = remove_single_char_func(clean_text_lemmatized_v_a)
print(clean_text_wo_single_char)

#%% Application to dataframe - single character removal
# Dataframe overview
print(df.head(3))

# Apply single character removal and take word count - create two new columns

# Character removal
df['Reviews_cleaned_wo_single_char'] = df['Reviews_lemmatized'].apply(remove_single_char_func)

# Word count
df['Word_Count_cleaned_Reviews_wo_single_char'] = df['Reviews_cleaned_wo_single_char'].apply(word_count_func)

# View
print(df.head(3).T)

# Word count before single character removal
print('Average of lemmatized words counted: ' + str(df['Word_Count_lemmatized_Reviews'].mean()))

# Word count after single character removal
print('Average of cleaned words wo single char counted: ' + str(df['Word_Count_cleaned_Reviews_wo_single_char'].mean()))

#%% Investigating single character removal - in dataframe
# Create a subset
df_subset = df[['Reviews_lemmatized', 'Word_Count_lemmatized_Reviews', 
                'Reviews_cleaned_wo_single_char', 'Word_Count_cleaned_Reviews_wo_single_char']]

# Subtract word count before and after
df_subset['Diff'] = df_subset['Word_Count_lemmatized_Reviews'] - \
                    df_subset['Word_Count_cleaned_Reviews_wo_single_char']

# Create new subset where word count difference is not zero
df_subset = df_subset[(df_subset["Diff"] != 0)]

# Sort by descending word count
df_subset = df_subset.sort_values(by='Diff', ascending=False)

# View
print(df_subset.head().T)

# Exploring the subset dataset

# First row - text before single character removal
print(df_subset['Reviews_lemmatized'].iloc[0])
print()

# First row - text after single character removal
print(df_subset['Reviews_cleaned_wo_single_char'].iloc[0])
print()

# Comparison with original review text
df[(df.index==7479)]['Reviews'].iloc[0]



#%% Application to dataframe - double character removal
# Apply double character removal and take word count - create two new columns

# Character removal
df["Reviews_cleaned_wo_char_length_2"] = df.apply(lambda x: remove_single_char_func(x["Reviews_lemmatized"], 
                                                            threshold=2), axis = 1)
# Word count
df['Word_Count_cleaned_Reviews_wo_char_length_2'] = df['Reviews_cleaned_wo_char_length_2'].apply(word_count_func)

# View
print(df.head(3).T)

# Word count before single character removal
print('Average of lemmatized words counted: ' + str(df['Word_Count_lemmatized_Reviews'].mean()))
# Word count after single character removal
print('Average of cleaned words wo single char counted: ' + str(df['Word_Count_cleaned_Reviews_wo_single_char'].mean()))
# Word count after two character removal
print('Average of cleaned words wo char length 2 counted: ' + str(df['Word_Count_cleaned_Reviews_wo_char_length_2'].mean()))

#%% Investigating double character removal - dataframe
# Create a subset
df_subset = df[['Reviews_lemmatized', 'Word_Count_lemmatized_Reviews', 
                'Reviews_cleaned_wo_char_length_2', 'Word_Count_cleaned_Reviews_wo_char_length_2']]

# Subtract word count before and after
df_subset['Diff'] = df_subset['Word_Count_lemmatized_Reviews'] - \
                    df_subset['Word_Count_cleaned_Reviews_wo_char_length_2']
                    
# Create new subset where word count difference is not zero
df_subset = df_subset[(df_subset["Diff"] != 0)]

# Sort by descending word count
# df_subset = df_subset.sort_values(by='Diff', ascending=False)
print(df_subset.head().T)

# Text before character removal
print(df_subset['Reviews_lemmatized'].iloc[1])
print()

# Text after character removal
print(df_subset['Reviews_cleaned_wo_char_length_2'].iloc[1])
print()

# Original text
# print(df[(df.index==7479)]['Reviews'].iloc[0])    # For sorted rows
print(df[(df.index==3)]['Reviews'].iloc[0])    # For unsorted rows

#%%# Save Output
# Sample text
pk.dump(clean_text_wo_single_char, open(r'D:\Project\text_mining\text_mining\data\interim\clean_text_wo_single_char.pkl', 'wb'))
# Dataframe
df.to_csv(r'D:\Project\text_mining\text_mining\data\interim\Amazon_Unlocked_Mobile_small_Part_IV.csv', index = False)

