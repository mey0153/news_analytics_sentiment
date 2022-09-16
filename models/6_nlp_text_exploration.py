#%% Processed Text Exploration
# Workflow
# Preprocess raw text data using functions provided in the 2_nlp_processing_cleaning.py script
# Tokenize and remove stopwords by using functions provided in 3_nlp_tokenize_stopwords.py script
# Word count is taken after processing to quantify the impact.
# Transform text data in string type - to ensure all data is string
# Apply tokenization and stopword removal function
# Apply normalization - lemmatization or stemming
# Remove single character words
# Create and explore text corpus - most/least common words and their visualizations
# Save output 
#  
#%% Script description
""" This script consists of a collection of functions used for creating a text corpus and exploring characteristics of the text corpus as well as example applications
    Functions:
    token_and_unique_word_count_func()
    most_common_word_func()
    least_common_word_func()       
"""

#%% Import the Libraries
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

#%% Change display setting
pd.set_option('display.max_colwidth', 30)    # Changed column width
# pd.set_option('display.max_colwidth', 50)    # to reset

#%% Import data
df = pd.read_csv(r'D:\Project\text_mining\text_mining\data\interim\Amazon_Unlocked_Mobile_small_Part_IV.csv')
print(df.head(3))

# Convert to string
df['Reviews_cleaned_wo_single_char'] = df['Reviews_cleaned_wo_single_char'].astype(str)

# Open sample row from dataset file
clean_text_wo_single_char = pk.load(open(r'D:\Project\text_mining\text_mining\data\interim\clean_text_wo_single_char.pkl','rb'))
print(clean_text_wo_single_char)

#%% Example of descriptive statistics
# Example 1
text_for_exploration = \
"To begin to toboggan first buy a toboggan, but do not buy too big a toboggan. \
Too big a toboggan is too big a toboggan to buy to begin to toboggan."
print(text_for_exploration)   

# Count the number of unique words and tokens
print(token_and_unique_word_count_func(text_for_exploration))

# Most common words
print(most_common_word_func(text_for_exploration))
print()
# Top 10 most common words - change n_words
df_most_common_words_10 = most_common_word_func(text_for_exploration, n_words=10)
print(df_most_common_words_10)

# Least common words
print(least_common_word_func(text_for_exploration, 3))

# Text visualization via Bar Charts
plt.figure(figsize=(11,7))
plt.bar(df_most_common_words_10['Word'], 
        df_most_common_words_10['Frequency'])

plt.xlabel('Most common Words')
plt.ylabel("Frequency")
plt.title("Frequency distribution of the 10 most common words")

plt.show()

# Text visualization via wordclouds
wordcloud = WordCloud(width = 800, height = 800,
                background_color ='white',
                min_font_size = 10).generate(text_for_exploration)

plt.figure(figsize = (8, 8), facecolor = None)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
  
plt.show()

#%% Application to the Example String
# String
print(clean_text_wo_single_char)

# Most common words
df_most_common_words = most_common_word_func(clean_text_wo_single_char)
# View df
print(df_most_common_words.head(10))

# Visualize most common using Bar Chart
plt.figure(figsize=(11,7))
plt.bar(df_most_common_words['Word'], 
        df_most_common_words['Frequency'])

plt.xticks(rotation = 45)

plt.xlabel('Most common Words')
plt.ylabel("Frequency")
plt.title("Frequency distribution of the 25 most common words")

plt.show()

# Least common Words
df_least_common_words = least_common_word_func(clean_text_wo_single_char, n_words=10)
print(df_least_common_words)

# Visualize least common using Bar Chart
plt.figure(figsize=(11,7))
plt.bar(df_least_common_words['Word'], 
        df_least_common_words['Frequency'])

plt.xticks(rotation = 45)

plt.xlabel('Least common Words')
plt.ylabel("Frequency")
plt.title("Frequency distribution of the 10 least common words")

plt.show()

#%% Application to the whole df
# Create text corpus
text_corpus = df['Reviews_cleaned_wo_single_char'].str.cat(sep=' ')
# View text corpus
print(text_corpus)

# Count tokens and words in text corpus
print(token_and_unique_word_count_func(text_corpus))

# Most common words
df_most_common_words_text_corpus = most_common_word_func(text_corpus)
print(df_most_common_words_text_corpus.head(10))

# Visualizing most common words in Bar Chart
plt.figure(figsize=(11,7))
plt.bar(df_most_common_words_text_corpus['Word'], 
        df_most_common_words_text_corpus['Frequency'])

plt.xticks(rotation = 45)

plt.xlabel('Most common Words')
plt.ylabel("Frequency")
plt.title("Frequency distribution of the 25 most common words")

plt.show()

# Least common words
df_least_common_words_text_corpus = least_common_word_func(text_corpus, n_words=10)

print(df_least_common_words_text_corpus)

# Visualizing least common words
plt.figure(figsize=(11,7))
plt.bar(df_least_common_words_text_corpus['Word'], 
        df_least_common_words_text_corpus['Frequency'])

plt.xticks(rotation = 45)

plt.xlabel('Least common Words')
plt.ylabel("Frequency")
plt.title("Frequency distribution of the 10 least common words")

plt.show()

#%% Text exploration by ratings categories

# Explore ratings
print(df['Rating'].value_counts())

# Function for creating labels

def label_func(rating):
    if rating <= 2:
        return 'negative'
    if rating >= 4:
        return 'positive'
    else:
        return 'neutral'

# Apply labeling function to df
df['Label'] = df['Rating'].apply(label_func)    

# Reorder columns
cols = list(df.columns)
cols = [cols[-1]] + cols[:-1]
df = df[cols]
print(df.T)

#%% Dividing reviews  according to their labels
# Positive reviews
positive_review = df[(df["Label"] == 'positive')]['Reviews_cleaned_wo_single_char']
# Neutral reviews
neutral_review = df[(df["Label"] == 'neutral')]['Reviews_cleaned_wo_single_char']
# Negative reviews
negative_review = df[(df["Label"] == 'negative')]['Reviews_cleaned_wo_single_char']

#%% Create separate text corpus for each review type
# Text corpus for positive reviews
text_corpus_positive_review = positive_review.str.cat(sep=' ')
# Text corpus for neutral reviews
text_corpus_neutral_review = neutral_review.str.cat(sep=' ')
# Text corpus for negative reviews
text_corpus_negative_review = negative_review.str.cat(sep=' ')

#%% Use most_common_word func and create separate dataframes for each review type
# Most common words in positive reviews
df_most_common_words_text_corpus_positive_review = most_common_word_func(text_corpus_positive_review)
# Most common words in neutral reviews
df_most_common_words_text_corpus_neutral_review = most_common_word_func(text_corpus_neutral_review)
# Most common words in negative reviews
df_most_common_words_text_corpus_negative_review = most_common_word_func(text_corpus_negative_review)

#%% Visualize most common words across three review types
# Combine dataframes
splited_data = [df_most_common_words_text_corpus_positive_review,
                df_most_common_words_text_corpus_neutral_review,
                df_most_common_words_text_corpus_negative_review]

# Define color and title list
color_list = ['green', 'red', 'cyan']
title_list = ['Positive Review', 'Neutral Review', 'Negative Review']

# For loop for plotting
for item in range(3):
    plt.figure(figsize=(11,7))
    plt.bar(splited_data[item]['Word'], 
            splited_data[item]['Frequency'],
            color=color_list[item])
    plt.xticks(rotation = 45)
    plt.xlabel('Most common Words')
    plt.ylabel("Frequency")
    plt.title("Frequency distribution of the 25 most common words")
    plt.suptitle(title_list[item], fontsize=15)
    plt.show()

#%% Save dataframe
df.to_csv(r'D:\Project\text_mining\text_mining\data\processed\Amazon_Unlocked_Mobile_small_Part_V.csv', index = False)

# Save frequency tables
df_most_common_words.to_csv(r'D:\Project\text_mining\text_mining\data\processed\df_most_common_words.csv', index = False)
df_least_common_words.to_csv(r'D:\Project\text_mining\text_mining\data\processed\df_least_common_words.csv', index = False)
df_most_common_words_text_corpus.to_csv(r'D:\Project\text_mining\text_mining\data\processed\df_most_common_words_text_corpus.csv', index = False)
df_least_common_words_text_corpus.to_csv(r'D:\Project\text_mining\text_mining\data\processed\df_least_common_words_text_corpus.csv', index = False)    