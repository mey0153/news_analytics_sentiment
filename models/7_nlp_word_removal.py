#%% Processed Text Word Removal
# Workflow
# Preprocess raw text data using functions provided in the 2_nlp_processing_cleaning.py script
# Tokenize and remove stopwords by using functions provided in 3_nlp_tokenize_stopwords.py script
# Word count is taken after processing to quantify the impact.
# Transform text data in string type - to ensure all data is string
# Apply tokenization and stopword removal function
# Apply normalization - lemmatization or stemming
# Remove single character words
# Create and explore text corpus - most/least common words and their visualizations
# Remove single or multiple words
# Save output 
#  
#%% Script description
""" This script consists of a collection of functions used for removing words from text corpus as well as example applications
    Functions:
    # word_count_func()
    # single_word_remove_func()
    # multiple_word_remove_func()
    # most_freq_word_func()
    # most_rare_word_func()       
"""

#%% Import the Libraries
#%% Import the Libraries and the Data
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

#%% Data import
# Set viewing options
pd.set_option('display.max_colwidth', 30)
# pd.set_option('display.max_colwidth', 50)    # to reset

# Import data
df = pd.read_csv(r'D:\Project\text_mining\text_mining\data\processed\Amazon_Unlocked_Mobile_small_Part_V.csv')
print(df.head(3).T)

# Set as string type
df['Reviews_cleaned_wo_single_char'] = df['Reviews_cleaned_wo_single_char'].astype(str)

# Open example text
clean_text_wo_single_char = pk.load(open(r'D:\Project\text_mining\text_mining\data\interim\clean_text_wo_single_char.pkl','rb'))
print(clean_text_wo_single_char)

## Import frequency tables 
df_most_common_words = pd.read_csv(r'D:\Project\text_mining\text_mining\data\processed\df_most_common_words.csv')
df_least_common_words = pd.read_csv(r'D:\Project\text_mining\text_mining\data\processed\df_least_common_words.csv')
df_most_common_words_text_corpus = pd.read_csv(r'D:\Project\text_mining\text_mining\data\processed\df_most_common_words_text_corpus.csv')
df_least_common_words_text_corpus = pd.read_csv(r'D:\Project\text_mining\text_mining\data\processed\df_least_common_words_text_corpus.csv')

#%% Example 1 text
text_for_word_removal = \
"Give papa a cup of proper coffe in a copper coffe cup."
print(text_for_word_removal)

# Removing a single word - remove coffe
print(single_word_remove_func(text_for_word_removal, "coffe"))

# Example Method 1
# Directly enter the word
print(multiple_word_remove_func(text_for_word_removal, ["coffe", "cup"]))

# Example method 2
# Create a list of words to be removed
list_with_words = ["coffe", "cup"]
print(multiple_word_remove_func(text_for_word_removal, list_with_words))

# Define words to be removed as parameters
params= [text_for_word_removal,["coffe", "cup"]]

print(multiple_word_remove_func(*params))

#%% Removing most frequent words
# Example 2 text
text_for_freq_word_removal = \
"Peter Pepper picked a pack of pickled peppers. How many pickled peppers did Peter Pepper pick?"
print(text_for_freq_word_removal)

# Get most frequent words
most_freq_words_list = most_freq_word_func(text_for_freq_word_removal, n_words=2)
print(most_freq_words_list)

# Remove multiple words
print(multiple_word_remove_func(text_for_freq_word_removal, most_freq_words_list))

#%% Least frequent word removal Example 3 text for rare word removal
text_for_rare_word_removal = \
"Sue sells seashells by the seashore. The seashells Sue sells are seashells Sue is sure."
print(text_for_rare_word_removal)

# Rare words list - top 3
most_rare_words_list = most_rare_word_func(text_for_rare_word_removal, n_words=3)
print(most_rare_words_list)

# Multiple rare word removal
multiple_word_remove_func(text_for_rare_word_removal, most_rare_words_list)

#%% Application to sample string
print(clean_text_wo_single_char)

# Examine frequency distribution of words
plt.figure(figsize=(11,7))
plt.bar(df_most_common_words['Word'], 
        df_most_common_words['Frequency'])
plt.xticks(rotation = 45)
plt.xlabel('Most common Words')
plt.ylabel("Frequency")
plt.title("Frequency distribution of the 25 most common words")
plt.show()

# View most common words df
print(df_most_common_words.head())

# Visualize the least common words
plt.figure(figsize=(11,7))
plt.bar(df_least_common_words['Word'], 
        df_least_common_words['Frequency'])
plt.xticks(rotation = 45)
plt.xlabel('Least common Words')
plt.ylabel("Frequency")
plt.title("Frequency distribution of the 10 least common words")
plt.show()

# View the least common words
print(df_least_common_words.tail())

#%% Single word removal - remove special
clean_text_wo_specific_word = single_word_remove_func(clean_text_wo_single_char, "special")
print(clean_text_wo_specific_word)
# Check string length before and after word removal
# Number of words before removal
print('Number of words (before single word removal): ' + str(word_count_func(clean_text_wo_single_char)))
# Number of words after removal
print('Number of words (after single word removal): ' + str(word_count_func(clean_text_wo_specific_word)))

#%% Multiple word removal
clean_text_wo_specific_words = multiple_word_remove_func(clean_text_wo_specific_word, 
                                                       ["expose", "currently", "character"])
print(clean_text_wo_specific_words)

# Check string length before and after word removal
# Number of words before removal
print('Number of words (bevore multiple word removal): ' + str(word_count_func(clean_text_wo_specific_word)))

# Number of words after removal
print('Number of words (after multiple word removal): ' + str(word_count_func(clean_text_wo_specific_words)))

#%% Removing most frequent words
# Example with sample string
print(clean_text_wo_single_char)

# Most frequent words
most_freq_words_list_Example_String = most_freq_word_func(clean_text_wo_single_char, n_words=2)
print(most_freq_words_list_Example_String)

# Removing most frequent words
clean_text_wo_freq_words = multiple_word_remove_func(clean_text_wo_single_char, 
                                                     most_freq_words_list_Example_String)

print(clean_text_wo_freq_words)

#%% Removing least frequent word Application on example string
# Example string
print(clean_text_wo_single_char)

# Visualize string
plt.figure(figsize=(11,7))
plt.bar(df_least_common_words['Word'], 
        df_least_common_words['Frequency'])

plt.xticks(rotation = 45)

plt.xlabel('Least common Words')
plt.ylabel("Frequency")
plt.title("Frequency distribution of the 10 least common words")

plt.show()



#%% Application to the dataframe

# Visualizing most common words
plt.figure(figsize=(11,7))
plt.bar(df_most_common_words_text_corpus['Word'], 
        df_most_common_words_text_corpus['Frequency'])
plt.xticks(rotation = 45)
plt.xlabel('Most common Words')
plt.ylabel("Frequency")
plt.title("Frequency distribution of the 25 most common words")
plt.show()

# Most common top 5 words
print(df_most_common_words_text_corpus.head())

# Visualizing least common words
plt.figure(figsize=(11,7))
plt.bar(df_least_common_words_text_corpus['Word'], 
        df_least_common_words_text_corpus['Frequency'])
plt.xticks(rotation = 45)
plt.xlabel('Least common Words')
plt.ylabel("Frequency")
plt.title("Frequency distribution of the 10 least common words")
plt.show()

# View least common words
print(df_least_common_words_text_corpus.tail())

#%% Application to dataframe - single word removal
# Single word removal from dataframe
df["Reviews_cleaned_wo_specific_word"] = df.apply(lambda x: single_word_remove_func(x["Reviews_cleaned_wo_single_char"], 
                                                            "phone"), axis = 1)

# Create text corpus to verify word has been removed
# Original corpus
text_corpus_original = df['Reviews_cleaned_wo_single_char'].str.cat(sep=' ')
# Word removed corpus
text_corpus_wo_specific_word = df['Reviews_cleaned_wo_specific_word'].str.cat(sep=' ')
# Word count before and after
print('Number of words (before single word removal): ' + str(word_count_func(text_corpus_original)))
print('Number of words (after single word removal): ' + str(word_count_func(text_corpus_wo_specific_word)))
print()
print('Diff: ' + str(word_count_func(text_corpus_original) - word_count_func(text_corpus_wo_specific_word)))

#%% Application to dataframe - multiple word removal
df["Reviews_cleaned_wo_specific_words"] = df.apply(lambda x: multiple_word_remove_func(x["Reviews_cleaned_wo_specific_word"], 
                                                         ["stabalize", "dazzle", "vague"]), axis = 1)
text_corpus_wo_specific_words = df['Reviews_cleaned_wo_specific_words'].str.cat(sep=' ')

print('Number of words (bevore multiple word removal): ' + str(word_count_func(text_corpus_wo_specific_word)))
print('Number of words (after multiple word removal): ' + str(word_count_func(text_corpus_wo_specific_words)))
print()
print('Diff: ' + str(word_count_func(text_corpus_wo_specific_word) - word_count_func(text_corpus_wo_specific_words)))

#%% Removing most frequent words - Application to dataframe 

# Visualize most frequent words
plt.figure(figsize=(11,7))
plt.bar(df_most_common_words_text_corpus['Word'], df_most_common_words_text_corpus['Frequency'])
plt.xticks(rotation = 45)
plt.xlabel('Most common Words')
plt.ylabel("Frequency")
plt.title("Frequency distribution of the 25 most common words")
plt.show()

# List of two most frequent words in dataset
most_freq_words_list_DataFrame = most_freq_word_func(text_corpus_original, n_words=2)
print(most_freq_words_list_DataFrame)

# Create new row with two most frequent words removed

df["Reviews_cleaned_wo_freq_words"] = df.apply(lambda x: multiple_word_remove_func(x["Reviews_cleaned_wo_single_char"], 
                                                         most_freq_words_list_DataFrame), axis = 1)

# Create new text corpus without frequent words                                                         
text_corpus_wo_freq_words = df['Reviews_cleaned_wo_freq_words'].str.cat(sep=' ')

# Count number of words before and after frequent word removal
print('Number of words (before freq word removal): ' + str(word_count_func(text_corpus_original)))
print('Number of words (after freq word removal): ' + str(word_count_func(text_corpus_wo_freq_words)))
print()
print('Diff: ' + str(word_count_func(text_corpus_original) - word_count_func(text_corpus_wo_freq_words)))

#%% Application to dataframe - least common word removal
# Visualize text
plt.figure(figsize=(11,7))
plt.bar(df_least_common_words_text_corpus['Word'], 
        df_least_common_words_text_corpus['Frequency'])

plt.xticks(rotation = 45)

plt.xlabel('Least common Words')
plt.ylabel("Frequency")
plt.title("Frequency distribution of the 10 least common words")

plt.show()


# List of least common words
print(df_least_common_words_text_corpus)

# Most rare words in dataframe
most_rare_words_list_DataFrame = most_rare_word_func(text_corpus_original, n_words=4)
print(most_rare_words_list_DataFrame)

# Remove rare words
df["Reviews_cleaned_wo_rare_words"] = df.apply(lambda x: multiple_word_remove_func(x["Reviews_cleaned_wo_single_char"], 
                                                         most_rare_words_list_DataFrame), axis = 1)

# Create before and after text corpus
text_corpus_wo_rare_words = df['Reviews_cleaned_wo_rare_words'].str.cat(sep=' ')

# Count words before and after
print('Number of words (before rare word removal): ' + str(word_count_func(text_corpus_original)))
print('Number of words (after rare word removal): ' + str(word_count_func(text_corpus_wo_rare_words)))
print()
print('Diff: ' + str(word_count_func(text_corpus_original) - word_count_func(text_corpus_wo_rare_words)))                                                         