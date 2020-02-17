
from collections import Counter
import numpy as np
import pandas as pd
import stop_words

class NLP():
    """
    A NLP class to perform count_vectorizer.
    """
    
    def __init__(self):
        self.vocab = None
    
    def count_vectorizer(self, text, train = True, stop_word=None, view=False):
        """
            TODO:
                * Better preprocessing using regex, remove numbers.
            Inputs:
                text: Input data as list of Text.
                stop_words: List or array of stop words. If none, default used.

            Outputs:
                Dataframe of count_vector

            Steps:
                * Lowercase applied
                * Punctuation removed
                * Removed stop words
                * Performed bag of words
                * Frequency of words
                * Dataframe of frequency of words
        
        """


        lower_case_documents = []
        documents=text
        for i in documents:
            lower_case_documents.append(i.lower())
        
        if view:
            print('Step: Applying Lower Case.... Done\n')
    #     print(lower_case_documents)
        sans_punctuation_documents = []
        
        import string

        for i in lower_case_documents:
            punctuation = string.punctuation

            k = ""
            for j in i:
                if j not in punctuation:
                    k+=j
                    
            sans_punctuation_documents.append(k)
        
        if view:
            print('Step: Removed Punctuation....\n')
    #     print(sans_punctuation_documents)
        
        if stop_word == None:
            stop_word = list(stop_words.ENGLISH_STOP_WORDS)
        
        preprocessed_documents = []
        for i in sans_punctuation_documents:
            sentence = []
            for word in i.split():
                if word not in stop_word:
                    sentence.append(word)
            preprocessed_documents.append(sentence)
        
        if train != True:
            return preprocessed_documents
        
        if view:
            print('Step: Bag of Words... Done\n')
    #     print(preprocessed_documents)

        frequency_list = []
        from collections import Counter

        for i in preprocessed_documents:
            frequency_list.append(dict(Counter(i)))
        
        if view:
            print('Step: Frequency of words... Done\n')
        
        # often called as vocabulary
        all_words = list(set([j for i in preprocessed_documents for j in i]))

        for doc in frequency_list:
            for word in all_words:
                if word not in list(doc.keys()):
                    doc[word] = 0
        df = pd.DataFrame(frequency_list)
        df = df[sorted(list(df.columns))]
        
        self.vocab = df.columns.to_list()
        
        if view:
            print('Step: Count vectorizer... Done\n')
#         print(df.head())
        return df

nlp = NLP()
documents = ['hello there', 'I will be available there', 'and again we won']
count_vector = nlp.count_vectorizer(documents)
count_vector
