from collections import Counter
from nlp import NLP

nlp = NLP()

class NaiveBayes():
    """
        A class to perform Naive Bayes on text.
        Methods:
            * fit: to train a model
            * predict: to do prediction

        Use Cases:
        spam = ['win money now', 'easy moneey now', 'win the money by replying']
        ham = ['can you borrow money', 'good boy', 'it was easy game', 'hello buddy', 'hi']

        all_txt = spam + ham
        classes = [0, 0, 0, 1, 1, 1, 1, 1]
        nb = NaiveBayes(all_txt, classes)
        nb.fit()

        test = ['easy boy easy. boy you need easy money?', "it was easy game" ]
        nb.predict([spam[1]])
        # {0: 0.06976744186046512, 1: 0.9302325581395349}
    """
    def __init__(self, text, label):
        self.text = text
        self.label = label
        self.cond_probs = {}
        self.features = []
        self.classes = []
        self.class_prob = {}
    
    def fit(self, view=False):
        """
            Input: List of texts.
            
            A method to find all the probability of P(word/class).
            It finds out the probabilty for each word to be on each class.
            Example:
            --------
                spam = ['win money now', 'easy moneey now', 'win the money by replying']
                ham = ['can you borrow money', 'good boy', 'it was easy game', 'hello buddy', 'hi']

                all_txt = spam + ham
                classes = [0, 0, 0, 1, 1, 1, 1, 1]
                nb = NaiveBayes(all_txt, classes)
                nb.fit()
                
            Steps:
            ---------
            * Find the BoW
            * Find the examples on each class
            * Find the probability of word on class p(w/c)
            * Find the probability of class given word. P(c/w)
        
        """
        
        text = self.text
        label = self.label
        
        
        bow = nlp.count_vectorizer(text)
        
        self.features = bow.columns.to_list() 
        
        if view:
            print('Your BoW is:\n', bow)
            
        classes = label
        
        self.classes = list(Counter(classes).keys())
        
        bow['out'] = classes
        bow_class = bow.groupby(by='out', axis=0)

        # count of each class examples
        counts = bow_class.count()
        
        # used for prediction
        class_prob = counts / counts.sum(axis=0)
        class_prob = dict(class_prob.mean(axis=1))
        self.class_prob = class_prob
        
        # count of each word on each class
        self.count_words_class = bow_class.sum()

        # find prob of word in each class.... no. of that word in class / total word in class
        prob_w_c = bow_class.sum() / counts
        
        # find p(word/class)
        
        prob_w_c = round(prob_w_c * counts / counts.sum(axis=0), 5)
        self.cond_probs = prob_w_c
        
    def classes_(self):
        """
        A method to see all classes counts for each word.
        """
        return self.count_words_class 
    
    def predict(self, example):
        """
            A method for prediction.
            Input: List of text. 
            Output: Prediction for each classes.
            
            Example:
            ----------
            
            >>>test = ['easy boy easy. boy you need easy money?', "it was easy game" ]
            >>>nb.predict([spam[1]])
            {0: 0.06976744186046512, 1: 0.9302325581395349}
        """
        txt = nlp.count_vectorizer(example, train= False)
        words = dict(Counter(txt[0]))
        
        vocab = self.features
        classes = self.classes
        class_prob = self.class_prob
        p = self.cond_probs
        
        # probs will store denominator value for each class. We have to add all values of it to get denominator
        # probs will store values of P(w/c) where c is classes and w is words.
        probs = {}

        # numinator
        # same as probs

        num = {k:v for k,v in class_prob.items()}
        
        
        """
        c, ~c
        p(~c/w1, w2, w3) = p(w1, w2, w3 / ~c) * p(~c) / (p(w1, w2, w3/c) * p(c) + p(w1, w2, w3/~c) * p(~c))
        
        
        p(c/ w1, w1, w3) = p(w1, w2, w3 / c) * p(c) / (p(w1, w2, w3/c) * p(c) + p(w1 , w2, w3 / ~c) * p(~c))
        p(w1, w2, w3/c) = p(w1/c) * p(w2/c) * p(w3/c) = p(w1 and w2 and w3 / c)
        """
        
        for w in words.keys():
            if w in vocab:
                for c in classes:
                    if probs.get(c) != None:
                        if p[w][c] != 0:
                            probs[c] *= p[w][c] 
                            num[c] *= p[w][c]
                    else:  
                        probs[c] = p[w][c] * class_prob[c] 
                        num[c] *= p[w][c]

        # to find probability of class given word or P(c/w), we have formula
        # = p(w/c) * p(c) / p(w)
        # p(w) = sum over all p(w/c) * p(c) is TP + TN 

        denom = sum(probs.values())
        probs = {k: v/denom for k,v in num.items()}
        return probs

        """     
            ### scarp code
                #         sum(probs.values()), probs
                #         example = nlp.count_vectorizer(example)
                #         words = example.columns.to_list()

                #         vocab = self.features
                #         classes = self.classes
                #         p = self.cond_probs

                #         probs = {}

                #         class_prob = self.class_prob

                #         for w in words:
                #             if w in vocab:
                #                 for c in classes:
                #                     if probs.get(c) != None:
                #                         # this is actually p(class/word) * p(class)
                #                         probs[c] *= p[w][c]
                #                     else:
                #                         probs[c] = p[w][c] * class_prob[c]

                #         return probs

        """
        
        
spam = ['win money now', 'easy moneey now', 'win the money by replying']
ham = ['can you borrow money', 'good boy', 'it was easy game', 'hello buddy', 'hi']

all_txt = spam + ham
classes = [0, 0, 0, 1, 1, 1, 1, 1]
nb = NaiveBayes(all_txt, classes)
nb.fit()

# nb.cond_probs
test = ['easy boy easy. boy you need easy money?', "buddy, easy money", 'win easy money by replying']
nb.predict([test[0]])