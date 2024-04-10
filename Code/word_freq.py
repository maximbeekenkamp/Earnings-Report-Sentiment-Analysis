from collections import Counter

from plot import Plotter

class WordFreq:
    def __init__(self, pres_text, qa_text, corpus):
        self.pres_text = [word for sublist in pres_text for word in sublist]
        self.pres_text = [word for sublist in self.pres_text for word in sublist]
        self.qa_text = [word for sublist in qa_text for word in sublist]
        self.qa_text = [word for sublist in self.qa_text for item in sublist for word in item]
        self.word_freq = {}
        self.tfidf_dict = {}
        self.corpus = corpus

    
    def count_words(self, company):
        self.word_freq[company] = {}
        self.tfidf_dict[company] = {}


        self.word_freq[company]["Presentation"] = Counter(self.pres_text)
        self.word_freq[company]["QA"] = Counter(self.qa_text)

        plot = Plotter(company, corpus=self.corpus, word_freq=self.word_freq)
        plot.plot_word_freq()