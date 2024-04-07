from collections import Counter

from plot import Plotter

class WordFreq:
    def __init__(self, pres_text, qa_text):
        self.pres_text = pres_text
        self.qa_text = qa_text
        self.word_freq = {}
        self.tfidf_dict = {}
    
    def count_words(self, company):
        self.word_freq[company] = {}
        self.tfidf_dict[company] = {}

        self.word_freq[company]["Presentation"] = Counter(self.pres_text)
        self.word_freq[company]["QA"] = Counter(self.qa_text)

        plot = Plotter(company, word_freq=self.word_freq)
        plot.plot_word_freq()