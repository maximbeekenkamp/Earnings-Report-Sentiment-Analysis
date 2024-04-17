from collections import Counter

from plot import Plotter


class WordFreq:
    def __init__(self, pres_text, qa_text):
        """
        WordFreq class to count the words in the presentation and QA sections.

        Args:
            pres_text (df): DataFrame containing the presentation text.
            Format: [Report#(0-15)[Para[Word[str]]]]
            qa_text (df): DataFrame containing the QA text.
            Format: [Report#(0-15)[(Ques[Word[str]], Ans[Word[str]])]]
        """
        self.pres_text = [word for sublist in pres_text for word in sublist]
        self.pres_text = [word for sublist in self.pres_text for word in sublist]
        self.qa_text = [word for sublist in qa_text for word in sublist]
        self.qa_text = [
            word 
            for sublist in self.qa_text 
            for item in sublist 
            for word in item
        ]
        self.word_freq = {}
        self.plotObj = None

    def count_words(self, company):
        """
        Counts the words in the presentation and QA sections.
        This also helps with creating the tokens.

        Args:
            company (str): A company's ticker symbol.
        """
        self.word_freq[company] = {}

        self.word_freq[company]["Presentation"] = Counter(self.pres_text)
        self.word_freq[company]["QA"] = Counter(self.qa_text)

        self.plotObj = Plotter(company, word_freq=self.word_freq)
        self.plotObj.plot_word_freq()
