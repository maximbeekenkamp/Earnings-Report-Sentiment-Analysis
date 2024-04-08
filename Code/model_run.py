from preprocessing import DataSet
from word_freq import WordFreq
from tfidf import TfIdf

class Runner:
    def __init__(self, singleCompany=False):
        """
        Class to run the project.
        """
        self.corpus = DataSet(singleCompany=singleCompany)
    
    def run(self, singleCompany):
        """
        Run the project.
        """
        if not singleCompany:
            print("Running for all companies")
            sim_list = []
            for company in self.corpus.df["Company"].unique():
                df = self.corpus.df[self.corpus.df["Company"] == company]
                pres_train, qa_train = self.corpus.split_pres_qa(df)
                tfidf = TfIdf(pres_train, qa_train, sim_list)
                
                word_freq = WordFreq(pres_train, qa_train)
                word_freq.count_words(company)

                tfidf.similarity(company)
                
            print("Similarity scores between for Presentation and Q&A sections all companies:")
            for comp, sim in tfidf.similarity_ranked:
                print(f"{comp}: {sim}")

        else:
            company = singleCompany
            print(f"Running for {company}")
            df = self.corpus.df[self.corpus.df["Company"] == company]
            pres_train, qa_train = self.corpus.split_pres_qa(df)
            
            word_freq = WordFreq(pres_train, qa_train)
            word_freq.count_words(company)

            tfidf = TfIdf(pres_train, qa_train, [])
            tfidf.similarity(company)

            print(f"Similarity for {company} between Presentation and Q&A sections: {tfidf.similarity_ranked[0][1]}")

