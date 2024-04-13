import numpy as np
import sys

from sklearn.feature_extraction.text import TfidfVectorizer


class Embeddings:
    def __init__(self, pres_df, qa_df, vocab):
        """
        Class to calculate the cosine similarity between the Presentation and QA
        sections of a company.

        Args:
            pres_df (df): df containing a list of paragraphs containing a list
            of words from the presentation section.
            qa_df (df): df containing a list of question answer tuples containing
            lists of words from the Q&A section.
            similarity_ranked (list): List of tuples containing the company name
            and its similarity score.
        """
        self.pres_list = list(pres_df) 
        self.qa_list = list(qa_df)
        self.tfidf_dict = {}
        self.vocab = vocab

    def embedding_matrix(self, company, mode):
        """
        Passes a single report to the similarity_per_report function to calculate
        the similarity between the presentation and QA sections.

        Args:
            company (str): Companies ticker symbol.
        """
        self.tfidf_dict[company] = {}
        assert len(self.pres_list) == len(self.qa_list), "Same number of reports required."
        for i, pres in enumerate(self.pres_list):
            if mode == "tfidf":
                self.tfidf(company, pres, self.qa_list[i], i)
            elif mode == "doc2vec":
                self.doc2vec(company, pres, self.qa_list[i], i)
            else:
                raise ValueError("Invalid mode. Please choose 'tfidf' or 'doc2vec'.")

        pass

    def tfidf(self, company, pres, qa, i):
        """
        Calculates the cosine similarity between the presentation and QA sections
        of a single report.

        Args:
            company (str): Companies ticker symbol.
            pres (list): Presentation section of a single report. 
            Format:
            [Para[Word[str]]]
            qa (list): QA section of a single report.
            Format:
            [(Ques[Word[str]], Ans[Word[str]])]
            i (int): Report number. (0-15)
        
        Raises:
            ValueError: If an error occurs in the presentation or QA sections.
        """
        self.tfidf_dict[company][i] = {}

        tot_pres = [word for sublist in pres for word in sublist]
        tot_qa = [word for sublist in qa for item in sublist for word in item]
        tot_tot = tot_pres + tot_qa
        vectorizer = TfidfVectorizer()
        vectorizer.fit_transform(tot_tot)
        try:
            self.tfidf_dict[company][i]["Presentation"] = [
                vectorizer.transform(para) for para in pres
            ]
        except:
            output_msg = "Error in Presentation"
            output_msg += f"\n Company: {company} \n Report number: {i}"
            for para in pres:
                output_msg += f"\n {para}"
            raise ValueError(f"{output_msg}")
        try:
            self.tfidf_dict[company][i]["QA"] = [
                (vectorizer.transform(ques), vectorizer.transform(ans))
                for ques, ans in qa
            ]
        except:
            output_msg = "Error in QA"
            output_msg += f"\n Company: {company} \n Report number: {i}"
            for ques, ans in qa:
                output_msg += f"\n QUES: {ques} \n ANS: {ans}"
            raise ValueError(f"{output_msg}")
    
    def doc2vec(self, company, pres, qa, i):
        pass