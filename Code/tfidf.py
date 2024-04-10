import numpy as np
import sys

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class TfIdf:
    def __init__(self, pres_df, qa_df, similarity_ranked, vocab):
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
        self.similarity_dict = {}
        self.tfidf_dict = {}
        self.similarity_ranked = similarity_ranked
        self.vocab = vocab

    def similarity(self, company):
        """
        Passes a single report to the similarity_per_report function to calculate
        the similarity between the presentation and QA sections.

        Args:
            company (str): Companies ticker symbol.
        """
        self.similarity_dict[company] = {}
        self.tfidf_dict[company] = {}
        assert len(self.pres_list) == len(self.qa_list), "Same number of reports required."
        for i, pres in enumerate(self.pres_list):
            self.similarity_per_report(company, pres, self.qa_list[i], i)
        self.similarity_dict[company] = np.mean(
            list(self.similarity_dict[company].values())
        )
        self.rank_similarities(company)
        pass

    def similarity_per_report(self, company, pres, qa, i):
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
        self.similarity_dict[company][i] = {}
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
            raise ValueError, f"{output_msg}"
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
            raise ValueError, f"{output_msg}"

        sim_para_question = []
        for ques, ans in self.tfidf_dict[company][i]["QA"]:
            best_para_question_match = []
            for para in self.tfidf_dict[company][i]["Presentation"]:
                q_sim = np.mean(cosine_similarity(para, ques))
                if not best_para_question_match or q_sim > best_para_question_match[1]:
                    best_para_question_match = [
                        np.mean(cosine_similarity(para, ans)),
                        q_sim,
                    ]
            sim_para_question.append(best_para_question_match[0])
        self.similarity_dict[company][i] = np.mean(sim_para_question)

        pass

    def rank_similarities(self, company):
        """
        Ranks the similarity scores between companies.

        Args:
            company (str): Companies ticker symbol.
        """
        if not self.similarity_ranked:
            self.similarity_ranked.append((company, self.similarity_dict[company]))
        else:
            for i, (_, sim) in enumerate(self.similarity_ranked):
                if self.similarity_dict[company] > sim:
                    self.similarity_ranked.insert(
                        i, (company, self.similarity_dict[company])
                    )
                    break
            else:
                self.similarity_ranked.append((company, self.similarity_dict[company]))
        pass
