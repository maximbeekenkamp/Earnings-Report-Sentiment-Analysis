import numpy as np

from sklearn.metrics.pairwise import cosine_similarity

class Similarity:
    def __init__(self, pres_df, embedding_dict, similarity_ranked, vocab):
        """
        Class to run the project.
        """
        self.pres_list = list(pres_df) 
        self.embedding_dict = embedding_dict
        self.similarity_dict = {}
        self.similarity_ranked = similarity_ranked
        self.vocab = vocab


    def sim_score(self, company):
        """
        Passes a single report to the similarity_per_report function to calculate
        the similarity between the presentation and QA sections.

        Args:
            company (str): Companies ticker symbol.
        """
        self.similarity_dict[company] = {}
        for i in range(len(self.pres_list)):
            self.similarity_per_report(company, i)
        self.similarity_dict[company] = np.mean(
            list(self.similarity_dict[company].values())
        )
        self.rank_similarities(company)
        pass

    def similarity_per_report(self, company, i):
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

        sim_para_question = []
        for ques, ans in self.embedding_dict[company][i]["QA"]:
            best_para_question_match = []
            for para in self.embedding_dict[company][i]["Presentation"]:
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