import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class Similarity:
    def __init__(self, num_reports, embedding_dict, similarity_dict, similarity_ranked):
        """
        Class to calculate similarity scores from embeddings.

        Args:
            num_reports (int): Number of reports for the given company.
            (len(pres_df) = len(qa_df) = 16)
            embedding_dict (dict): dictionary containing the document embeddings for both
            the presentation and QA sections for the given company.
            similarity_ranked (list): List of tuples containing the company name
            and its similarity score.
            corpus (DataSet): DataSet class object containing the data.
        """
        self.num_reports = num_reports
        self.embedding_dict = embedding_dict
        self.similarity_dict = similarity_dict
        self.similarity_dict_mean = {}
        self.similarity_ranked = similarity_ranked

    def sim_score(self, company):
        """
        Passes a single report to the similarity_per_report function to calculate
        the similarity between the presentation and QA sections.

        Args:
            company (str): Companies ticker symbol.
        """
        self.similarity_dict[company] = {}
        self.similarity_dict_mean[company] = {}

        for i in range(self.num_reports):
            self.similarity_dict[company][i] = {}
            sim_para_question = []
            for ques, ans in self.embedding_dict[company][i]["QA"]:
                best_para_question_match = []
                for para in self.embedding_dict[company][i]["Presentation"]:
                    q_sim = np.mean(cosine_similarity(para, ques))
                    if (
                        not best_para_question_match
                        or q_sim > best_para_question_match[1]
                    ):
                        best_para_question_match = [
                            np.mean(cosine_similarity(para, ans)),
                            q_sim,
                        ]
                sim_para_question.append(best_para_question_match[0])
            self.similarity_dict[company][i] = np.mean(sim_para_question)
        self.similarity_dict_mean[company] = np.mean(
            list(self.similarity_dict[company].values())
        )
        self.rank_similarities(company)

    def rank_similarities(self, company):
        """
        Ranks the similarity scores between companies.

        Args:
            company (str): Companies ticker symbol.
        """
        if not self.similarity_ranked:
            self.similarity_ranked.append((company, self.similarity_dict_mean[company]))
        else:
            for i, (_, sim) in enumerate(self.similarity_ranked):
                if self.similarity_dict_mean[company] > sim:
                    self.similarity_ranked.insert(
                        i, (company, self.similarity_dict_mean[company])
                    )
                    break
            else:
                self.similarity_ranked.append(
                    (company, self.similarity_dict_mean[company])
                )
