from preprocessing import DataSet
from wordfreq import WordFreq
from embeddings import Embeddings
from similarity import Similarity
from plot import Plotter


class Runner:
    def __init__(self, singleCompany=False):
        """
        Class to run the project.
        """
        self.corpus = DataSet(singleCompany=singleCompany)
        self.vocab = self.corpus.vocab

    def run(self, singleCompany):
        """
        Run the project.
        """
        if not singleCompany:
            print("Running for all companies")
            sim_list = []
            sim_dict = {}
            for company in self.corpus.df["Company"].unique():
                train = self.corpus.train[self.corpus.train["Company"] == company]
                pres_train, qa_train = self.corpus.split_pres_qa(train)
                # pres_train format: [Report#(0-15)[Para[Word[str]]]]
                # qa_train format: [Report#(0-15)[(Ques[Word[str]], Ans[Word[str]])]]
                wordfreq = WordFreq(pres_train, qa_train)
                wordfreq.count_words(company)

                tfidf_emmbedings = Embeddings(pres_train, qa_train)
                tfidf_emmbedings.embedding_matrix(company, "tfidf")

                similarity = Similarity(
                    len(pres_train), tfidf_emmbedings.tfidf_dict, sim_dict, sim_list
                )
                sim_dict[company] = similarity.similarity_dict
                sim_list = similarity.similarity_ranked
                similarity.sim_score(company)
            plot = Plotter("All")
            plot.plot_report_similarity_bar(sim_dict)

            print(
                "Similarity scores between Presentation and Q&A "
                "sections for all companies:"
            )
            for comp, sim in similarity.similarity_ranked:
                print(f"{comp}: {sim}")

        else:
            company = singleCompany
            print(f"Running for {company}")
            train = self.corpus.train[self.corpus.train["Company"] == company]
            pres_train, qa_train = self.corpus.split_pres_qa(train)

            wordfreq = WordFreq(pres_train, qa_train)
            wordfreq.count_words(company)

            tfidf_emmbedings = Embeddings(pres_train, qa_train)
            tfidf_emmbedings.embedding_matrix(company, "tfidf")

            similarity = Similarity(
                len(pres_train), tfidf_emmbedings.tfidf_dict, {}, []
            )
            similarity.sim_score(company)

            plot = Plotter(company)
            plot.plot_report_similarity_line(
                similarity.similarity_dict, singleCompany=True
            )

            print(
                f"Similarity for {company} between Presentation and Q&A "
                f"sections: {similarity.similarity_ranked[0][1]}"
            )
