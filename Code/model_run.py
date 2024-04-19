from preprocessing import DataSet
from wordfreq import WordFreq
from embeddings import Embeddings
from similarity import Similarity
from plot import Plotter


class Runner:
    def __init__(self, embedding_type, training_vars, singleCompany):
        """
        Class to run the project.

        Args:
            embedding_type (str): Type of embedding to use.
            training_vars (dict): Dictionary containing the training variables.
            singleCompany (str, None): If None, run for all companies.
            If str, run for that single company.
        """
        self.corpus = DataSet(singleCompany=singleCompany)
        self.vocab = self.corpus.vocab
        self.embedding_type = embedding_type
        self.training_vars = training_vars

    def run(self, singleCompany):
        """
        Runs the project.

        Args:
            singleCompany (str, None): If None, run for all companies.
            If str, run for that single company.
        """
        if not singleCompany:
            print("Running for all companies")
            sim_list = []
            sim_dict = {}

            for company in self.corpus.df["Company"].unique():
                train = self.corpus.train[self.corpus.train["Company"] == company]
                test = self.corpus.test[self.corpus.test["Company"] == company]
                pres_train, qa_train = self.corpus.split_pres_qa(train)
                pres_test, qa_test = self.corpus.split_pres_qa(test)
                # pres_train format: [Report#(0-15)[Para[Word]]]
                # qa_train format: [Report#(0-15)[(Ques[Word], Ans[Word])]]
                # wordfreq = WordFreq(pres_train, qa_train)
                # wordfreq.count_words(company)

                embeddings = Embeddings(
                    self.corpus,
                    pres_train,
                    qa_train,
                    pres_test,
                    qa_test,
                    self.training_vars,
                )

                num_reports = len(pres_train) + len(pres_test)
                if self.embedding_type == "tfidf":
                    emmbedings = embeddings.embedding_matrix(company, "tfidf")
                    similarity = Similarity(
                        num_reports, emmbedings.tfidf_dict, sim_dict, sim_list
                    )
                elif self.embedding_type == "lstm":
                    embeddings = embeddings.embedding_matrix(company, "lstm")
                    similarity = Similarity(
                        num_reports, emmbedings.lstm_dict, sim_dict, sim_list
                    )
                elif self.embedding_type == "gru":
                    embeddings = embeddings.embedding_matrix(company, "gru")
                    similarity = Similarity(
                        num_reports, emmbedings.gru_dict, sim_dict, sim_list
                    )
                elif self.embedding_type == "sa":
                    embeddings = embeddings.embedding_matrix(company, "sa")
                    similarity = Similarity(
                        num_reports, emmbedings.sa_dict, sim_dict, sim_list
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
            test = self.corpus.test[self.corpus.test["Company"] == company]
            pres_train, qa_train = self.corpus.split_pres_qa(train)
            pres_test, qa_test = self.corpus.split_pres_qa(test)
            # pres_train format: [Report#(0-15)[Para[Word]]]
            # qa_train format: [Report#(0-15)[(Ques[Word], Ans[Word])]]
            # wordfreq = WordFreq(pres_train, qa_train)
            # wordfreq.count_words(company)

            embeddings = Embeddings(
                self.corpus,
                pres_train,
                qa_train,
                pres_test,
                qa_test,
                self.training_vars,
            )

            num_reports = len(pres_train) + len(pres_test)
            if self.embedding_type == "tfidf":
                emmbedings = embeddings.embedding_matrix(company, "tfidf")
                similarity = Similarity(
                    num_reports, emmbedings.tfidf_dict, sim_dict, sim_list
                )
            elif self.embedding_type == "lstm":
                embeddings = embeddings.embedding_matrix(company, "lstm")
                similarity = Similarity(
                    num_reports, emmbedings.lstm_dict, sim_dict, sim_list
                )
            elif self.embedding_type == "gru":
                embeddings = embeddings.embedding_matrix(company, "gru")
                similarity = Similarity(
                    num_reports, emmbedings.gru_dict, sim_dict, sim_list
                )
            elif self.embedding_type == "sa":
                embeddings = embeddings.embedding_matrix(company, "sa")
                similarity = Similarity(
                    num_reports, emmbedings.sa_dict, sim_dict, sim_list
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
