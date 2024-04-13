from preprocessing import DataSet
from word_freq import WordFreq

from embeddings import Embeddings
from similarity import Similarity


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
            for company in self.corpus.df["Company"].unique():
                train = self.corpus.train[self.corpus.train["Company"] == company]
                pres_train, qa_train = self.corpus.split_pres_qa(train)
                # pres_train format: [Report#(0-15)[Para[Word[str]]]]
                # qa_train format: [Report#(0-15)[(Ques[Word[str]], Ans[Word[str]])]]

                tfidf_emmbedings = Embeddings(pres_train, qa_train, self.vocab[company])
                tfidf_emmbedings.embedding_matrix(company, "tfidf")

                similarity = Similarity(pres_train, tfidf_emmbedings.tfidf_dict, sim_list, self.vocab[company])
                similarity.sim_score(company)


                word_freq = WordFreq(pres_train, qa_train, self.corpus)
                word_freq.count_words(company)

            print(
                "Similarity scores between Presentation and Q&A sections for all companies:"
            )
            for comp, sim in similarity.similarity_ranked:
                print(f"{comp}: {sim}")

        else:
            company = singleCompany
            print(f"Running for {company}")
            df = self.corpus.df[self.corpus.df["Company"] == company]
            pres_train, qa_train = self.corpus.split_pres_qa(df)

            tfidf_emmbedings = Embeddings(pres_train, qa_train, self.vocab[company])
            tfidf_emmbedings.embedding_matrix(company, "tfidf")

            similarity = Similarity(pres_train, tfidf_emmbedings.tfidf_dict, [], self.vocab[company])
            similarity.sim_score(company)

            word_freq = WordFreq(pres_train, qa_train, self.corpus)
            word_freq.count_words(company)

            print(
                f"Similarity for {company} between Presentation and Q&A \
                sections: {similarity.similarity_ranked[0][1]}"
            )
