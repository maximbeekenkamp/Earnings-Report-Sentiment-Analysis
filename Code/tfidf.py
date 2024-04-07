from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class TfIdf:
    def __init__(self, pres_text, qa_text, similarity_ranked):
        self.pres_text = pres_text
        self.qa_text = qa_text
        self.similarity_dict = {}
        self.tfidf_dict = {}
        self.similarity_ranked = similarity_ranked
        
    
    def similarity(self, company):
        #TODO: Implement the document wide cosine similarity function
        self.similarity_dict[company] = {}
        self.tfidf_dict[company] = {}

        vectorizer = TfidfVectorizer()
        self.tfidf_dict[company]["Presentation"] = vectorizer.fit_transform(self.pres_text)
        self.tfidf_dict[company]["QA"] = vectorizer.transform(self.qa_text)

        self.similarity_dict[company] = cosine_similarity(
            self.tfidf_dict[company]["Presentation"],
            self.tfidf_dict[company]["QA"]
        )
        if not self.similarity_ranked:
            self.similarity_ranked.append((company, self.similarity_dict[company]))
        else:
            for i, (_, sim) in enumerate(self.similarity_ranked):
                if self.similarity_dict[company] > sim:
                    self.similarity_ranked.insert(i, (company, self.similarity_dict[company]))
                    break
            else:
                self.similarity_ranked.append((company, self.similarity_dict[company]))
        
        pass
        