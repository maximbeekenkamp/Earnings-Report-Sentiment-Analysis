from sklearn.feature_extraction.text import TfidfVectorizer
import tensorflow as tf

from transformer import Decoder, Encoder
from autoencoder import VAE

class Tokens:
    def __init__(self, corpus, company):
        """
        Class to create tokens for the presentation and QA sections of
        a single company's report.

        Args:
            corpus (DataSet): DataSet class object containing the data.
            company (str): A company's ticker symbol.
        """
        self.word_to_token_dict = corpus.vocab
        self.word_to_token_dict = self.word_to_token_dict[company]
        self.token_to_word_dict = {"Presentation": {}, "QA": {}}
        for section in self.token_to_word_dict.keys():
            self.token_to_word_dict[section] = {
                v: k for k, v in self.word_to_token_dict[section].items()
            }

class Embeddings:
    def __init__(self, pres_df, qa_df, training_params):
        """
        Class to create embeddings for the Presentation and QA
        sections of a company's report.

        Args:
            pres_df (df): df containing a list of paragraphs containing a list
            of words from the presentation section.
            qa_df (df): df containing a list of question answer tuples containing
            lists of words from the Q&A section.
            training_params (dict): dictionary containing the training parameters.
        """
        self.pres_list = list(pres_df)
        self.qa_list = list(qa_df)
        self.tfidf_dict = {}
        self.training_params = training_params

    def embedding_matrix(self, company, mode):
        """
        Passes a single report to the similarity_per_report function to calculate
        the similarity between the presentation and QA sections.

        Args:
            company (str): Companies ticker symbol.
            mode (str): Mode to choose between the embedding method. Choose between
            'tfidf' and 'doc2vec'.
        """
        self.tfidf_dict[company] = {}
        assert len(self.pres_list) == len(
            self.qa_list
        ), "Same number of reports required."
        for i, pres in enumerate(self.pres_list):
            if mode == "tfidf":
                self.tfidf(company, pres, self.qa_list[i], i)
            elif mode == "lstm":
                self.lstm_embed(company, pres, self.qa_list[i], i)
            elif mode == "gru":
                self.gru_embed(company, pres, self.qa_list[i], i)
            elif mode == "sa":
                self.sa_embed(company, pres, self.qa_list[i], i, self.training_params)
            else:
                raise ValueError("Invalid mode. Please choose 'tfidf' or 'doc2vec'.")

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

    def lstm_embed(self, company, pres, qa, i):
        pass

    def gru_embed(self, company, pres, qa, i):
        pass

    def sa_embed(self, company, pres, qa, i, training_vars):
        # Define your transformer parameters
        # embedding_size = 128
        # num_heads = 8
        # input_sequence_length = 100
        # output_sequence_length = 100

        # training_vars = {
        #     'num_heads': num_heads,
        #     'embedding_size': embedding_size,
        #     'input_sequence_length': input_sequence_length,
        #     'output_sequence_length': output_sequence_length,
        # }
        # Create your transformer model
        encoder = Encoder(training_vars)
        decoder = Decoder(training_vars)

        # latent_dim = 64

        # # Create your autoencoder model
        encoder_layers = tf.keras.Sequential([
            encoder,
            tf.keras.layers.Dense(training_vars["latent_dim"], activation="relu"),
        ])
        decoder_layers = tf.keras.Sequential([
            tf.keras.layers.Dense(training_vars["embedding_size"], activation="relu"),
            decoder,
        ])
        mu_layers = tf.keras.layers.Dense(training_vars["latent_dim"])
        logvar_layers = tf.keras.layers.Dense(training_vars["latent_dim"])
        autoencoder = VAE(encoder_layers, decoder_layers, mu_layers, logvar_layers)
        
        autoencoder.compile(
            optimizer= tf.keras.optimizers.Adam(training_vars["learning_rate"]),
            rec_loss=self.rec_loss, 
            kld_loss=self.kld_loss, 
            metrics= [
                tf.keras.metrics.MeanSquaredError(),
                tf.keras.metrics.BinaryCrossentropy()
                ]
        )

        # Generate some sample tokenized data (replace this with your actual data loading process)
        # Assuming tokenized_data is a numpy array of shape (num_samples, max_seq_length, embedding_size)
        num_samples = 1000
        max_seq_length = 100
        embedding_size = 128
        tokenized_data = np.random.rand(num_samples, max_seq_length, embedding_size)

        # Train the autoencoder
        autoencoder.fit((tokenized_data, None), tokenized_data, epochs=10, batch_size=32)

        # Train the transformer model using the autoencoder
        transformer_model.fit(tokenized_data, autoencoder.encoder.predict(tokenized_data), epochs=10, batch_size=32)


    def kld_loss(self, mu, logvar):
        """
        Computes the Kullback-Leibler divergence loss.

        Args:
            mu (tf.Tensor): Mean of the latent space.
            logvar (tf.Tensor): Log variance of the latent space.

        Returns:
            tf.Tensor: The Kullback-Leibler divergence loss.
        """
        return 0.5 * tf.reduce_sum(-logvar + (mu ** 2) -1 + tf.exp(logvar), axis=1)
    
    def rec_loss(self, x_true, x_pred):
        return tf.reduce_sum(tf.keras.losses.binary_crossentropy(x_true, x_pred), axis=(1, 2))