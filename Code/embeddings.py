from sklearn.feature_extraction.text import TfidfVectorizer
import tensorflow as tf
from tqdm import tqdm

from autoencoder import VAE
from rnn import MyLSTM, MyGRU
from transformer import MHA, PositionalEncoding


class Tokens:
    def __init__(self, corpus, company, total_bool=False):
        """
        Class to create tokens for the presentation and QA sections of
        a single company's report.

        Args:
            corpus (DataSet): DataSet class object containing the data.
            company (str): A company's ticker symbol.
            total_bool (bool, optional): Boolean to choose whether to split the tokens into
            presentation and QA sections. False if splitting is desired. Defaults to False.
        """
        if total_bool:
            self.word_to_token_dict = corpus.vocab_total[company]
            self.token_to_word_dict = {v: k for k, v in self.word_to_token_dict.items()}
        else:
            self.word_to_token_dict = corpus.vocab[company]
            self.token_to_word_dict = {"Presentation": {}, "QA": {}}
            for section in self.token_to_word_dict.keys():
                self.token_to_word_dict[section] = {
                    v: k for k, v in self.word_to_token_dict[section].items()
                }

    def tokenise(self, pres, qa):
        """
        Tokenises the input data.

        Args:
            pres (list): Presentation data.
            Format:
            [Report#(0-15)[Para[Word]]]
            qa (list): QA data.
            Format:
            [Report#(0-15)[(Ques[Word], Ans[Word])]]

        Returns:
            list: The tokenised data.
            Format:
            [Para/Question/Answer[Token]]
        """
        data = []
        max_sequence_len = 0
        for report_num, (pres, qa) in enumerate(zip(pres, qa)):
            for para in pres:
                data.append([self.word_to_token_dict[word] for word in para])
                if len(para) > max_sequence_len:
                    max_sequence_len = len(para)
            for ques, ans in qa:
                data.append([self.word_to_token_dict[word] for word in ques])
                if len(ques) > max_sequence_len:
                    max_sequence_len = len(ques)
                data.append([self.word_to_token_dict[word] for word in ans])
                if len(ans) > max_sequence_len:
                    max_sequence_len = len(ans)
        return data, max_sequence_len


class Embeddings:
    def __init__(
        self,
        corpus,
        pres_train_df,
        qa_train_df,
        pres_test_df,
        qa_test_df,
        training_vars,
    ):
        """
        Class to create embeddings for the Presentation and QA
        sections of a company's report.

        Args:
            corpus (DataSet): DataSet class object containing the data.
            pres_df (df): df containing a list of paragraphs containing a list
            of words from the presentation section.
            qa_df (df): df containing a list of question answer tuples containing
            lists of words from the Q&A section.
            training_vars (dict): dictionary containing the training variables.
        """
        self.corpus = corpus
        self.pres_train_list = list(pres_train_df)
        self.qa_train_list = list(qa_train_df)
        assert len(self.pres_train_list) == len(
            self.qa_train_list
        ), "Same number of reports required."

        self.pres_test_list = list(pres_test_df)
        self.qa_test_list = list(qa_test_df)
        assert len(self.pres_test_list) == len(
            self.qa_test_list
        ), "Same number of reports required."

        self.tfidf_dict = {}

        self.lstm_dict = {}
        self.lstm_decode_dict = {}

        self.gru_dict = {}
        self.gru_decode_dict = {}

        self.sa_dict = {}
        self.sa_decode_dict = {}

        self.training_vars = training_vars
        self.non_trainable_emmbedding = None
        self.company_tokens = None

    def embedding_matrix(self, company, mode):
        """
        Passes a single report to the similarity_per_report function to calculate
        the similarity between the presentation and QA sections.

        Args:
            company (str): Companies ticker symbol.
            mode (str): Mode to choose between the embedding method. Choose between
            'tfidf', 'lstm', 'gru', or 'sa'.
        """
        pres_list = self.pres_train_list
        for report in self.pres_test_list:
            pres_list.append(report)

        qa_list = self.qa_train_list
        for report in self.qa_test_list:
            qa_list.append(report)

        if mode == "tfidf":
            self.tfidf_dict[company] = {}
            for i, pres in enumerate(pres_list):
                self.tfidf(company, pres, qa_list[i], i)
        else:
            self.company_tokens = Tokens(self.corpus, company, total_bool=True)
            self.training_vars["vocab_size"] = len(
                self.company_tokens.word_to_token_dict
            )
            train_data, max_seq_1 = self.company_tokens.tokenise(
                self.pres_train_list, self.qa_train_list
            )
            val_data, max_seq_2 = self.company_tokens.tokenise(
                self.pres_test_list, self.qa_test_list
            )

            max_sequence_len = max(max_seq_1, max_seq_2)
            if max_sequence_len > self.training_vars["seq_len"]:
                max_sequence_len = self.training_vars["seq_len"]
            self.training_vars["seq_len"] = max_sequence_len

            self.non_trainable_emmbedding = tf.keras.layers.Embedding(
                input_dim=self.training_vars["vocab_size"] + 1,
                output_dim=self.training_vars["embedding_size"],
                # mask_zero=True,
                trainable=False,
            )

            train_data = tf.keras.utils.pad_sequences(
                train_data,
                maxlen=self.training_vars["seq_len"],
                padding="post",
                truncating="post",
                value=0,
                dtype="float32",
            )
            train_data = self.non_trainable_emmbedding(train_data)
            train_data = tf.data.Dataset.from_tensor_slices(
                train_data, name=f"Train_{company}"
            ).batch(self.training_vars["batch_size"], drop_remainder=True)
            train_data = train_data.shuffle(len(train_data))

            val_data = tf.keras.utils.pad_sequences(
                val_data,
                maxlen=self.training_vars["seq_len"],
                padding="post",
                truncating="post",
                value=0,
                dtype="float32",
            )
            val_data = self.non_trainable_emmbedding(val_data)
            val_data = tf.data.Dataset.from_tensor_slices(
                val_data, name=f"Train_{company}"
            ).batch(self.training_vars["batch_size"], drop_remainder=True)
            val_data = val_data.shuffle(len(val_data))

            func_inputs = (
                company,
                self.training_vars,
                pres_list,
                qa_list,
                train_data,
                val_data,
            )
            if mode == "lstm":
                self.lstm_embed(*func_inputs)
            elif mode == "gru":
                self.gru_embed(*func_inputs)
            elif mode == "sa":
                self.sa_embed(*func_inputs)
            else:
                raise ValueError(
                    "Invalid mode. Please choose 'tfidf', 'lstm', 'gru', or 'sa'."
                )

    def tfidf(self, company, pres, qa, i):
        """
        Calculates the cosine similarity between the presentation and QA sections
        of a single report.

        Args:
            company (str): Companies ticker symbol.
            pres (list): Presentation section of a single report.
            Format:
            [Para[Word]]
            qa (list): QA section of a single report.
            Format:
            [(Ques[Word], Ans[Word])]
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
        except Exception as e:
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
        except Exception as e:
            output_msg = "Error in QA"
            output_msg += f"\n Company: {company} \n Report number: {i}"
            for ques, ans in qa:
                output_msg += f"\n QUES: {ques} \n ANS: {ans}"
            raise ValueError(f"{output_msg}")

    def lstm_embed(
        self, company, training_vars, pres_list, qa_list, train_data, val_data
    ):
        """
        Creates the contextualised embeddings for the Presentation and QA sections using the
        LSTM model.

        Args:
            company (str): The name of the company.
            training_vars (dict): A dictionary containing training variables.
            pres_list (list): The entire list of presentation sections.
            qa_list (list): The entire list of QA sections.
            train_data (list): The train data containing both the presentation and QA sections.
            val_data (list): The test data containing both the presentation and QA sections.

        Returns:
            None, but updates the self.lstm_dict and self.lstm_decode_dict dictionaries.
        """

        def train(train_data, val_data, training_vars):
            """
            Trains the VAE model using the given training data and validation data.

            Args:
                train_data (list: The training data.
                val_data (list): The validation data.
                training_vars (dict): A dictionary containing the training variables.

            Returns:
                VAE: The trained VAE model.
            """
            encoder = CustomSequential()
            decoder = CustomSequential()

            encoder.add(
                PositionalEncoding(
                    training_vars["vocab_size"],
                    training_vars["embedding_size"],
                    training_vars["seq_len"],
                )
            )
            encoder.add(MyLSTM(units=training_vars["embedding_size"]))
            encoder.add(
                tf.keras.layers.Dense(training_vars["latent_dim"], activation="relu")
            )

            decoder.add(
                tf.keras.layers.Dense(
                    (training_vars["latent_dim"] + training_vars["embedding_size"])
                    // 2,
                    activation="relu",
                )
            )
            decoder.add(
                tf.keras.layers.Dense(
                    training_vars["embedding_size"], activation="relu"
                )
            )

            mu_layers = tf.keras.layers.Dense(training_vars["latent_dim"])
            logvar_layers = tf.keras.layers.Dense(training_vars["latent_dim"])
            vae = VAE(encoder, decoder, mu_layers, logvar_layers)

            vae.compile(
                optimizer=tf.keras.optimizers.Adam(training_vars["learning_rate"]),
                rec_loss=self.rec_loss,
                kld_loss=self.kld_loss,
                metrics=[
                    tf.keras.metrics.MeanSquaredError(),
                    tf.keras.metrics.BinaryCrossentropy(),
                ],
            )

            vae.fit(
                x=train_data,
                epochs=training_vars["vae epochs"],
                batch_size=training_vars["batch_size"],
                validation_data=(val_data),
            )

            return vae

        vae = train(train_data, val_data, training_vars)

        self.lstm_dict, self.lstm_decode_dict = self.context_embed(
            company, pres_list, qa_list, self.lstm_dict, self.lstm_decode_dict, vae
        )

    def gru_embed(
        self, company, training_vars, pres_list, qa_list, train_data, val_data
    ):
        """
        Creates the contextualised embeddings for the Presentation and QA sections using the
        GRU model.

        Args:
            company (str): The name of the company.
            training_vars (dict): A dictionary containing training variables.
            pres_list (list): The entire list of presentation sections.
            qa_list (list): The entire list of QA sections.
            train_data (list): The train data containing both the presentation and QA sections.
            val_data (list): The test data containing both the presentation and QA sections.

        Returns:
            None, but updates the self.gru_dict and self.gru_decode_dict dictionaries.
        """

        def train(train_data, val_data, training_vars):
            """
            Trains the VAE model using the given training data and validation data.

            Args:
                train_data (list: The training data.
                val_data (list): The validation data.
                training_vars (dict): A dictionary containing the training variables.

            Returns:
                VAE: The trained VAE model.
            """
            encoder = CustomSequential()
            decoder = CustomSequential()

            encoder.add(
                PositionalEncoding(
                    training_vars["vocab_size"],
                    training_vars["embedding_size"],
                    training_vars["seq_len"],
                )
            )
            encoder.add(MyGRU(units=training_vars["embedding_size"]))
            encoder.add(
                tf.keras.layers.Dense(training_vars["latent_dim"], activation="relu")
            )

            decoder.add(
                tf.keras.layers.Dense(
                    (training_vars["latent_dim"] + training_vars["embedding_size"])
                    // 2,
                    activation="relu",
                )
            )
            decoder.add(
                tf.keras.layers.Dense(
                    training_vars["embedding_size"], activation="relu"
                )
            )

            mu_layers = tf.keras.layers.Dense(training_vars["latent_dim"])
            logvar_layers = tf.keras.layers.Dense(training_vars["latent_dim"])
            vae = VAE(encoder, decoder, mu_layers, logvar_layers)

            vae.compile(
                optimizer=tf.keras.optimizers.Adam(training_vars["learning_rate"]),
                rec_loss=self.rec_loss,
                kld_loss=self.kld_loss,
                metrics=[
                    tf.keras.metrics.MeanSquaredError(),
                    tf.keras.metrics.BinaryCrossentropy(),
                ],
            )

            vae.fit(
                x=train_data,
                epochs=training_vars["vae epochs"],
                batch_size=training_vars["batch_size"],
                validation_data=(val_data),
            )

            return vae

        vae = train(train_data, val_data, training_vars)

        self.gru_dict, self.gru_decode_dict = self.context_embed(
            company, pres_list, qa_list, self.gru_dict, self.gru_decode_dict, vae
        )

    def sa_embed(
        self, company, training_vars, pres_list, qa_list, train_data, val_data
    ):
        """
        Creates the contextualised embeddings for the Presentation and QA sections using the
        transformer model.

        Args:
            company (str): The name of the company.
            training_vars (dict): A dictionary containing training variables.
            pres_list (list): The entire list of presentation sections.
            qa_list (list): The entire list of QA sections.
            train_data (list): The train data containing both the presentation and QA sections.
            val_data (list): The test data containing both the presentation and QA sections.

        Returns:
            None, but updates the self.sa_dict and self.sa_decode_dict dictionaries.
        """

        def train(train_data, val_data, training_vars):
            """
            Trains the VAE model using the given training data and validation data.

            Args:
                train_data (list: The training data.
                val_data (list): The validation data.
                training_vars (dict): A dictionary containing the training variables.

            Returns:
                VAE: The trained VAE model.
            """
            encoder = CustomSequential()
            decoder = CustomSequential()

            encoder.add(MHA(training_vars))
            encoder.add(
                tf.keras.layers.Dense(
                    training_vars["embedding_size"], activation="relu"
                )
            )
            encoder.add(
                tf.keras.layers.Dense(training_vars["latent_dim"], activation="relu")
            )

            decoder.add(
                tf.keras.layers.Dense(
                    (training_vars["latent_dim"] + training_vars["embedding_size"])
                    // 2,
                    activation="relu",
                )
            )
            decoder.add(
                tf.keras.layers.Dense(
                    training_vars["embedding_size"], activation="relu"
                )
            )

            mu_layers = tf.keras.layers.Dense(training_vars["latent_dim"])
            logvar_layers = tf.keras.layers.Dense(training_vars["latent_dim"])
            vae = VAE(encoder, decoder, mu_layers, logvar_layers)

            # to allow @tf.function
            dummy = [["dummy_word1"], ["dummy_word2"]]
            self.extract_embeddings(vae, dummy)

            vae.compile(
                optimizer=tf.keras.optimizers.Adam(training_vars["learning_rate"]),
                rec_loss=self.rec_loss,
                kld_loss=self.kld_loss,
                metrics=[
                    tf.keras.metrics.MeanSquaredError(),
                    tf.keras.metrics.BinaryCrossentropy(),
                ],
            )

            vae.fit(
                x=train_data,
                epochs=training_vars["vae epochs"],
                batch_size=training_vars["batch_size"],
                validation_data=(val_data),
            )

            return vae

        vae = train(train_data, val_data, training_vars)

        self.sa_dict, self.sa_decode_dict = self.context_embed(
            company, pres_list, qa_list, self.sa_dict, self.sa_decode_dict, vae
        )

    def context_embed(
        self, company, pres_list, qa_list, embedding_dict, reconstruction_dict, vae
    ):
        """
        Stores the embeddings and reconstructions for the entire dataset.

        Args:
            company (str): The name of the company.
            pres_list (list): The entire list of presentation sections.
            qa_list (list): The entire list of QA sections.
            embedding_dict (dict): A dictionary to store the embeddings.
            reconstruction_dict (dict): A dictionary to store the reconstructions.
            vae (VAE): VAE class object, the trained model.

        Raises:
            ValueError: Catches errors in the presentation section and provides
            a more informative error message.
            ValueError: Catches errors in the QA section and provides
            a more informative error message.

        Returns:
            dict, dict: A tuple containing the embedding and reconstruction dictionaries.
        """
        embedding_dict[company] = {}
        reconstruction_dict[company] = {}
        for report_num, (pres, qa) in enumerate(zip(tqdm(pres_list, desc=f"Extracting embeddings for {company}"), qa_list)):
            embedding_dict[company][report_num] = {}
            reconstruction_dict[company][report_num] = {}
            for para in pres:
                try:
                    embedding, reconstruction = self.extract_embeddings(vae, para)
                    embedding_dict[company][report_num]["Presentation"] = embedding
                    reconstruction_dict[company][report_num]["Presentation"] = (
                        reconstruction,
                        para,
                    )
                except Exception as e:
                    output_msg = "Error in Presentation"
                    output_msg += (
                        f"\n Company: {company} \n Report number: {report_num}"
                    )
                    output_msg += f"\n {para}"
                    raise ValueError(f"{output_msg}")
            for ques, ans in qa:
                try:
                    embedding_ques, reconstruction_ques = self.extract_embeddings(
                        vae, ques
                    )
                    embedding_ans, reconstruction_ans = self.extract_embeddings(
                        vae, ans
                    )
                    embedding_dict[company][report_num]["QA"] = (
                        embedding_ques,
                        embedding_ans,
                    )
                    reconstruction_dict[company][report_num]["QA"] = (
                        reconstruction_ques,
                        ques,
                        reconstruction_ans,
                        ans,
                    )
                except Exception as e:
                    output_msg = "Error in QA"
                    output_msg += (
                        f"\n Company: {company} \n Report number: {report_num}"
                    )
                    output_msg += f"\n QUES: {ques} \n ANS: {ans}"
                    raise ValueError(f"{output_msg}")
        return embedding_dict, reconstruction_dict

    def kld_loss(self, mu, logvar):
        """
        Computes the Kullback-Leibler divergence loss.

        Args:
            mu (tf.Tensor): Mean of the latent space.
            logvar (tf.Tensor): Log variance of the latent space.

        Returns:
            tf.Tensor: The Kullback-Leibler divergence loss.
        """
        return 0.5 * tf.reduce_sum(-logvar + (mu**2) - 1 + tf.exp(logvar), axis=1)

    def rec_loss(self, x_true, x_pred):
        """
        Computes the reconstruction loss.

        Args:
            x_true (list): The true data.
            x_pred (list): The predicted data.

        Returns:
            tf.Tensor: The reconstruction loss.
        """
        # TODO: rec loss shouldnt be using binary cross entropy
        error = tf.reduce_sum(tf.keras.losses.binary_crossentropy(x_true, x_pred))
        return tf.reduce_sum(error)

    def extract_embeddings(self, vae, x):
        """
        Extracts the embeddings from the VAE model, as well as the reconstruction.

        Args:
            vae (VAE): VAE class object, the trained model.
            x (list): The input data.

        Returns:
            tuple: A tuple containing the embeddings and the reconstruction.
        """
        if isinstance(x[0], list):  # dummy variable to allow @tf.function
            for i, para in enumerate(x):
                for j, word in enumerate(para):
                    x[i][j] = 0
        else:
            for i, word in enumerate(x):
                try:
                    x[i] = self.company_tokens.word_to_token_dict[word]
                except KeyError:
                    x[i] = 0
            x = [x, [0] * self.training_vars["seq_len"]]

        assert (
            len(x) == self.training_vars["batch_size"]
        ), "extract_embeddings: shape error in x"

        x = tf.keras.utils.pad_sequences(
            x,
            maxlen=self.training_vars["seq_len"],
            padding="post",
            truncating="post",
            value=0,
            dtype="float32",
        )
        x = self.non_trainable_emmbedding(x)

        xp = vae.encoder(x)
        zp = vae.latent_ops(xp)
        z = vae.decoder(zp)
        return zp, z


class CustomSequential:
    def __init__(self):
        """
        Replacement for the tf.keras.Sequential class.
        """
        self.layers = []

    def add(self, layer):
        """
        Adds a layer to the model.

        Args:
            layer (tf.keras.layers.Layer): The layer to add.
        """
        self.layers.append(layer)

    def __call__(self, x):
        """
        Forward pass for the model.

        Args:
            x (tf.Tensor): The input data.

        Returns:
            tf.Tensor: The output tensor.
        """
        for layer in self.layers:
            x = layer(x)
        return x
