import numpy as np
import tensorflow as tf


class AttentionMatrix(tf.keras.layers.Layer):
    def __init__(self, *args, use_mask=True, **kwargs):
        """
        Class to compute the attention matrix.

        Args:
            use_mask (bool, optional): Whether to apply masking to the
            attention score matrix. When True, self-attention is applied.
            When False, cross-attention is applied. Defaults to True.
        """
        super().__init__(*args, **kwargs)
        self.use_mask = use_mask

    def call(self, inputs):
        """
        Computes the attention weights matrix for the given key and query tensors.

        Args:
            inputs (tuple): Tuple containing the key and query tensors.
            - K is [batch_size x window_size_keys x embedding_size]
            - Q is [batch_size x window_size_queries x embedding_size]

        Returns:
            tf.Tensor: Attention weights matrix.
        """
        K, Q = inputs
        window_size_queries = Q.get_shape()[1]
        window_size_keys = K.get_shape()[1]

        # - Mask is [batch_size x window_size_queries x window_size_keys]
        mask = np.triu(np.ones((window_size_queries, window_size_keys)) * np.NINF, k=1)
        atten_mask = tf.convert_to_tensor(value=mask, dtype=tf.float32)

        if self.use_mask == True:
            score = tf.nn.softmax(
                (tf.matmul(Q, tf.transpose(K, perm=[0, 2, 1])) + atten_mask)
                / (window_size_keys**0.5)
            )
        else:
            score = tf.nn.softmax(
                tf.matmul(Q, tf.transpose(K, perm=[0, 2, 1])) / (window_size_keys**0.5)
            )

        return score


class AttentionHead(tf.keras.layers.Layer):
    def __init__(self, input_size, output_size, is_self_attention=True, **kwargs):
        """
        Class to compute the attention head. Initialises the weights matrices for 
        the keys, values, and queries.

        Args:
            input_size (int): The size of the input embeddings tensor.
            output_size (int): The size of the output embeddings tensor.
            is_self_attention (bool, optional): Whether to apply mask in AttentionMatrix.
            True for self-attention, False for cross-attention. Defaults to True.
        """
        super(AttentionHead, self).__init__(**kwargs)
        self.use_mask = is_self_attention

        self.K = self.add_weight(
            shape=(input_size, output_size), initializer="glorot_normal"
        )
        self.V = self.add_weight(
            shape=(input_size, output_size), initializer="glorot_normal"
        )
        self.Q = self.add_weight(
            shape=(input_size, output_size), initializer="glorot_normal"
        )

        self.attention_mat = AttentionMatrix(use_mask=self.use_mask)

    @tf.function
    def call(self, inputs_for_keys, inputs_for_values, inputs_for_queries):
        """
        Runs a single attention head.

        Args:
            inputs_for_keys (tf.Tensor): Input tensor for keys. 
            Shape: [batch_size x window_size_keys x embedding_size]
            inputs_for_values (tf.Tensor): Input tensor for values.
            Shape: [batch_size x window_size_keys x embedding_size]
            inputs_for_queries (tf.Tensor): Input tensor for queries.
            Shape: [batch_size x window_size_queries x embedding_size]

        Returns:
            tf.Tensor: Attention head. 
            Shape: [batch_size x window_size_queries x output_size]
        """
        K = tf.tensordot(inputs_for_keys, self.K, 1)
        V = tf.tensordot(inputs_for_values, self.V, 1)
        Q = tf.tensordot(inputs_for_queries, self.Q, 1)

        return tf.matmul(self.attention_mat((K, Q)), V)

class MultiHeadedAttention(tf.keras.layers.Layer):
    def __init__(self, emb_sz, num_heads, use_mask=True, **kwargs):
        """
        Class to compute multi-headed attention.

        Args:
            emb_sz (int): The size of the embedding.
            num_heads (int): The number of attention heads.
            use_mask (bool, optional): Whether to apply mask in AttentionMatrix.
            True for self-attention, False for cross-attention. Defaults to True.
        """
        super(MultiHeadedAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.emb_sz = emb_sz
        assert (
            emb_sz % num_heads == 0
        ), "Embedding size must be divisible by number of heads."

        self.head_size = emb_sz // num_heads
        self.attention_heads = [
            AttentionHead(self.head_size, self.head_size, True)
            for _ in range(num_heads)
        ]
        self.linear = tf.keras.layers.Dense(emb_sz)

    @tf.function
    def call(self, inputs_for_keys, inputs_for_values, inputs_for_queries):
        """
        Computes the multi-headed attention.

        Args:
            inputs_for_keys (tf.Tensor): Input tensor for keys.
            Shape: [batch_size x window_size_keys x embedding_size]
            inputs_for_values (tf.Tensor): Input tensor for values.
            Shape: [batch_size x window_size_keys x embedding_size]
            inputs_for_queries (tf.Tensor): Input tensor for queries.
            Shape: [batch_size x window_size_queries x embedding_size]

        Returns:
            tf.Tensor: Multi-headed attention.
            Shape: [batch_size x window_size_queries x output_size]
        """
        attention_heads_output = [
            attention_head(inputs_for_keys, inputs_for_values, inputs_for_queries)
            for attention_head in self.attention_heads
        ]
        concat_attention = tf.concat(attention_heads_output, axis=-1)
        return self.linear(concat_attention)

class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, emb_sz, num_heads=1, **kwargs):
        """
        Implements a transformer block layer.

        Args:
            emb_sz (int): The size of the embedding.
            num_heads (int, optional): The number of attention heads. Defaults to 1.
        """
        super(TransformerBlock, self).__init__(**kwargs)
        self.emb_sz = emb_sz
        self.num_heads = num_heads

        self.ff_layer = tf.keras.layers.Dense(emb_sz, activation="relu")

        if num_heads == 1:
            self.self_atten = AttentionHead(emb_sz, emb_sz)
            self.self_context_atten = AttentionHead(emb_sz, emb_sz, False) 

        else:
            self.multi_atten = MultiHeadedAttention(emb_sz, num_heads)
            self.self_context_atten_multi = MultiHeadedAttention(emb_sz, num_heads, False)

        

        self.layer_norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layer_norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layer_norm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        

    @tf.function
    def call(self, inputs, decode_bool):
        """
        Runs the transformer block layer.
        See reference for more details.
        https://www.tensorflow.org/text/tutorials/transformer#the_embedding_and_positional_encoding_layer

        Args:
            inputs (tf.Tensor): Input tensor.
            Shape: [batch_size x input_seq_length x embedding_size]
            decode_bool (bool): Boolean to determine whether to apply 
            self-attention or not.

        Returns:
            tf.Tensor: Output tensor.
            Shape: [batch_size x input_seq_length x embedding_size]
        """
        if decode_bool:
            if self.num_heads == 1:
                masked_attention = self.self_atten(inputs, inputs, inputs)
            else:
                masked_attention = self.multi_atten(inputs, inputs, inputs)
            out1 = self.layer_norm1(inputs + masked_attention)
        if self.num_heads == 1:
            un_masked_attention = self.self_context_atten(out1, out1, out1)
        else:
            un_masked_attention = self.self_context_atten_multi(out1, out1, out1)
        out2 = self.layer_norm2(out1 + un_masked_attention)
            
        ffout = self.ff_layer(out2)
        out3 = self.layer_norm2(out2 + ffout)
        return out3

class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, max_seq_len, embed_size):
        """
        Class to add positional encoding to the input embeddings.

        Args:
            max_seq_len (int): The maximum sequence length.
            embed_size (int): Embedding size.
        """
        super().__init__()
        self.pos_encoding = self.positional_encoding(max_seq_len, embed_size)

    def call(self, x):
        """
        Adds positional encoding to the input embeddings.

        Args:
            x (tf.Tensor): Input tensor.

        Returns:
            tf.Tensor: Output tensor with positional encoding added.
        """
        return x + self.pos_encoding[: tf.shape(x)[1], :]

    def positional_encoding(self, length, depth):
        """
        Generates positional encoding.
        See reference for more details.
        https://www.tensorflow.org/text/tutorials/transformer#the_embedding_and_positional_encoding_layer

        Args:
            length (int): Length of sequence.
            depth (int): Depth of embeddings.

        Returns:
            tf.Tensor: Positional encodings.
        """
        depth = depth / 2
        positions = np.arange(length)[:, np.newaxis]  # (seq, 1)
        depths = np.arange(depth)[np.newaxis, :] / depth  # (1, depth)
        angle_rates = 1 / (10000**depths)  # (1, depth)
        angle_rads = positions * angle_rates  # (pos, depth)
        pos_encoding = np.concatenate([np.sin(angle_rads), np.cos(angle_rads)], axis=-1)
        return tf.cast(pos_encoding, dtype=tf.float32)

class Encoder:
    def __init__(self, training_vars):
        """
        Class to create the encoder model.

        Args:
            num_heads (int): The number of attention heads.
            embedding_size (int): The size of the embedding.
        """
        self.num_heads = training_vars["num_heads"]
        self.embedding_size = training_vars["embedding_size"]
        self.max_seq_len = training_vars["max_seq_len"]
        self.num_layers = training_vars["num_layers"]
        self.positional_encoding = PositionalEncoding(self.max_seq_len, self.embedding_size)
        self.transformer = TransformerBlock(self.embedding_size, self.num_heads)

    def call(self, inputs):
        """
        Creates the encoder model.

        Returns:
            tf.keras.Model: Encoder model.
        """
        x = self.positional_encoding(inputs)
        for _ in range(self.num_layers):
            x = self.transformer(x, False)
        return x

class Decoder:
    def __init__(self, training_vars):
        """
        Class to create the decoder model.

        Args:
            num_heads (int): The number of attention heads.
            embedding_size (int): The size of the embedding.
        """
        self.num_heads = training_vars["num_heads"]
        self.embedding_size = training_vars["embedding_size"]
        self.max_seq_len = training_vars["max_seq_len"]
        self.num_layers = training_vars["num_layers"]
        self.positional_encoding = PositionalEncoding(self.max_seq_len, self.embedding_size)
        self.transformer = TransformerBlock(self.embedding_size, self.num_heads)

    def call(self, inputs):
        """
        Creates the decoder model.

        Returns:
            tf.keras.Model: Decoder model.
        """
        x = self.positional_encoding(inputs)
        for _ in range(self.num_layers):
            x = self.transformer(x, True)
        return x