import tensorflow as tf
import numpy as np
from transformer import SA_Encoder, SA_Decoder

def create_dummy_input(batch_size, seq_len, vocab_size):
    return tf.constant(np.random.randint(0, vocab_size, size=(batch_size, seq_len)), dtype=tf.int32)

def create_training_vars():
    return {
        "num_heads": 2,
        "embedding_size": 32,
        "seq_len": 20,
        "num_layers": 3,
        "vocab_size": 100
    }

def test_sa_encoder_decoder():
    training_vars = create_training_vars()
    sa_encoder = SA_Encoder(training_vars)

    sa_decoder = SA_Decoder(training_vars)

    batch_size = 4
    seq_len = 10
    vocab_size = training_vars["vocab_size"]
    encoder_inputs = create_dummy_input(batch_size, seq_len, vocab_size)
    decoder_inputs = create_dummy_input(batch_size, seq_len, vocab_size)

    encoded_output = sa_encoder(encoder_inputs)

    decoder_output = sa_decoder(decoder_inputs, encoded_output)

    expected_shape_encoder = (batch_size, seq_len, training_vars["embedding_size"])
    assert encoded_output.shape == expected_shape_encoder, f"Expected shape for encoder output: {expected_shape_encoder}, Actual shape: {encoded_output.shape}"

    expected_shape_decoder = (batch_size, seq_len, training_vars["embedding_size"])
    assert decoder_output.shape == expected_shape_decoder, f"Expected shape for decoder output: {expected_shape_decoder}, Actual shape: {decoder_output.shape}"

    print("SA_Encoder and SA_Decoder test passed.")

if __name__ == "__main__":
    test_sa_encoder_decoder()
