import tensorflow as tf
import numpy as np
from transformer import SA_Encoder  # Replace 'your_module' with the actual name of your module containing the SA_Encoder class

# Define a function to create a dummy input tensor
def create_dummy_input(batch_size, seq_len, vocab_size):
    return tf.constant(np.random.randint(0, vocab_size, size=(batch_size, seq_len)), dtype=tf.int32)

# Define a function to create dummy training variables
def create_training_vars():
    return {
        "num_heads": 2,
        "embedding_size": 32,
        "seq_len": 20,
        "num_layers": 3,
        "vocab_size": 100
    }

# Define a test case for SA_Encoder
def test_sa_encoder():
    # Create a SA_Encoder instance
    training_vars = create_training_vars()
    sa_encoder = SA_Encoder(training_vars)

    # Create a dummy input tensor
    batch_size = 4
    seq_len = 10
    vocab_size = training_vars["vocab_size"]
    inputs = create_dummy_input(batch_size, seq_len, vocab_size)

    # Pass the input through the SA_Encoder
    encoded_output = sa_encoder(inputs)

    # Check the shape of the encoded output
    expected_shape = (batch_size, seq_len, training_vars["embedding_size"])
    assert encoded_output.shape == expected_shape, f"Expected shape: {expected_shape}, Actual shape: {encoded_output.shape}"

    print("SA_Encoder test passed.")

# Run the test
if __name__ == "__main__":
    test_sa_encoder()
