import sys
import tensorflow as tf


class VAE(tf.keras.Model):
    def __init__(self, encoder, decoder, mu_layers, logvar_layers, **kwargs):
        """
        Variational Autoencoder class.

        Args:
            encoder (tf.keras.Model): Encoder layers grouped into a linear stack.
            decoder (tf.keras.Model): Decoder layers grouped into a linear stack.
            mu_layers (tf.keras.Layer): Layers for computing the mean of the
            latent space.
            logvar_layers (tf.keras.Layer): Layers for computing the log variance
            of the latent space.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.kld_tracker = tf.keras.metrics.Mean(name="kld")
        self.rec_tracker = tf.keras.metrics.Mean(name="rec")
        self.mu_layers = mu_layers
        self.logvar_layers = logvar_layers

    def call(self, inputs):
        """
        Forward pass for the VAE.

        Args:
            inputs (tf.Tensor): The input data

        Returns:
            tf.Tensor: Output tensor.
        """
        outputs = self.encoder(inputs)
        outputs = self.latent_ops(outputs)
        return self.decoder(outputs)

    def latent_ops(self, z):
        """
        Forward pass for the latent operations.

        Args:
            z (tf.Tensor): Input tensor.

        Returns:
            tf.Tensor: Output tensor.
        """
        self.mu = self.mu_layers(z)
        self.logvar = self.logvar_layers(z)
        z = self.sample_z()
        return z

    def sample_z(self):
        """
        Sample z from the latent space.

        Returns:
            tf.Tensor: Sampled z from the latent space.
        """
        e = tf.random.normal(shape=tf.shape(self.mu))
        return self.mu + e * tf.sqrt(tf.exp(self.logvar))

    def compile(self, rec_loss, kld_loss, *args, **kwargs):
        """
        Compiles the model.

        Args:
            rec_loss (_type_): The reconstruction loss function.
            kld_loss (_type_): The Kullback-Leibler divergence loss function.
        """
        super().compile(*args, **kwargs)
        self.rec_loss = rec_loss
        self.kld_loss = kld_loss

    def train_step(self, data):
        """
        Training step for the VAE.

        Args:
            data (tf.Tensor): Input data.

        Returns:
            tf.Tensor: The output of the batch step.
        """
        return self.batch_step(data, training=True)

    def test_step(self, data):
        """
        Testing step for the VAE.

        Args:
            data (tf.Tensor): Input data.

        Returns:
            tf.Tensor: The output of the batch step.
        """
        return self.batch_step(data, training=False)

    def batch_step(self, data, training=True):
        """
        Performs a single batch step of the autoencoder model.

        Args:
            data (tf.Tensor): Input data.
            training (bool, optional): Whether the model is in training
            mode or not. Defaults to True.

        Returns:
            dict: A dictionary containing the output metrics and losses
            of the batch step.
        """
        x = data
        with tf.GradientTape() as tape:
            y_pred = self.call(x)
            assert (
                y_pred.shape == x.shape
            ), f"Output shape: {y_pred.shape}, Input shape: {x.shape}"
            kld_loss = self.kld_loss(self.mu, self.logvar)
            rec_loss = self.rec_loss(x, y_pred)
            total_loss = rec_loss + kld_loss
        if training:
            gradients = tape.gradient(total_loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.rec_tracker.update_state(rec_loss)
        self.kld_tracker.update_state(kld_loss)
        output = {}
        for metric in self.metrics:
            metric.update_state(x, y_pred)
            output = {metric.name: metric.result()}
        output = {
            **output,
            "rec": self.rec_tracker.result(),
            "kld": self.kld_tracker.result(),
        }
        return output
