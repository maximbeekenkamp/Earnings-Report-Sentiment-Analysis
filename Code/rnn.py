import tensorflow as tf


class MyLSTM(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        """
        Custom LSTM layer.

        Args:
            units (int): Dimensionality of the output space.
            (embedding_size)
        """
        self.units = units
        super(MyLSTM, self).__init__(**kwargs)

    def build(self, input_shape):
        """
        Builds the custom LSTM layer.

        Args:
            input_shape (TensorShape): Shape of the input tensor.
        """
        kernel_shape = tf.TensorShape((input_shape[-1], 4 * self.units))

        self.kernel = self.add_weight(
            name="kernel",
            shape=kernel_shape,
            dtype=tf.float32,
            initializer="glorot_uniform",
            trainable=True,
        )
        self.recurrent_kernel = self.add_weight(
            name="recurrent_kernel",
            shape=kernel_shape,
            dtype=tf.float32,
            initializer="orthogonal",
            trainable=True,
        )
        self.bias = self.add_weight(
            name="bias",
            shape=(4 * self.units,),
            dtype=tf.float32,
            initializer="zeros",
            trainable=True,
        )

        super().build(input_shape)

    def call(self, inputs, initial_state=None):
        """
        Computes the output of the layer.

        Args:
            inputs (tf.Tensor): Input tensor.
            initial_state (tuple, optional): Initial state fo the cell.
            Defaults to None.

        Returns:
            tuple: Tuple containing the output tensor, the final hidden state,
            and the final cell state.
        """
        if initial_state:
            ht, ct = tf.identity(initial_state[0]), tf.identity(initial_state[1])
        else:
            ht = tf.zeros(shape=(inputs.shape[0], self.units), dtype=tf.float32)
            ct = tf.zeros(shape=(inputs.shape[0], self.units), dtype=tf.float32)

        W, U, b, units = self.kernel, self.recurrent_kernel, self.bias, self.units
        W_i, W_f, W_c, W_o = (
            W[:, :units],
            W[:, units : (2 * units)],
            W[:, (2 * units) : (3 * units)],
            W[:, (3 * units) :],
        )
        U_i, U_f, U_c, U_o = (
            U[:, :units],
            U[:, units : (2 * units)],
            U[:, (2 * units) : (3 * units)],
            U[:, (3 * units) :],
        )
        b_i, b_f, b_c, b_o = (
            b[:units],
            b[units : (units * 2)],
            b[(units * 2) : (units * 3)],
            b[(units * 3) :],
        )

        outputs = []
        inputs_time_major = tf.transpose(inputs, perm=[1, 0, 2])

        for input_each_step in inputs_time_major:
            ft = tf.sigmoid(tf.matmul(input_each_step, W_f) + tf.matmul(ht, U_f) + b_f)
            it = tf.sigmoid(tf.matmul(input_each_step, W_i) + tf.matmul(ht, U_i) + b_i)
            ct_squiggle = tf.tanh(
                tf.matmul(input_each_step, W_c) + tf.matmul(ht, U_c) + b_c
            )
            ct = (ft * ct) + (it * ct_squiggle)
            ot = tf.sigmoid(tf.matmul(input_each_step, W_o) + tf.matmul(ht, U_o) + b_o)
            ht = ot * tf.tanh(ct)
            outputs += [ht]

        outputs = tf.stack(outputs, axis=0)
        outputs = tf.transpose(outputs, perm=[1, 0, 2])

        return outputs, ht, ct

    def compute_output_shape(self, input_shape):
        """
        Computes the output shape of the layer.

        Args:
            input_shape (TensorShape): Shape of the input tensor.

        Returns:
            TensorShape: Shape of the output tensor.
        """
        shape = tf.TensorShape(input_shape).as_list()
        shape[-1] = self.units
        return tf.TensorShape(shape)

    def get_config(self):
        """
        Gets the configuration of the layer.

        Returns:
            dict: Configuration dictionary of the layer.
        """
        base_config = super(MyLSTM, self).get_config()
        base_config["units"] = self.units
        return base_config


class MyGRU(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        """
        Custom GRU layer.

        Args:
            units (int): Dimensionality of the output space.
        """
        self.units = units
        super(MyGRU, self).__init__(**kwargs)

    def build(self, input_shape):
        """
        Builds the custom GRU layer.

        Args:
            input_shape (TensorShape): Shape of the input tensor.
        """
        kernel_shape = tf.TensorShape((input_shape[-1], 3 * self.units))

        self.kernel = self.add_weight(
            name="kernel",
            shape=kernel_shape,
            dtype=tf.float32,
            initializer="glorot_uniform",
            trainable=True,
        )

        self.recurrent_kernel = self.add_weight(
            name="recurrent_kernel",
            shape=kernel_shape,
            dtype=tf.float32,
            initializer="orthogonal",
            trainable=True,
        )

        self.bias = self.add_weight(
            name="bias",
            shape=kernel_shape,
            dtype=tf.float32,
            initializer="zeros",
            trainable=True,
        )

        super(MyGRU, self).build(input_shape)

    def call(self, inputs, initial_state=None):
        """
        Computes the output of the layer.

        Args:
            inputs (tf.Tensor): Input tensor.
            initial_state (tf.Tensor, optional): Initial state of the cell.
            Defaults to None.

        Returns:
            tuple: Tuple containing the output tensor and the final hidden state.
        """
        ## Hidden state
        if initial_state is None:
            ht = tf.zeros(shape=(inputs.shape[0], self.units), dtype=tf.float32)
        else:
            ht = tf.identity(initial_state)

        W, U, b, units = self.kernel, self.recurrent_kernel, self.bias, self.units
        W_z, W_r, W_h = (W[:, :units], W[:, units : (2 * units)], W[:, (2 * units) :])
        U_z, U_r, U_h = (U[:, :units], U[:, units : (2 * units)], U[:, (2 * units) :])
        b = tf.reduce_sum(b, axis=0)
        b_z, b_r, b_h = (b[:units], b[units : (units * 2)], b[(units * 2) :])

        outputs = []
        inputs_time_major = tf.transpose(inputs, perm=[1, 0, 2])

        for input_each_step in inputs_time_major:
            Z = tf.math.sigmoid(
                tf.matmul(input_each_step, W_z) + tf.matmul(ht, U_z) + b_z
            )
            R = tf.math.sigmoid(
                tf.matmul(input_each_step, W_r) + tf.matmul(ht, U_r) + b_r
            )
            H = tf.math.tanh(
                tf.matmul(input_each_step, W_h) + (R * (tf.matmul(ht, U_h))) + b_h
            )
            ht = (Z * ht) + ((1 - Z) * H)
            outputs += [ht]
        outputs = tf.stack(outputs)
        outputs = tf.transpose(outputs, perm=[1, 0, 2])

        return outputs, ht

    def compute_output_shape(self, input_shape):
        """
        Computes the output shape of the layer.

        Args:
            input_shape (TensorShape): Shape of the input tensor.

        Returns:
            TensorShape: Shape of the output tensor.
        """
        shape = tf.TensorShape(input_shape).as_list()
        shape[-1] = self.units
        return tf.TensorShape(shape)

    def get_config(self):
        """
        Gets the configuration of the layer.

        Returns:
            dict: Configuration dictionary of the layer.
        """
        base_config = super(MyGRU, self).get_config()
        base_config["units"] = self.units
        return base_config
