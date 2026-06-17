import tensorflow as tf
from .RNNCellsWithAttention import RNNCellsWithAttention
from .ArchitectureOptions import ArchitectureOptions
from .TokenMap import TokenMap
from .TrainingOptions import TrainingOptions
from .InferenceOptions import InferenceOptions


class KerasModel(tf.keras.Model):
    def __init__(
            self,
            architecture_options: ArchitectureOptions,
            token_map: TokenMap
    ) -> None:
        super().__init__()

        self.architecture_options = architecture_options
        """Define specific sizes, dimensions and layer counts"""

        self.token_map = token_map
        """Defines the mapping between the output layer and LMX tokens"""

        self.BOS = self.token_map.bos_token_index
        """BOS token index"""

        self.EOS = self.token_map.eos_token_index
        """EOS token index"""

        self.training_options: TrainingOptions | None = None
        """
        Options for training, must be set
        via prepare_for_training() before training
        """

        self.inference_options: InferenceOptions | None = None
        """
        Options for inference, must be set
        via prepare_for_inference() before inference
        """

        # === build encoder ===

        inputs = tf.keras.layers.Input(
            shape=[self.architecture_options.height, None, 1],
            dtype=tf.float32,
            ragged=True
        )

        hidden = inputs.to_tensor()
        hidden = tf.keras.layers.Conv2D(
            self.architecture_options.cnn_dim,
            3, 1, "same", use_bias=False
        )(hidden)
        for i in range(self.architecture_options.cnn_stages):
            filters = min(
                self.architecture_options.rnn_dim,
                self.architecture_options.cnn_dim * (1 << i)
            )
            residual = tf.keras.layers.Conv2D(filters, 3, 2, "same", use_bias=False)(hidden)
            residual = tf.keras.layers.BatchNormalization()(residual)
            hidden = tf.keras.layers.Conv2D(filters, 3, 2, "same", use_bias=False)(hidden)
            hidden = tf.keras.layers.BatchNormalization()(hidden)
            hidden = tf.keras.layers.ReLU()(hidden)
            hidden = tf.keras.layers.Conv2D(filters, 3, 1, "same", use_bias=False)(hidden)
            hidden = tf.keras.layers.BatchNormalization()(hidden)
            hidden += residual
            hidden = tf.keras.layers.ReLU()(hidden)
            for _ in range(self.architecture_options.cnn_resblocks - 1):
                residual = hidden
                hidden = tf.keras.layers.Conv2D(filters, 3, 1, "same", use_bias=False)(hidden)
                hidden = tf.keras.layers.BatchNormalization()(hidden)
                hidden = tf.keras.layers.ReLU()(hidden)
                hidden = tf.keras.layers.Conv2D(filters, 3, 1, "same", use_bias=False)(hidden)
                hidden = tf.keras.layers.BatchNormalization()(hidden)
                hidden += residual
                hidden = tf.keras.layers.ReLU()(hidden)

        hidden = tf.transpose(hidden, [0, 2, 1, 3])
        hidden = tf.reshape(hidden, [tf.shape(hidden)[0], tf.shape(hidden)[1], hidden.shape[2] * hidden.shape[3]])

        remaining = self.architecture_options.timestep_width \
            // (1 << self.architecture_options.cnn_stages)
        if remaining < 1:
            raise ValueError(
                "Inconsistent settings of timestep width {} and cnn stages {}" \
                    .format(
                        self.architecture_options.timestep_width,
                        self.architecture_options.cnn_stages
                    )
            )
        if remaining > 1:
            hidden = tf.pad(hidden, [[0, 0], [0, (-tf.shape(hidden)[1]) % remaining], [0, 0]])
            hidden = tf.reshape(
                hidden, [tf.shape(hidden)[0], tf.shape(hidden)[1] // remaining, hidden.shape[2] * remaining])

        hidden = tf.keras.layers.Dropout(self.architecture_options.dropout)(hidden)

        reduced_row_lengths = inputs.row_lengths(axis=2)[:, :1].to_tensor()[:, 0]
        reduced_row_lengths = (
            reduced_row_lengths + self.architecture_options.timestep_width - 1
        ) // self.architecture_options.timestep_width
        mask = tf.sequence_mask(reduced_row_lengths)
        for layer in range(self.architecture_options.rnn_layers):
            residual = hidden
            rnn_layer = tf.keras.layers.LSTM(self.architecture_options.rnn_dim, return_sequences=True)
            hidden = tf.keras.layers.Bidirectional(rnn_layer, merge_mode="sum")(hidden, mask=mask)
            hidden = tf.keras.layers.Dropout(self.architecture_options.dropout)(hidden)
            if layer: hidden += residual

        self.encoder = tf.keras.Model(inputs=inputs, outputs=hidden)

        # === build decoder ===

        self._target_embedding = tf.keras.layers.Embedding(
            len(self.token_map),
            self.architecture_options.rnn_dim
        )
        self._target_rnn = tf.keras.layers.RNN(
            RNNCellsWithAttention(
                [
                    tf.keras.layers.LSTMCell(self.architecture_options.rnn_dim)
                    for _ in range(self.architecture_options.rnn_layers_decoder)
                ],
                self.architecture_options.rnn_dim
            ),
            return_sequences=True
        )
        self._target_output_layer = tf.keras.layers.Dense(
            len(self.token_map)
        )
    
    def prepare_for_training(
            self,
            training_options: TrainingOptions,
            training_batch_count: int
    ):
        """Sets the model up for training"""
        self.training_options = training_options

        lr = 1e-3
        if training_options.lr_decay == "cos":
            lr = tf.optimizers.schedules.CosineDecay(
                lr,
                training_batch_count * training_options.epochs
            )
        elif training_options.lr_decay != "none":
            raise ValueError("Unknown decay '{}'".format(training_options.lr_decay))

        self.compile(
            optimizer=tf.optimizers.Adam(lr, jit_compile=False),
            loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True)
        )

    def prepare_for_inference(self, inference_options: InferenceOptions):
        """Sets the model up for inference"""
        self.inference_options = inference_options

    def decoder_training(
            self,
            encoded: tf.Tensor,
            targets: tf.Tensor
    ) -> tf.Tensor:
        """
        This method performs teacher-forcing training of the decoder.
        It is a private method, called only from the train_step() method.
        """
        self._target_rnn.cell.setup_memory(encoded)

        decoder_input = tf.concat([tf.fill([tf.shape(targets)[0], 1], self.BOS), targets[:, :-1]], axis=-1)
        hidden = self._target_embedding(decoder_input)
        hidden = self._target_rnn(hidden)
        hidden = self._target_output_layer(hidden)
        return hidden

    @tf.function
    def decoder_inference(
        self,
        encoded: tf.Tensor,
        max_length: tf.Tensor
    ) -> tf.Tensor:
        """
        This method performs autoregressive inference of the decoder.
        It is private method, called from the predict_step() method,
        but it is also dummy-called before weights are loaded to fully
        materialize the decoder weigths.
        """
        self._target_rnn.cell.setup_memory(encoded)

        batch_size = tf.shape(encoded)[0]
        index = tf.zeros([], tf.int32)
        inputs = tf.fill([batch_size], self.BOS)
        states = self._target_rnn.cell.get_initial_state(batch_size=batch_size, dtype=tf.float32)
        results = tf.TensorArray(tf.int32, size=max_length)
        result_lengths = tf.fill([batch_size], max_length)
        while tf.math.logical_and(index < max_length, tf.math.reduce_any(result_lengths == max_length)):
            hidden = self._target_embedding(inputs)
            hidden, states = self._target_rnn.cell(hidden, states)
            hidden = self._target_output_layer(hidden)
            predictions = tf.argmax(hidden, axis=-1, output_type=tf.int32)
            results = results.write(index, predictions)
            result_lengths = tf.where((predictions == self.EOS) & (result_lengths > index), index, result_lengths)
            inputs = predictions
            index += 1
        results = tf.RaggedTensor.from_tensor(tf.transpose(results.stack()), lengths=result_lengths)
        return results

    def train_step(self, data):
        """
        Called by TF Keras on training (from the model.train() method)
        https://www.tensorflow.org/versions/r2.12/api_docs/python/tf/keras/Model#train_step
        """
        assert self.training_options is not None, \
            "You must call prepare_for_training() before doing training"

        x, y = data
        y = tf.concat([y, tf.fill([tf.shape(y)[0], 1], self.EOS)], axis=-1)[:, :self.training_options.max_training_length]
        with tf.GradientTape() as tape:
            encoded = self.encoder(x, training=True)
            y_pred = self.decoder_training(encoded, y)
            loss = self.compiled_loss(y.values, y_pred.values)
        self.optimizer.minimize(loss, self.trainable_variables, tape=tape)
        return {metric.name: metric.result() for metric in self.compiled_loss.metrics}

    def predict_step(self, data):
        """
        Called by TF Keras on inference (from the model.predict() method)
        https://www.tensorflow.org/versions/r2.12/api_docs/python/tf/keras/Model#predict_step
        """
        assert self.inference_options is not None, \
            "You must call prepare_for_inference() before doing inference"
        
        if isinstance(data, tuple):
            data = data[0]
        encoded = self.encoder(data, training=False)
        y_pred = self.decoder_inference(
            encoded,
            self.inference_options.max_prediction_length
        )
        return y_pred
