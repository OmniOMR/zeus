import tensorflow as tf


class RNNCellsWithAttention(tf.keras.layers.AbstractRNNCell):
    """A class adding Bahdanau attention to the given RNN cell."""
    def __init__(self, cells, attention_dim):
        super().__init__()
        self._cells = cells
        self._project_encoder_layer = tf.keras.layers.Dense(attention_dim)
        self._project_decoder_layer = tf.keras.layers.Dense(attention_dim)
        self._output_layer = tf.keras.layers.Dense(1)

    @property
    def state_size(self):
        return tuple(cell.state_size for cell in self._cells)

    def setup_memory(self, encoded):
        self._encoded = encoded
        self._encoded_projected = self._project_encoder_layer(encoded)

    def call(self, inputs, states):
        projected = self._encoded_projected + tf.expand_dims(self._project_decoder_layer(tf.concat(states[0], axis=1)), axis=1)
        weights = tf.nn.softmax(self._output_layer(tf.tanh(projected)), axis=1)
        attention = tf.reduce_sum(self._encoded * weights, axis=1)
        inputs, new_states = tf.concat([inputs, attention], axis=1), []
        for i, (cell, state) in enumerate(zip(self._cells, states)):
            outputs, new_state = cell(inputs, state)
            inputs = outputs if i == 0 else inputs + outputs
            new_states.append(new_state)
        return outputs, tuple(new_states)
