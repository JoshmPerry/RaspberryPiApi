import tensorflow as tf

@tf.keras.utils.register_keras_serializable()
class MLPBlock(tf.keras.layers.Layer):
    def __init__(self, units, residual=True, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.residual = residual
        self.dense1 = tf.keras.layers.Dense(units, activation='relu')
        self.dense2 = tf.keras.layers.Dense(units)
        self.add = tf.keras.layers.Add()
        self.activation = tf.keras.layers.Activation('relu')

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        if self.residual:
            shortcut = inputs
            # If input shape does not match, project with Dense
            if shortcut.shape[-1] != self.units:
                shortcut = tf.keras.layers.Dense(self.units)(shortcut)
            x = self.add([x, shortcut])
        x = self.activation(x)
        return x
