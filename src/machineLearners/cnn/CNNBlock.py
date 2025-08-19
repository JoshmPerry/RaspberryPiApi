import tensorflow as tf

@tf.keras.utils.register_keras_serializable()
class CNNBlock(tf.keras.layers.Layer):
    def __init__(self, filters, residual=True, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.residual = residual
        self.conv1 = tf.keras.layers.Conv2D(filters, (3, 3), padding='same', activation='relu')
        self.conv2 = tf.keras.layers.Conv2D(filters, (3, 3), padding='same')
        if self.residual:
            self.shortcut_conv = tf.keras.layers.Conv2D(filters, (1, 1), padding='same')
        self.add = tf.keras.layers.Add()
        self.activation = tf.keras.layers.Activation('relu')

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        if self.residual:
            shortcut = self.shortcut_conv(inputs)
            x = self.add([x, shortcut])
        x = self.activation(x)
        return x
