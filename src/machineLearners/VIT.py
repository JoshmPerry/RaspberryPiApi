import numpy as np
import tensorflow as tf

class TransformerModel:
    def __init__(self, num_layers=4):
        input_shape = (16, 16, 1)
        patch_size = 4
        num_patches = (input_shape[0] // patch_size) ** 2
        embedding_dim = 64
        num_heads = 4
        ff_dim = 128
        num_classes = 10

        inputs = tf.keras.Input(shape=input_shape)

        # Patch extraction
        patches = tf.image.extract_patches(
            images=tf.expand_dims(inputs, axis=0),
            sizes=[1, patch_size, patch_size, 1],
            strides=[1, patch_size, patch_size, 1],
            rates=[1, 1, 1, 1],
            padding='VALID'
        )[0]
        patches = tf.reshape(patches, (-1, patch_size * patch_size))
        x = tf.keras.layers.Dense(embedding_dim)(patches)

        # Positional embedding
        pos_embed = tf.keras.layers.Embedding(input_dim=num_patches, output_dim=embedding_dim)
        positions = tf.range(start=0, limit=num_patches, delta=1)
        x = x + pos_embed(positions)

        # Transformer encoder stack
        for _ in range(num_layers):
            attn_output = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embedding_dim)(x, x)
            x = tf.keras.layers.LayerNormalization()(x + attn_output)
            ff_output = tf.keras.layers.Dense(ff_dim, activation='relu')(x)
            ff_output = tf.keras.layers.Dense(embedding_dim)(ff_output)
            x = tf.keras.layers.LayerNormalization()(x + ff_output)

        # Classification head
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

        self.model = tf.keras.Model(inputs=inputs, outputs=outputs)
        self.confidenceModel = None

    def train(self, trainDataset, validationDataset):
        self.model.compile(optimizer='adam',
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])
        self.model.fit(trainDataset, validation_data=validationDataset, epochs=10)

        self.confidenceModel = tf.keras.Sequential([
            self.model,
            tf.keras.layers.Softmax()
        ])

    def save_model(self, path):
        if self.confidenceModel is None:
            raise ValueError("Model has not been trained yet.")
        self.confidenceModel.save(path)

    def load_model(self, path):
        self.confidenceModel = tf.keras.models.load_model(path)

    def predict(self, data):
        if self.confidenceModel is None:
            raise ValueError("Model has not been trained yet.")
        answerConfidences = self.confidenceModel.predict(data)[0]
        answerVal = np.argmax(answerConfidences)
        return answerVal, answerConfidences[answerVal]