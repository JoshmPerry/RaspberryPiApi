import numpy as np
import tensorflow as tf
from machineLearners.vit.VITBlock import VITBlock

class VIT:
    def __init__(self, num_layers=4):
        input_shape = (16, 16, 1)
        patch_size = 4
        num_patches = (input_shape[0] // patch_size) ** 2
        embedding_dim = 64
        num_heads = 4
        ff_dim = 128
        num_classes = 10

        inputs = tf.keras.Input(shape=input_shape)

        # Patch extraction using Lambda layer to keep it symbolic
        @tf.keras.utils.register_keras_serializable()
        def extract_patches(images):
            patches = tf.image.extract_patches(
                images=images,
                sizes=[1, patch_size, patch_size, 1],
                strides=[1, patch_size, patch_size, 1],
                rates=[1, 1, 1, 1],
                padding='VALID'
            )
            # Flatten the patches
            patch_dims = patch_size * patch_size * input_shape[2]
            patches = tf.reshape(patches, (tf.shape(patches)[0], -1, patch_dims))
            return patches

        patches = tf.keras.layers.Lambda(extract_patches, name="patch_extraction")(inputs)
        x = tf.keras.layers.Dense(embedding_dim)(patches)

        # Positional embedding
        pos_embed = tf.keras.layers.Embedding(input_dim=num_patches, output_dim=embedding_dim)
        positions = tf.range(start=0, limit=num_patches, delta=1)
        x = x + pos_embed(positions)

        # Transformer encoder stack using VITBlock
        for _ in range(num_layers):
            x = VITBlock(embed_dim=embedding_dim, num_heads=num_heads, mlp_dim=ff_dim, residual=True)(x)

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