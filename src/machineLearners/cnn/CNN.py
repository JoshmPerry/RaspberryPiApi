import numpy as np
import tensorflow as tf
from machineLearners.cnn.CNNBlock import CNNBlock

class CNN:
    def __init__(self):
        inputs = tf.keras.Input(shape=(16, 16, 1))

        # First Conv Block
        x = tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)

        # Residual Block 1 (using CNNBlock)
        x = CNNBlock(64, residual=True)(x)
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)

        # Residual Block 2 (using CNNBlock)
        x = CNNBlock(128, residual=True)(x)

        # Classification Head
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        outputs = tf.keras.layers.Dense(10, activation='softmax')(x)

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