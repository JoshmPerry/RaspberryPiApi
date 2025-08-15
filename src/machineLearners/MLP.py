import numpy as np
import tensorflow as tf

class MLP:
    def __init__(self):
        inputs = tf.keras.Input(shape=(60,))  # Adjust input shape as needed

        # First Dense layer
        x = tf.keras.layers.Dense(128, activation='relu')(inputs)
        x = tf.keras.layers.Dropout(0.2)(x)

        # Residual Block 1
        shortcut = x
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        x = tf.keras.layers.Dense(128)(x)

        x = tf.keras.layers.Add()([x, shortcut])
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.Dropout(0.2)(x)

        # Residual Block 2
        shortcut = x
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        x = tf.keras.layers.Dense(128)(x)

        x = tf.keras.layers.Add()([x, shortcut])
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.Dropout(0.2)(x)

        # Output layer
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