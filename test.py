import tensorflow as tf
import scipy.io
import numpy as np

# Use PCAHelper's load_data logic for USPS.mat
usps = scipy.io.loadmat('./data/PCA/USPS.mat')
# Use the same keys as in PCAHelper: 'A' for data, 'L' for labels
x_all = usps['A']  # shape: (features, samples)
y_all = [x[0] for x in usps['L']]  # shape: (samples,)

# Convert to numpy arrays
x_all = np.array(x_all)  # shape: (samples, features)
y_all = np.array(y_all)

# Scramble (shuffle) data order before splitting
idx = np.random.permutation(len(x_all))
x_all = x_all[idx]
y_all = y_all[idx]

# Split into train/test (e.g., 80/20 split)
split_idx = int(0.8 * len(x_all))
x_train, x_test = x_all[:split_idx], x_all[split_idx:]
y_train, y_test = y_all[:split_idx], y_all[split_idx:]
#print(f"x_train shape: {x_test.shape}, y_train shape: {y_test.shape}")

# Normalize images (assuming pixel values are 0-255)
x_train = x_train / 255.0
x_test = x_test / 255.0

model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
  loss='sparse_categorical_crossentropy',
  metrics=['accuracy'])

model.fit(x_train, y_train, epochs=50)
model.evaluate(x_test, y_test)

probability_model = tf.keras.Sequential([
  model,
  tf.keras.layers.Softmax()
])

ans = probability_model.predict(np.array([[0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.546 ,1.879 ,0.255 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.284 ,1.804 ,2.0 ,1.42 ,0.336 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.02200001 ,1.713 ,2.0 ,1.027 ,1.408 ,1.947 ,1.56 ,0.462 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.882 ,2.0 ,1.665 ,0.09799999 ,0.03100002 ,0.64 ,1.805 ,1.987 ,1.327 ,0.203 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.065 ,1.764 ,2.0 ,0.633 ,0.0 ,0.0 ,0.0 ,0.08600003 ,0.744 ,1.833 ,1.778 ,0.78 ,0.008000016 ,0.0 ,0.0 ,0.0 ,0.744 ,2.0 ,1.538 ,0.014 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.163 ,1.551 ,2.0 ,0.715 ,0.0 ,0.0 ,0.06400001 ,1.844 ,2.0 ,0.737 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.173 ,1.841 ,1.913 ,0.05199999 ,0.0 ,0.855 ,2.0 ,1.765 ,0.02499998 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,1.23 ,2.0 ,0.383 ,0.0 ,1.57 ,2.0 ,0.959 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,1.179 ,2.0 ,0.434 ,0.07800001 ,1.939 ,1.976 ,0.135 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,1.36 ,1.998 ,0.221 ,0.133 ,1.983 ,1.656 ,0.007000029 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.06400001 ,1.878 ,1.878 ,0.04299998 ,0.4 ,1.998 ,1.415 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.723 ,2.0 ,1.273 ,0.0 ,0.217 ,1.996 ,1.6 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.03399998 ,0.562 ,1.807 ,1.985 ,0.474 ,0.0 ,0.0 ,1.629 ,1.912 ,0.687 ,0.118 ,0.0 ,0.0 ,0.0 ,0.101 ,0.664 ,1.382 ,2.0 ,1.988 ,1.19 ,0.0 ,0.0 ,0.0 ,0.204 ,1.525 ,2.0 ,1.979 ,1.613 ,1.613 ,1.613 ,1.884 ,2.0 ,2.0 ,1.888 ,0.911 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.04400003 ,0.717 ,1.488 ,2.0 ,2.0 ,2.0 ,2.0 ,1.506 ,0.826 ,0.189 ,0.0 ,0.0 ,0.0 ,0.0]]))
ansVal = np.argmax(ans, axis=1)[0]
print(ansVal, ans[0][ansVal])  # Print the predicted class

#print(x_test[0], x_test[0].shape)