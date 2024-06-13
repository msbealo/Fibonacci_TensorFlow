import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random

# Generate a Fibonacci-like series
def generate_fibonacci_series(n):
    series = [0, 1]
    for i in range(2, n):
        series.append(series[-1] + series[-2])
    return series

# Prepare the dataset
series = generate_fibonacci_series(50)

# Apply exponential scaling to the data
series = np.array(series, dtype=np.float32)
log_series = np.log(series + 1)  # Add 1 to avoid log(0)

X = []
y = []

# We will use a window size of 5 for the input
window_size = 5
for i in range(len(log_series) - window_size):
    X.append(log_series[i:i + window_size])
    y.append(log_series[i + window_size])

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.float32)

# Split the data into training and testing sets using random sampling
indices = list(range(len(X)))
random.shuffle(indices)
split_index = int(len(X) * 0.8)
train_indices = indices[:split_index]
test_indices = indices[split_index:]

X_train, X_test = X[train_indices], X[test_indices]
y_train, y_test = y[train_indices], y[test_indices]

# Build the model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(window_size,)),
    tf.keras.layers.Dense(50, activation='relu'),
    tf.keras.layers.Dense(50, activation='relu'),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse')

# Train the model and save the history
history = model.fit(X_train, y_train, epochs=1000, validation_data=(X_test, y_test), verbose=0)

# Plot training & validation loss values on a logarithmic scale
plt.semilogy(history.history['loss'])
plt.semilogy(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper right')
plt.show()

# Make 10 predictions in the series at random and compare to the correct prediction. Create an average error for all predictions
total_error = 0
number_tests = 10
for i in range(number_tests):
    index = random.randint(0, len(series) - window_size - 1)
    last_window = np.array(log_series[index:index + window_size]).reshape(1, -1)
    next_number_log = model.predict(last_window)
    next_number = np.exp(next_number_log[0][0]) - 1  # Apply exponential and subtract 1 to revert log scaling
    actual_next_number = series[index + window_size]
    error = abs(next_number - actual_next_number) / actual_next_number
    total_error += error
    print(f"The last window in the series is: {np.exp(last_window)}")
    print(f'The predicted next number in the series is: {next_number}')
    print(f"The actual next number in the series is: {actual_next_number}")
    print(f"The error is: {error}")

# Output the total error
print(f'The average error for all predictions is: {total_error / number_tests * 100}')

# using a semilog plot, plot the predicted and actual values of the series
predicted_series = []
for i in range(len(series) - window_size):
    last_window = np.array(log_series[i:i + window_size]).reshape(1, -1)
    next_number_log = model.predict(last_window)
    next_number = np.exp(next_number_log[0][0]) - 1
    predicted_series.append(next_number)

# This should plot actual on the x axis and predicted on the y axis using a log log scale
plt.loglog(series[window_size:], predicted_series)
plt.title('Actual vs Predicted Series')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.show()

