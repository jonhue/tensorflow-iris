import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Import data
data = pd.read_csv('iris.data', sep=",", header=None)
## Dimensions of datasheet
n_examples = data.shape[0]
n_features = data.shape[1]
## Make data a np.array
data = data.values
print(data)
## Convert strings to numbers
for example in data:
    if example[4] == 'Iris-setosa':
        example[4] = -1
    elif example[4] == 'Iris-versicolor':
        example[4] = 0
    elif example[4] == 'Iris-virginica':
        example[4] = 1
print(data)
## Training & test data
train_start = 0
train_end = int(np.floor(0.8*n_examples))
test_start = train_end + 1
test_end = n_examples
data_train = data[np.arange(train_start, train_end), :]
data_test = data[np.arange(test_start, test_end), :]
## Scale data
scaler = MinMaxScaler(feature_range=(-1, 1))
scaler.fit(data_train)
data_train = scaler.transform(data_train)
data_test = scaler.transform(data_test)
print(data_train)
## Build X & Y
x_train = data_train[:, :4]
y_train = data_train[:, 4:]
x_test = data_test[:, :4]
y_test = data_test[:, 4:]

n_inputs = 4
n_hidden_1 = 8
n_hidden_2 = 4
n_hidden_3 = 2
n_output = 1

X = tf.placeholder(dtype=tf.float32, shape=[None, n_inputs])
Y = tf.placeholder(dtype=tf.float32, shape=[None, n_output])

# Initializers
sigma = 1
weight_initializer = tf.variance_scaling_initializer(mode="fan_avg", distribution="uniform", scale=sigma)
bias_initializer = tf.zeros_initializer()

## Layer 1
weight_hidden_1 = tf.Variable(weight_initializer([n_inputs, n_hidden_1]))
bias_hidden_1 = tf.Variable(bias_initializer([n_hidden_1]))
## Layer 2
weight_hidden_2 = tf.Variable(weight_initializer([n_hidden_1, n_hidden_2]))
bias_hidden_2 = tf.Variable(bias_initializer([n_hidden_2]))
## Layer 3
weight_hidden_3 = tf.Variable(weight_initializer([n_hidden_2, n_hidden_3]))
bias_hidden_3 = tf.Variable(bias_initializer([n_hidden_3]))
## Output
weight_output = tf.Variable(weight_initializer([n_hidden_3, n_output]))
bias_output = tf.Variable(bias_initializer([n_output]))

## Layer 1
hidden_1 = tf.nn.relu(tf.add(tf.matmul(X, weight_hidden_1), bias_hidden_1))
## Layer 2
hidden_2 = tf.nn.relu(tf.add(tf.matmul(hidden_1, weight_hidden_2), bias_hidden_2))
## Layer 3
hidden_3 = tf.nn.relu(tf.add(tf.matmul(hidden_2, weight_hidden_3), bias_hidden_3))
## Output
output = tf.nn.relu(tf.add(tf.matmul(hidden_3, weight_output), bias_output))

# Cost function
mse = tf.reduce_mean(tf.squared_difference(output, Y))

# Optimizer
optimizer = tf.train.AdamOptimizer().minimize(mse)


# Training

## Initialize session
graph = tf.Session()
graph.run(tf.global_variables_initializer())

## Epochs & batch size
epochs = 10
batch_size = 150

for e in range(epochs):

    ## Shuffle training data
    shuffle_indices = np.random.permutation(np.arange(len(y_train)))
    x_train = x_train[shuffle_indices]
    y_train = y_train[shuffle_indices]

    ## Minibatch training
    for i in range(0, len(y_train) // batch_size):
        start = i * batch_size
        batch_x = x_train[start:start + batch_size]
        batch_y = y_train[start:start + batch_size]
        ## Run optimizer with batch
        graph.run(optimizer, feed_dict={X: batch_x, Y: batch_y})

# Print final MSE after training
mse_final = graph.run(mse, feed_dict={X: x_test, Y: y_test})
print(mse_final)

print(graph.run(mse, feed_dict={X: [[5.1,3.5,1.4,0.2]], Y: [[0]]}))
print(graph.run(mse, feed_dict={X: [[5.9,3.2,4.8,1.8]], Y: [[0]]}))
