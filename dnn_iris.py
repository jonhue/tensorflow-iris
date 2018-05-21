from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

# Our application logic will be added here

BATCH_SIZE=100
TRAIN_STEPS=1000

def main(unused_argv):
  # Load training and eval data
  iris = tf.contrib.learn.datasets.load_dataset("iris")
  data = iris.data
  labels = iris.target
  train_data = data[:100]
  train_labels = labels[:100]
  eval_data = data[100:]
  eval_labels = labels[100:]
  # Feature columns
  feature_columns = [
    tf.feature_column.numeric_column(key="SepalLength"),
    tf.feature_column.numeric_column(key="SepalWidth"),
    tf.feature_column.numeric_column(key="PetalLength"),
    tf.feature_column.numeric_column(key="PetalWidth")
  ]
  # Build DNN with 2 hidden layers, 10 neurons each
  classifier = tf.estimator.DNNClassifier(feature_columns=feature_columns, hidden_units=[10, 10], n_classes=3)
  # Train model
  train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x=train_data,
    y=train_labels,
    batch_size=BATCH_SIZE,
    num_epochs=None,
    shuffle=True
  )
  classifier.train(input_fn=train_input_fn, steps=TRAIN_STEPS)
  # Evaluate model
  eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    x=eval_data,
    y=eval_labels,
    num_epochs=1,
    shuffle=False
  )
  eval_results = classifier.evaluate(input_fn=eval_input_fn)
  print(eval_results)

if __name__ == "__main__":
  tf.app.run()
