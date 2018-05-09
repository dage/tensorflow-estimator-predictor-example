# An example of using a tensorflow custom core estimator with contrib predictor for increased inference performance.
# Attempts to use up-to-date best practice for tensorflow development and keep dependencies to a minimum.
# Performs a regression using a deep neural network where the number of inputs and outputs can easily be tweaked by changing a couple of constants.
# Initial version written by Dag Erlandsen, Spinning Owl AS in may 2018.
# MIT license

import tensorflow as tf
import numpy as np
import datetime
import time

FEATURES_RANK = 3	# The number of inputs
LABELS_RANK = 2		# The number of outputs

# Returns a numpy array of rank LABELS_RANK based on the features argument.
# Can be used when creating a training dataset.
def features_to_labels(features):
	sum_column = features.sum(1).reshape(features.shape[0], 1)
	labels = np.hstack((sum_column*i for i in range(1, LABELS_RANK+1)))
	return labels

def serving_input_fn():
    x = tf.placeholder(dtype=tf.float32, shape=[None, FEATURES_RANK], name='x')		# match dtype in input_fn
    inputs = {'x': x }
    return tf.estimator.export.ServingInputReceiver(inputs, inputs)

def model_fn(features, labels, mode):
	net = features["x"]			# input
	for units in [4, 8, 4]:		# hidden units
		net = tf.layers.dense(net, units=units, activation=tf.nn.relu)
		net = tf.layers.dropout(net, rate=0.1)
	output = tf.layers.dense(net, LABELS_RANK, activation=None)
	
	if mode == tf.estimator.ModeKeys.PREDICT:
		return tf.estimator.EstimatorSpec(mode, predictions=output, export_outputs={"out": tf.estimator.export.PredictOutput(output)})
	
	loss = tf.losses.mean_squared_error(labels, output)

	if mode == tf.estimator.ModeKeys.EVAL:
		return tf.estimator.EstimatorSpec(mode, loss=loss)
	
	optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)
	train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
	return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

# expecting a numpy array of shape (1, FEATURE_RANK) for constant_feature argument
def input_fn(num_samples, constant_feature = None, is_infinite = True):
	feature_values = np.full((num_samples, FEATURES_RANK), constant_feature) if isinstance(constant_feature, np.ndarray) else np.random.rand(num_samples, FEATURES_RANK)
	feature_values = np.float32(feature_values)	# match dtype in serving_input_fn
	labels = features_to_labels(feature_values)
	dataset = tf.data.Dataset.from_tensors(({"x": feature_values}, labels))
	if is_infinite:
		dataset = dataset.repeat()
	return dataset.make_one_shot_iterator().get_next()

estimator = tf.estimator.Estimator(
    model_fn=model_fn,	
    model_dir="model_dir\\estimator-predictor-test-{date:%Y-%m-%d %H.%M.%S}".format(date=datetime.datetime.now()))

train = estimator.train(input_fn=lambda : input_fn(50), steps=500)
evaluate = estimator.evaluate(input_fn=lambda : input_fn(20), steps=1)

predictor = tf.contrib.predictor.from_estimator(estimator, serving_input_fn)

consistency_check_features = np.random.rand(1, FEATURES_RANK)
consistency_check_labels = features_to_labels(consistency_check_features)

num_calls_predictor = 100
predictor_input = {"x": consistency_check_features}
start_time_predictor = time.clock()
for i in range(num_calls_predictor):
	predictor_prediction = predictor(predictor_input)
delta_time_predictor = 1./num_calls_predictor*(time.clock() - start_time_predictor)

num_calls_estimator_predict = 10
estimator_input = lambda : input_fn(1, consistency_check_features, False)
start_time_estimator_predict = time.clock()
for i in range(num_calls_estimator_predict):
	estimator_prediction = list(estimator.predict(input_fn=estimator_input))
delta_time_estimator = 1./num_calls_estimator_predict*(time.clock() - start_time_estimator_predict)

print("{} --> {}\n  predictor={}\n  estimator={}.\n".format(consistency_check_features, consistency_check_labels, predictor_prediction, estimator_prediction))
print("Time used per estimator.predict() call: {:.5f}s, predictor(): {:.5f}s ==> predictor is {:.0f}x faster!".format(delta_time_estimator, delta_time_predictor, delta_time_estimator/delta_time_predictor))