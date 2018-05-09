# tensorflow-estimator-predictor-example
An example of using a custom tensorflow core estimator with predictor for increased inference performance when using the tensorflow estimator high-level API.

Performs a regression using a deep neural network where the number of inputs and outputs can easily be tweaked by changing a couple of constants. A check for prediction consistency between estimator.predict() and predictor() is performed, and a performance cost comparison is done.

For new developers, Tensorflow can have a pretty steep learning curve and given the rapid pace of development, examples found online can often be out-of-date. I want to give back to the community by sharing this code as a public repository and I hope that it will be useful for new developers. This example strives to use up-to-date best practice for tensorflow development and keep dependencies to a minimum.

Initial version written by Dag Erlandsen, Spinning Owl AS in may 2018.
MIT license
