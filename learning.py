import time
import sys

import theano
import theano.tensor as T
import lasagne
import numpy as np
import pandas as pd


def load_dataset():
    data = pd.read_csv('drivers_5000.csv')

    data.at[data['Accidents'] > 0, 'AccidentsBin'] = 1
    data.at[data['Accidents'] == 0, 'AccidentsBin'] = 0

    xx_data_frame = data[['Age', 'Experience', 'PreviousAccidents', 'RouteDistance', 'Distance', 'HomeLat', 'HomeLng',
                          'WorkLat', 'WorkLng']]
    y_data_frame = data['AccidentsBin']
    xx = xx_data_frame.as_matrix().reshape(5000, 1, 9)
    y = y_data_frame.as_matrix().astype(np.int32)
    xx_train, xx_cv, xx_test = xx[:3000], xx[3000:4000], xx[4000:]
    y_train, y_cv, y_test = y[:3000], y[3000:4000], y[4000:]
    return xx_train, xx_cv, xx_test, y_train, y_cv, y_test


def build_mlp(input_var=None):
    l_in = lasagne.layers.InputLayer(shape=(None, 1, 9),
                                     input_var=input_var)
    l_in_drop = lasagne.layers.DropoutLayer(l_in, p=0.2)

    l_hid1 = lasagne.layers.DenseLayer(
            l_in_drop, num_units=100,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())
    l_hid1_drop = lasagne.layers.DropoutLayer(l_hid1, p=0.5)

    l_out = lasagne.layers.DenseLayer(
            l_hid1_drop, num_units=2,
            nonlinearity=lasagne.nonlinearities.softmax)

    return l_out


def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]


def main(num_epochs=500):
    x_train, x_cv, x_test, y_train, y_cv, y_test = load_dataset()
    input_var = T.tensor3('inputs')
    target_var = T.ivector('targets')
    network = build_mlp(input_var)

    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean()
    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.nesterov_momentum(
            loss, params, learning_rate=0.01, momentum=0.9)
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                            target_var)
    test_loss = test_loss.mean()
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                      dtype=theano.config.floatX)

    train_fn = theano.function([input_var, target_var], loss, updates=updates)
    cv_fn = theano.function([input_var, target_var], [test_loss, test_acc])

    for epoch in range(num_epochs):
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(x_train, y_train, 500, shuffle=True):
            inputs, targets = batch
            train_err += train_fn(inputs, targets)
            train_batches += 1

        val_err = 0
        val_acc = 0
        val_batches = 0
        for batch in iterate_minibatches(x_cv, y_cv, 500, shuffle=False):
            inputs, targets = batch
            err, acc = cv_fn(inputs, targets)
            val_err += err
            val_acc += acc
            val_batches += 1

        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
        print("  validation accuracy:\t\t{:.2f} %".format(
            val_acc / val_batches * 100))

    test_err = 0
    test_acc = 0
    test_batches = 0
    for batch in iterate_minibatches(x_test, y_test, 500, shuffle=False):
        inputs, targets = batch
        err, acc = cv_fn(inputs, targets)
        test_err += err
        test_acc += acc
        test_batches += 1
    print("Final results:")
    print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
    print("  test accuracy:\t\t{:.2f} %".format(
        test_acc / test_batches * 100))


if __name__ == '__main__':
    kwargs = {}
    if len(sys.argv) > 1:
        kwargs['num_epochs'] = int(sys.argv[1])
        main(**kwargs)
    else:
        main(500)
