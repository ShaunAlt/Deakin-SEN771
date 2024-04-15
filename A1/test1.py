# =============================================================================
# Created by: Shaun Altmann
# =============================================================================
'''
TensorFlow Implementation
-
Contains the implementation of the TensorFlow machine learning framework.
'''
# =============================================================================

# =============================================================================
# Imports
# =============================================================================

# used for copying models
from copy import (
    deepcopy,
)

# used for keras implementation
import keras

# used for showing images
from matplotlib import pyplot as plt

# used for numpy implementation
import numpy as np

# used for tensorflow implementation
import tensorflow as tf

# used for type hinting
from typing import (
    Callable,
)


# =============================================================================
# Testing Function
# =============================================================================
def test(func: Callable[[int], float]):
    ''' Testing Function. '''

    vals: list[tuple[int, float]] = []
    for i in range(1, 11):
        vals.append((i, func(epochs = i)))

    print('Final Results')
    for i in vals:
        print('Epochs: {}, Accuracy: {}'.format(i[0], i[1]))

    return 1


# =============================================================================
# MNIST Dataset Implementation
# =============================================================================
def mnist(epochs: list[int] = [1]) -> list[tuple[int, float]]:
    '''
    MNIST Dataset Implementation
    -
    Tests a machine learning framework against the MNIST dataset.

    Parameters
    -
    - epochs : `list[int]`
        - List of the number of epochs to test.
    
    Returns
    -
    - `list[tuple[int, float]]`
        - List of the epoch number and accuracy of the model after being
            trained.
    '''

    print('MNIST Dataset Implementation')

    # load dataset
    print('| - Loading MNIST Dataset...')
    _mnist = keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = _mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # build machine learning model
    print('| - Build Machine Learning Model')
    model = keras.models.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10)
    ])

    # define loss function
    print('| - Define Loss Function')
    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits = True)

    # train model and get accuracy results
    results: list[tuple[int, float]] = []
    print('| - Train Model')
    def train_mnist(epochs: int) -> float:
        ''' Train MNIST Model. '''
        print(f'\t| - Training with {epochs} Epochs')

        # copy model
        print('\t\t| - Copying Model')
        m = keras.models.clone_model(model)

        # compile model
        print('\t\t| - Compile Model')
        m.compile(
            optimizer = 'adam',
            loss = loss_fn,
            metrics = ['accuracy']
        )

        # train model
        print('\t\t| - Train Model')
        m.fit(x_train, y_train, epochs = epochs, verbose = 0)

        # evaluate model
        print('\t\t| - Evaluate Model')
        _, accuracy = m.evaluate(x_test, y_test, verbose = 0)
        print(f'\t\t| - Accuracy: {accuracy}')
        return accuracy
    
    for i in epochs:
        results.append((i, train_mnist(i)))

    print('| - Final Results')
    for r in results:
        print(f'\t| - Epochs: {r[0]}, Accuracy: {r[1]}')
    return results


# =============================================================================
# MNIST Fashion Dataset Implementation
# =============================================================================
def mnist_fashion(epochs: list[int] = [1]) -> list[tuple[int, float]]:
    '''
    MNIST Fashion Dataset Implementation
    -
    Tests a machine learning framework against the MNIST Fashion dataset.

    Parameters
    -
    - epochs : `list[int]`
        - List of the number of epochs to test.
    
    Returns
    -
    - `list[tuple[int, float]]`
        - List of the epoch number and accuracy of the model after being
            trained.
    '''

    print('MNIST Dataset Implementation')

    # load dataset
    print('| - Loading MNIST Dataset...')
    _mnist = keras.datasets.fashion_mnist
    (x_train, y_train), (x_test, y_test) = _mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    class_names = [
        'T-Shirt / Top',
        'Trouser',
        'Pullover',
        'Dress',
        'Coat',
        'Sandal',
        'Skirt',
        'Sneaker',
        'Bag',
        'Ankle Boot',
    ]

    # build machine learning model
    print('| - Build Machine Learning Model')
    model = keras.models.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(10)
    ])

    # define loss function
    print('| - Define Loss Function')
    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits = True)

    # train model and get accuracy results
    results: list[tuple[int, float]] = []
    print('| - Train Model')
    def train(epochs: int) -> tuple[float,  keras.models.Sequential]:
        ''' Train MNIST Fashion Model. '''
        print(f'\t| - Training with {epochs} Epochs')

        # copy model
        print('\t\t| - Copying Model')
        m = keras.models.clone_model(model)

        # compile model
        print('\t\t| - Compile Model')
        m.compile(
            optimizer = 'adam',
            loss = loss_fn,
            metrics = ['accuracy']
        )

        # train model
        print('\t\t| - Train Model')
        m.fit(x_train, y_train, epochs = epochs, verbose = 0)

        # evaluate model
        print('\t\t| - Evaluate Model')
        _, accuracy = m.evaluate(x_test, y_test, verbose = 0)
        print(f'\t\t| - Accuracy: {accuracy}')
        return (accuracy, m)
    
    m = None
    for i in epochs:
        accuracy, m = train(i)
        results.append((i, accuracy))

    # showing final results
    print('| - Final Results')
    for r in results:
        print(f'\t| - Epochs: {r[0]}, Accuracy: {r[1]}')

    # creating probability model + testing
    print('| - Testing Final Image Processor')
    print('\t| - Creating Probability Model')
    prob_model = keras.Sequential([m, keras.layers.Softmax()])
    print('\t| - Getting Predictions')
    predictions = prob_model.predict(x_test)

    def _plot_image(i, _pred_vals, actual_val, img):
        actual_val, img = actual_val[i], img[i]
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])

        plt.imshow(img, cmap=plt.cm.binary)

        _prev_txt = np.argmax(_pred_vals)
        color = {True: 'green', False: 'red'}[_prev_txt == actual_val]

        plt.xlabel(
            (
                f'{class_names[_prev_txt]} {100*np.max(_pred_vals):2.0f}% (' \
                + f'{class_names[actual_val]})'
            ),
            color = color
        )

    def _plot_value_array(i, _pred_vals, actual_val):
        actual_val = actual_val[i]
        plt.grid(False)
        plt.xticks(range(10))
        plt.yticks([])
        thisplot = plt.bar(range(10), _pred_vals, color="#777777")
        plt.ylim([0, 1])
        predicted_label = np.argmax(_pred_vals)

        thisplot[predicted_label].set_color('red')
        thisplot[actual_val].set_color('green')

    num_rows = 5
    num_cols = 3
    num_images = num_rows*num_cols
    plt.figure(figsize=(2*2*num_cols, 2*num_rows))
    for i in range(num_images):
        plt.subplot(num_rows, 2*num_cols, 2*i+1)
        _plot_image(i, predictions[i], y_test, x_test)
        plt.subplot(num_rows, 2*num_cols, 2*i+2)
        _plot_value_array(i, predictions[i], y_test)
    plt.tight_layout()
    plt.show()
            
    return results


# =============================================================================
# End of File
# =============================================================================
