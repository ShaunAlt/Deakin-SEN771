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
# End of File
# =============================================================================
