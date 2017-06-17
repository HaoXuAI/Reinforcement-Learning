"""Loss functions."""

import tensorflow as tf
import semver


def huber_loss(y_true, y_pred, max_grad=1.):
    """Calculate the huber loss.

    See https://en.wikipedia.org/wiki/Huber_loss

    Parameters
    ----------
    y_true: np.array, tf.Tensor
      Target value.
    y_pred: np.array, tf.Tensor
      Predicted value.
    max_grad: float, optional
      Positive floating point value. Represents the maximum possible
      gradient magnitude.

    Returns
    -------
    tf.Tensor
      The huber loss.
    """
    if tf.abs(y_true - y_pred) <= max_grad:
        return 1/2 * (y_true - y_pred) ** 2
    else:
        return max_grad * abs(y_true - y_pred) - 1/2 * max_grad ** 2


def mean_huber_loss(y_true, y_pred, max_grad=1.):
    """Return mean huber loss.

    Same as huber_loss, but takes the mean over all values in the
    output tensor.

    Parameters
    ----------
    y_true: np.array, tf.Tensor
      Target value.
    y_pred: np.array, tf.Tensor
      Predicted value.
    max_grad: float, optional
      Positive floating point value. Represents the maximum possible
      gradient magnitude.

    Returns
    -------
    tf.Tensor
      The mean huber loss.
    """
    alpha = tf.abs(y_true - y_pred)
    conditiion = tf.less(alpha, max_grad)
    minVal = 1/2 * alpha ** 2
    maxVal = max_grad * alpha - 1/2 * max_grad ** 2
    return tf.where(conditiion, minVal, maxVal)