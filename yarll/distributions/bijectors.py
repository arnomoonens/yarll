"""
Tensorflow Bijecters for conditional shift and scale

Source: https://github.com/rail-berkeley/softlearning
License:
MIT License

Copyright (c) 2018 Softlearning authors and contributors

Softlearning uses a shared copyright model: each contributor holds copyright over
their contributions to Softlearning. The project versioning records all such
contribution and copyright details.

By contributing to the Softlearning repository through pull-request, comment,
or otherwise, the contributor releases their content to the license and
copyright terms herein.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import tensorflow as tf
from tensorflow_probability import bijectors as tfb
from tensorflow_probability.python.internal import assert_util, dtype_util

class ConditionalShift(tfb.Bijector):
    """Compute `Y = g(X; shift) = X + shift`.

    where `shift` is a numeric `Tensor`.

    Example Use:

    ```python
    shift = Shift([-1., 0., 1])
    x = [1., 2, 3]
    # `forward` is equivalent to:
    # y = x + shift
    y = shift.forward(x)  # [0., 2., 4.]
    ```

    """

    def __init__(self,
                 dtype=tf.float32,
                 validate_args=False,
                 name='conditional_shift'):
        """Instantiates the `ConditionalShift` bijector.

        Args:
          validate_args: Python `bool` indicating whether arguments should be
            checked for correctness.
          name: Python `str` name given to ops managed by this object.
        """
        parameters = dict(locals())
        with tf.name_scope(name) as name:
            super().__init__(
                forward_min_event_ndims=0,
                is_constant_jacobian=True,
                dtype=dtype,
                validate_args=validate_args,
                parameters=parameters,
                name=name)

    @classmethod
    def _is_increasing(cls):
        return True

    def _forward(self, x, shift):
        return x + shift

    def _inverse(self, y, shift):
        return y - shift

    def _forward_log_det_jacobian(self, x, shift):
        # is_constant_jacobian = True for this bijector, hence the
        # `log_det_jacobian` need only be specified for a single input, as this will
        # be tiled to match `event_ndims`.
        return tf.zeros((), dtype=dtype_util.base_dtype(x.dtype))


class ConditionalScale(tfb.Bijector):
    def __init__(self,
                 dtype=tf.float32,
                 validate_args=False,
                 name='conditional_scale'):
        """Instantiates the `ConditionalScale` bijector.

        This `Bijector`'s forward operation is:

        ```none
        Y = g(X) = scale * X
        ```

        Args:
          validate_args: Python `bool` indicating whether arguments should be
            checked for correctness.
          name: Python `str` name given to ops managed by this object.
        """
        parameters = dict(locals())
        with tf.name_scope(name) as name:
            super().__init__(
                forward_min_event_ndims=0,
                is_constant_jacobian=True,
                validate_args=validate_args,
                dtype=dtype,
                parameters=parameters,
                name=name)

    def _maybe_assert_valid_scale(self, scale):
        if not self.validate_args:
            return ()
        is_non_zero = assert_util.assert_none_equal(
            scale,
            tf.zeros((), dtype=scale.dtype),
            message='Argument `scale` must be non-zero.')
        return (is_non_zero, )

    def _forward(self, x, scale):
        with tf.control_dependencies(self._maybe_assert_valid_scale(scale)):
            return x * scale

    def _inverse(self, y, scale):
        with tf.control_dependencies(self._maybe_assert_valid_scale(scale)):
            return y / scale

    def _forward_log_det_jacobian(self, x, scale):
        with tf.control_dependencies(self._maybe_assert_valid_scale(scale)):
            return tf.math.log(tf.abs(scale))
