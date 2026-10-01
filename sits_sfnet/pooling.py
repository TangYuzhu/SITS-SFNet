"""Serializable argmax pooling and unpooling for the supplied SegNet."""

import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="sits_sfnet")
class MaxPoolingWithArgmax2D(tf.keras.layers.Layer):
    def __init__(self, pool_size=(2, 2), strides=(2, 2), padding="same", **kwargs):
        super().__init__(**kwargs)
        self.pool_size = tuple(pool_size)
        self.strides = tuple(strides)
        self.padding = padding.lower()

    def call(self, inputs):
        return tf.nn.max_pool_with_argmax(
            inputs,
            ksize=[1, *self.pool_size, 1],
            strides=[1, *self.strides, 1],
            padding=self.padding.upper(),
            output_dtype=tf.int64,
            include_batch_in_index=False,
        )

    def get_config(self):
        return {
            **super().get_config(),
            "pool_size": self.pool_size,
            "strides": self.strides,
            "padding": self.padding,
        }


@tf.keras.utils.register_keras_serializable(package="sits_sfnet")
class MaxUnpooling2D(tf.keras.layers.Layer):
    def __init__(self, size=(2, 2), **kwargs):
        super().__init__(**kwargs)
        self.size = tuple(size)

    def call(self, inputs):
        values, indices = inputs
        shape = tf.shape(values, out_type=tf.int64)
        output_shape = shape * tf.constant([1, *self.size, 1], tf.int64)
        batch = tf.reshape(tf.range(shape[0], dtype=tf.int64), (-1, 1, 1, 1))
        offsets = batch * tf.reduce_prod(output_shape[1:])
        flat_indices = tf.reshape(tf.cast(indices, tf.int64) + offsets, (-1, 1))
        output = tf.scatter_nd(
            flat_indices, tf.reshape(values, (-1,)), [tf.reduce_prod(output_shape)]
        )
        output = tf.reshape(output, output_shape)
        output.set_shape(self.compute_output_shape([values.shape, indices.shape]))
        return output

    def compute_output_shape(self, input_shape):
        batch, height, width, channels = input_shape[0]
        return (
            batch,
            None if height is None else height * self.size[0],
            None if width is None else width * self.size[1],
            channels,
        )

    def get_config(self):
        return {**super().get_config(), "size": self.size}
