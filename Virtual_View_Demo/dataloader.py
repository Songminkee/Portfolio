from __future__ import absolute_import, division, print_function
import tensorflow as tf
from util import *

def string_length_tf(t):
    return tf.cast(tf.py_func(len ,[t], [tf.int32]),tf.int64)

class MonodepthDataloader(object):
    def __init__(self, data_path, params,  mode):
        self.data_path = data_path
        self.params = params
        self.mode = mode

        self.param_path_batch =[]
        self.reference_image_batch = None
        self.left_image_batch = None
        self.right_image_batch = None

        if mode == 'demo':
            self.left_image_path = tf.placeholder(tf.string)
            self.right_image_path = tf.placeholder(tf.string)

            self.left_param_path = tf.reshape(tf.placeholder(tf.string),[1])
            self.right_param_path = tf.reshape(tf.placeholder(tf.string),[1])
            self.param_path_batch = tf.expand_dims(tf.concat([self.left_param_path,self.right_param_path],0),0)

            self.left_image_batch = self.read_image(self.left_image_path)
            self.right_image_batch = self.read_image(self.right_image_path)

            self.left_image_batch.set_shape([1,params.synthesis_height,params.synthesis_width,3])
            self.right_image_batch.set_shape([1, params.synthesis_height, params.synthesis_width, 3])
            self.param_path_batch.set_shape([1, 2])

    def read_image(self, image_path):
        path_length = string_length_tf(image_path)[0]
        file_extension = tf.substr(image_path, path_length - 3, 3)
        file_cond = tf.equal(file_extension, 'jpg')
        image = tf.cond(file_cond, lambda: tf.image.decode_jpeg(tf.read_file(image_path)),
                        lambda: tf.image.decode_png(tf.read_file(image_path)))

        image = tf.image.convert_image_dtype(image, tf.float32)
        if self.mode =='synthesis':
            image = tf.image.resize_images(image, [self.params.synthesis_height, self.params.synthesis_width],
                                           tf.image.ResizeMethod.AREA)
            image = tf.reshape(image, [1, self.params.synthesis_height, self.params.synthesis_width, 3])
        elif self.mode =='demo':
            image = tf.image.resize_images(image, [self.params.synthesis_height, self.params.synthesis_width],
                                           tf.image.ResizeMethod.AREA)
            image = tf.reshape(image, [1, self.params.synthesis_height, self.params.synthesis_width, 3])

        else:
            image = tf.image.resize_images(image, [self.params.height, self.params.width], tf.image.ResizeMethod.AREA)
            image = tf.reshape(image, [1, self.params.height, self.params.width, 3])
        return image
