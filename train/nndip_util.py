import tensorflow as tf
import numpy as np
import math

FLAGS = tf.app.flags.FLAGS

def _FSpecialGauss(size, sigma):
    """Function to mimic the 'fspecial' gaussian MATLAB function."""
    radius = size // 2
    offset = 0.0
    start, stop = -radius, radius + 1
    if size % 2 == 0:
      offset = 0.5
      stop -= 1
    x, y = np.mgrid[offset + start:stop, offset + start:stop]
    assert len(x) == size
    g = np.exp(-((x**2 + y**2)/(2.0 * sigma**2)))
    g = np.reshape(g, [size, size, 1, 1])
    return g / g.sum()

def blur(sess, image, size, sigma):
    kernel = _FSpecialGauss(size, sigma)
    r = tf.nn.conv2d(tf.expand_dims(image[:, :, :, 0], axis=3), kernel, strides=[1, 1, 1, 1], padding='SAME')
    g = tf.nn.conv2d(tf.expand_dims(image[:, :, :, 1], axis=3), kernel, strides=[1, 1, 1, 1], padding='SAME')
    b = tf.nn.conv2d(tf.expand_dims(image[:, :, :, 2], axis=3), kernel, strides=[1, 1, 1, 1], padding='SAME')
    convolved = tf.concat(3, [r, g, b])
    return convolved

def psnr(img, ref, crop):
    #assume RGB image
    img_cropped = img[:, crop:-crop, crop:-crop]

    ref_cropped = ref[:, crop:-crop, crop:-crop]
	
    diff = img_cropped - ref_cropped
    rmse = tf.sqrt( tf.reduce_mean(tf.square(diff)) )

    # Assume the maximum pixel value is 1.0
    return 20*tf.log(1.0/rmse)/math.log(10)
    
