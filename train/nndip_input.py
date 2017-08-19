import tensorflow as tf

import nndip_util

FLAGS = tf.app.flags.FLAGS

def setup_inputs(sess, filenames, image_size=None, capacity_factor=3):

    if image_size is None:
        image_size = FLAGS.sample_size
    
    # Read each JPEG file
    reader = tf.WholeFileReader()
    filename_queue = tf.train.string_input_producer(filenames)
    key, value = reader.read(filename_queue)
    channels = 3
    image = tf.image.decode_jpeg(value, channels=channels, name="dataset_image")
    image.set_shape([None, None, channels])

    # Crop and other random augmentations
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_saturation(image, .95, 1.05)
    image = tf.image.random_brightness(image, .05)
    image = tf.image.random_contrast(image, .95, 1.05)

    wiggle = 8
    off_x, off_y = 25-wiggle, 60-wiggle
    crop_size = 128
    crop_size_plus = crop_size + 2*wiggle
    image = tf.image.crop_to_bounding_box(image, off_y, off_x, crop_size_plus, crop_size_plus)
    image = tf.random_crop(image, [crop_size, crop_size, 3])

    image = tf.reshape(image, [1, crop_size, crop_size, 3])
    image = tf.cast(image, tf.float32)/255.0

    if crop_size != image_size:
        image = tf.image.resize_area(image, [image_size, image_size])

    # Add blurring to image using gaussian convolution
    blur_size = FLAGS.blur_size
    blur_sigma = FLAGS.blur_sigma
    if blur_sigma > 0:
        image_blurred = nndip_util.blur(sess, image, blur_size, blur_sigma)
    else:
        image_blurred = image

    # Add downsampling
    image_lres = image_blurred
    K = FLAGS.downsampling
    if K > 1:
        image_lres = tf.image.resize_area(image, [image_size//K, image_size//K])

    # Add gaussian noise
    noise_level = FLAGS.noise
    image_noisy = image_lres
    if noise_level > 0:
        image_noisy = image_noisy + \
                           tf.random_normal(image_noisy.get_shape(), stddev=noise_level)

    feature = image_noisy
    label   = image

    feature = tf.reshape(feature, [image_size//K,   image_size//K, 3])
    label   = tf.reshape(label, [image_size, image_size, 3])

    # Using asynchronous queues
    features, labels = tf.train.batch([feature, label],
                                      batch_size=FLAGS.batch_size,
                                      num_threads=4,
                                      capacity = capacity_factor*FLAGS.batch_size,
                                      name='labels_and_features')

    tf.train.start_queue_runners(sess=sess)
      
    return features, labels
