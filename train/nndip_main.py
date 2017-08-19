import nndip_input
import nndip_model
import nndip_train

import os.path
import random
import numpy as np
import numpy.random

import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

# Configuration (alphabetically)
tf.app.flags.DEFINE_integer('batch_size', 16,
                            "Number of samples per batch.")

tf.app.flags.DEFINE_float('blur_sigma', 0,
                          "Gaussian convolution window size")

tf.app.flags.DEFINE_integer('blur_size', 0,
                          "Gaussian convolution window size")

tf.app.flags.DEFINE_string('checkpoint_dir', 'checkpoint_dblur_scene',
                           "Output folder where checkpoints are dumped.")

tf.app.flags.DEFINE_integer('checkpoint_period', 2000,
                            "Number of batches in between checkpoints")

tf.app.flags.DEFINE_string('dataset', '../testSet_resize',
                           "Path to the dataset directory.")

tf.app.flags.DEFINE_string('checkpoint', 'nocheckpoint',
                           "Path to the checkpoint file.")

tf.app.flags.DEFINE_float('epsilon', 1e-8,
                          "Fuzz term to avoid numerical instability")

tf.app.flags.DEFINE_integer('downsampling', 2,
                            "Downsampling ratio (can be 1, 2, 4, 8)")

tf.app.flags.DEFINE_float('noise', 0.0,
                          "Guassian noise added to the image")

tf.app.flags.DEFINE_string('run', 'demo',
                            "Which operation to run. [demo|train]")

tf.app.flags.DEFINE_float('gene_l1_factor', .9,
                          "Multiplier for generator L1 loss term")

tf.app.flags.DEFINE_float('learning_beta1', 0.5,
                          "Beta1 parameter used for AdamOptimizer")

tf.app.flags.DEFINE_float('learning_rate_start', 0.00020,
                          "Starting learning rate used for AdamOptimizer")

tf.app.flags.DEFINE_integer('learning_rate_half_life', 5000,
                            "Number of batches until learning rate is halved")

tf.app.flags.DEFINE_bool('log_device_placement', False,
                         "Log the device where variables are placed.")

tf.app.flags.DEFINE_integer('sample_size', 64,
                            "Image sample size in pixels. Range [64,128]")

tf.app.flags.DEFINE_integer('summary_period', 240,
                            "Number of batches between summary data dumps")

tf.app.flags.DEFINE_integer('random_seed', 0,
                            "Seed used to initialize rng.")

tf.app.flags.DEFINE_integer('dev_vectors', 10,
                            """Number of features to use for devlopment""")

tf.app.flags.DEFINE_integer('test_vectors', 10,
                            """Number of features to use for testing""")
                            
tf.app.flags.DEFINE_string('train_dir', 'train',
                           "Output folder where training logs are dumped.")

tf.app.flags.DEFINE_integer('train_time', 180,
                            "Time in minutes to train the model")

def prepare_dirs(delete_train_dir=False):
    # Create checkpoint dir (do not delete anything)
    if not tf.gfile.Exists(FLAGS.checkpoint_dir):
        tf.gfile.MakeDirs(FLAGS.checkpoint_dir)
    
    # Cleanup train dir
    if delete_train_dir:
        if tf.gfile.Exists(FLAGS.train_dir):
            tf.gfile.DeleteRecursively(FLAGS.train_dir)
        tf.gfile.MakeDirs(FLAGS.train_dir)

    # Return names of training files
    if not tf.gfile.Exists(FLAGS.dataset) or \
       not tf.gfile.IsDirectory(FLAGS.dataset):
        raise FileNotFoundError("Could not find folder `%s'" % (FLAGS.dataset,))

    filenames = tf.gfile.ListDirectory(FLAGS.dataset)
    filenames = sorted(filenames)
    random.shuffle(filenames)
    filenames = [os.path.join(FLAGS.dataset, f) for f in filenames]

    return filenames


def setup_tensorflow():
    # Create session
    config = tf.ConfigProto(log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=config)

    # Initialize rng with a deterministic seed
    with sess.graph.as_default():
        tf.set_random_seed(FLAGS.random_seed)
        
    random.seed(FLAGS.random_seed)
    np.random.seed(FLAGS.random_seed)

    summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, sess.graph)

    return sess, summary_writer

class TrainData(object):
    def __init__(self, dictionary):
        self.__dict__.update(dictionary)

def _train():
    # Setup global tensorflow state
    sess, summary_writer = setup_tensorflow()

    # Prepare directories
    all_filenames = prepare_dirs(delete_train_dir=True)

    # Separate training and test sets
    train_filenames = all_filenames[:-(FLAGS.dev_vectors+FLAGS.test_vectors)]
    dev_filenames  = all_filenames[-(FLAGS.dev_vectors+FLAGS.test_vectors):-FLAGS.test_vectors]
    test_filenames  = all_filenames[-FLAGS.test_vectors:]
    print(test_filenames)

    # Setup async input queues
    train_features, train_labels = nndip_input.setup_inputs(sess, train_filenames)
    dev_features,  dev_labels  = nndip_input.setup_inputs(sess, dev_filenames)
    test_features,  test_labels  = nndip_input.setup_inputs(sess, test_filenames)

    # Add some noise during training (think denoising autoencoders)
    noise_level = 0.03
    noisy_train_features = train_features + \
                           tf.random_normal(train_features.get_shape(), stddev=noise_level)

    # Create and initialize model
    [gene_minput, gene_moutput,
     gene_output, gene_var_list,
     disc_real_output, disc_fake_output, disc_var_list] = \
            nndip_model.create_model(sess, noisy_train_features, train_labels)

    gene_loss = nndip_model.create_generator_loss(disc_fake_output, gene_output, train_features)
    disc_real_loss, disc_fake_loss = \
                     nndip_model.create_discriminator_loss(disc_real_output, disc_fake_output)
    disc_loss = tf.add(disc_real_loss, disc_fake_loss, name='disc_loss')
    
    (global_step, learning_rate, gene_minimize, disc_minimize) = \
            nndip_model.create_optimizers(gene_loss, gene_var_list,
                                         disc_loss, disc_var_list)

    # Train model
    train_data = TrainData(locals())
    nndip_train.train_model(train_data)

def main(argv=None):
    # Training or showing off?

    if FLAGS.run == 'demo':
        print("Please write demo code.")
    elif FLAGS.run == 'train':
        _train()

if __name__ == '__main__':
  tf.app.run()
