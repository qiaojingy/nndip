import numpy as np
import os.path
import scipy.misc
import tensorflow as tf
import time
import nndip_util

FLAGS = tf.app.flags.FLAGS

def _summarize_progress(train_data, feature, label, gene_output, batch, suffix, max_samples=8):
    td = train_data

    size = [label.shape[1], label.shape[2]]

    nearest = feature
    if FLAGS.downsampling > 1:
        nearest = tf.image.resize_nearest_neighbor(feature, size)
    nearest = tf.maximum(tf.minimum(nearest, 1.0), 0.0)

    clipped = tf.maximum(tf.minimum(gene_output, 1.0), 0.0)

    image   = tf.concat(2, [nearest, clipped, label])

    image = image[0:max_samples,:,:,:]
    image = tf.concat(0, [image[i,:,:,:] for i in range(max_samples)])
    image = td.sess.run(image)

    filename = 'batch%06d_%s.png' % (batch, suffix)
    filename = os.path.join(FLAGS.train_dir, filename)
    scipy.misc.toimage(image, cmin=0., cmax=1.).save(filename)
    print("    Saved %s" % (filename,))

    psnr_value = nndip_util.psnr(clipped, label, 3)
    print("    PSNR %f" % (td.sess.run(psnr_value),))

def _save_checkpoint(train_data, batch):
    td = train_data

    oldname = 'checkpoint_old.txt'
    newname = 'checkpoint_new.txt'

    oldname = os.path.join(FLAGS.checkpoint_dir, oldname)
    newname = os.path.join(FLAGS.checkpoint_dir, newname)

    # Delete oldest checkpoint
    try:
        tf.gfile.Remove(oldname)
        tf.gfile.Remove(oldname + '.meta')
    except:
        pass

    # Rename old checkpoint
    try:
        tf.gfile.Rename(newname, oldname)
        tf.gfile.Rename(newname + '.meta', oldname + '.meta')
    except:
        pass

    # Generate new checkpoint
    saver = tf.train.Saver()
    saver.save(td.sess, newname)

    print("    Checkpoint saved")

def train_model(train_data):
    print('Training...')
    td = train_data

    summaries = tf.merge_all_summaries()
    if FLAGS.checkpoint == 'nocheckpoint':
        td.sess.run(tf.initialize_all_variables())
    else:
        saver = tf.train.Saver()
        saver.restore(td.sess, FLAGS.checkpoint)
        print("Model restored.")

    lrval       = FLAGS.learning_rate_start
    start_time  = time.time()
    done  = False
    batch = 0

    assert FLAGS.learning_rate_half_life % 10 == 0

    # Cache dev and test features and labels (they are small)
    dev_feature, dev_label = td.sess.run([td.dev_features, td.dev_labels])
    test_feature, test_label = td.sess.run([td.test_features, td.test_labels])

    while not done:
        batch += 1
        gene_loss = disc_real_loss = disc_fake_loss = -1.234

        feed_dict = {td.learning_rate : lrval}

        ops = [td.gene_minimize, td.disc_minimize, td.gene_loss, td.disc_real_loss, td.disc_fake_loss]
        _, _, gene_loss, disc_real_loss, disc_fake_loss = td.sess.run(ops, feed_dict=feed_dict)
        
        if batch % 120 == 0:
            # Show we are alive
            elapsed = int(time.time() - start_time)/60
            print('Progress[%3d%%], ETA[%4dm], Batch [%4d], G_Loss[%3.3f], D_Real_Loss[%3.3f], D_Fake_Loss[%3.3f]' %
                  (int(100*elapsed/FLAGS.train_time), FLAGS.train_time - elapsed,
                   batch, gene_loss, disc_real_loss, disc_fake_loss))

            # Finished?            
            current_progress = elapsed / FLAGS.train_time
            if current_progress >= 1.0:
                done = True
            
            # Update learning rate
            if batch % FLAGS.learning_rate_half_life == 0:
                lrval *= .5

        if batch % FLAGS.summary_period == 0:
            # Show progress with test features
            feed_dict = {td.gene_minput: dev_feature}
            gene_output = td.sess.run(td.gene_moutput, feed_dict=feed_dict)
            _summarize_progress(td, dev_feature, dev_label, gene_output, batch, 'dev_out')
            feed_dict = {td.gene_minput: test_feature}
            gene_output = td.sess.run(td.gene_moutput, feed_dict=feed_dict)
            _summarize_progress(td, test_feature, test_label, gene_output, batch, 'test_out')
            
        if batch % FLAGS.checkpoint_period == 0:
            # Save checkpoint
            _save_checkpoint(td, batch)

    _save_checkpoint(td, batch)
    print('Finished training!')
