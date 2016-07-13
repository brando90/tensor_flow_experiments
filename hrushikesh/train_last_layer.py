import tensorflow as tf

with tf.Session() as sess:
  # Restore variables from disk.
  saver.restore(sess, "./tmp_test_experiments/tmp_July_12_jtest/tmp_mdl_July_12_slurm_sj00/model.ckpt")
  print("Model restored.")
  # Do some work with the model
  
