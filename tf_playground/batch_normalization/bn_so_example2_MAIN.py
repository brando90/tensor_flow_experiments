##Example:

n_in, n_out = 3, 16
ksize = 3
stride = 1
phase_train = tf.placeholder(tf.bool, name='phase_train')
input_image = tf.placeholder(tf.float32, name='input_image')
kernel = tf.Variable(tf.truncated_normal([ksize, ksize, n_in, n_out],
                                   stddev=math.sqrt(2.0/(ksize*ksize*n_out))),
                                   name='kernel')
conv = tf.nn.conv2d(input_image, kernel, [1,stride,stride,1], padding='SAME')
conv_bn = batch_norm(conv, n_out, phase_train)
relu = tf.nn.relu(conv_bn)

with tf.Session() as session:
    session.run(tf.initialize_all_variables())
    for i in range(20):
        test_image = np.random.rand(4,32,32,3)
        sess_outputs = session.run([relu],
          {input_image.name: test_image, phase_train.name: True})
