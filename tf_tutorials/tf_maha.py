def tf_maha(x,y,VI):
    num_var = len(x)
    with tf.Session() as sess:
        xx = tf.placeholder(tf.types.float32,[num_var])
        yy = tf.placeholder(tf.types.float32,[num_var])
        vivi = tf.placeholder(tf.types.float32,[num_var, num_var])
        diff = tf.sub(xx,yy)
        dt = tf.reshape(tf.transpose(diff),[1,len(x)])
        ds = tf.reshape(diff, [len(x),1])
        M1 = tf.matmul(dt,vivi)
        M2 = tf.matmul(M1,ds)
        output = tf.sqrt(M2)
        ans = sess.run([output], feed_dict = {xx:x, yy; y, vivi: VI})
    return ans
