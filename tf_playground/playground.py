## run cmd to collect model: python playground.py --logdir=/tmp/playground_tmp
## show board on browser run cmd: tensorboard --logdir=/tmp/playground_tmp
## browser: http://localhost:6006/

import tensorflow as tf

# x variable
x = tf.Variable(10.0,name='x')
# b placeholder (simualtes the "data" part of the training)
b = tf.placeholder(tf.float32)
# make model (1/2)(x-b)^2
xx_b = 0.5*tf.pow(x-b,2)
y=xx_b

learning_rate = 1.0
# get optimizer
opt = tf.train.GradientDescentOptimizer(learning_rate)
# gradient variable list = [ (gradient,variable) ]
gv = opt.compute_gradients(y,[x])
# transformed gradient variable list = [ (T(gradient),variable) ]
decay = 0.9 # decay the gradient for the sake of the example
# apply transformed gradients
tgv = [ (decay*g, v) for (g,v) in gv] #list [(grad,var)]
apply_transform_op = opt.apply_gradients(tgv)

# track value of x
x_scalar_summary = tf.scalar_summary("x", x)
x_histogram_sumarry = tf.histogram_summary('x_his', x)
with tf.Session() as sess:
    merged = tf.merge_all_summaries()
    tensorboard_data_dump = '/tmp/playground_tmp'
    writer = tf.train.SummaryWriter(tensorboard_data_dump, sess.graph)

    sess.run(tf.initialize_all_variables())
    epochs = 120
    for i in range(epochs):
        b_val = 1.0 #fake data (in SGD it would be different on every epoch)

        # applies the gradients
        [summary_str_apply_transform,_] = sess.run([merged,apply_transform_op], feed_dict={b: b_val})
        writer.add_summary(summary_str_apply_transform, i)
