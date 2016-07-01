## run cmd to collect model: python quadratic_minimizer_tensorboard.py --logdir=/tmp/quaratic_temp
## show board on browser run cmd: tensorboard --logdir=/tmp/quaratic_temp
## browser: http://localhost:6006/

import tensorflow as tf

def get_quaratic():
    # x variable
    with tf.variable_scope('quadratic'):
        x = tf.get_variable('x', [1], initializer= tf.constant_initializer(10.0))
        #x = tf.Variable(10.0,name='x')
        # b placeholder (simualtes the "data" part of the training)
        b = tf.placeholder(tf.float32,name='b')
        # make model (1/2)(x-b)^2
        xx_b = 0.5*tf.pow(x-b,2)
        y=xx_b
        return y,x

y,x = get_quaratic()
learning_rate = 1.0
# get optimizer
opt = tf.train.GradientDescentOptimizer(learning_rate)
# gradient variable list = [ (gradient,variable) ]
print x.name
with tf.variable_scope('quadratic', reuse = True):
    x = tf.get_variable('x', [1])
#x = tf.get_variable('quadratic/x', [1])

#x = tf.get_variable('quadratic/x')
#x = tf.get_variable(x.name)

gv = opt.compute_gradients(y,[x])
# transformed gradient variable list = [ (T(gradient),variable) ]
decay = 0.9 # decay the gradient for the sake of the example
# apply transformed gradients
tgv = [ (decay*g, v) for (g,v) in gv] #list [(grad,var)]
apply_transform_op = opt.apply_gradients(tgv)

# get first grad
(dydx,_) = tgv[0]
# track value of x
x_scalar_summary = tf.scalar_summary("x", x)
x_histogram_sumarry = tf.histogram_summary('x_his', x)
# track value of dydx (gradient)
grad_scalar_summary = tf.scalar_summary("dydx", dydx)
grad_histogram_sumarry = tf.histogram_summary('dydx_his', dydx)
with tf.Session() as sess:
    merged = tf.merge_all_summaries()
    tensorboard_data_dump = '/tmp/quaratic_temp_tensorboard'
    writer = tf.train.SummaryWriter(tensorboard_data_dump, sess.graph)

    sess.run(tf.initialize_all_variables())
    epochs = 100
    for i in range(epochs):
        b_val = 1.0 #fake data (in SGD it would be different on every epoch)
        # get gradients
        #grad_list = [g for (g,v) in gv]
        (summary_str_grad,grad_val) = sess.run([merged] + [dydx], feed_dict={b: b_val})

        # applies the gradients
        [summary_str_apply_transform,_] = sess.run([merged,apply_transform_op], feed_dict={b: b_val})
        writer.add_summary(summary_str_apply_transform, i)
