## run cmd to collect model: python qm_tb_scopes.py --logdir=/tmp/qm_logs
## show board on browser run cmd: tensorboard --logdir=/tmp/qm_logs
## browser: http://localhost:6006/

import tensorflow as tf
import shutil

def delete_dir_contents(path):
    shutil.rmtree(path)
    return

def get_quaratic(b):
    with tf.variable_scope('quadratic'):
        x = tf.get_variable('x', [], initializer= tf.constant_initializer(10.0))
        # b placeholder (simualtes the "data" part of the training)
        # make model (1/2)(x-b)^2
        xx_b = 0.5*tf.pow(x-b,2)
        y=xx_b
        return y,x

def T(g, decay=1.0):
    #return decayed gradient
    return decay*g

b = tf.placeholder(tf.float32,name='b')
y,x1 = get_quaratic(b)
learning_rate = 1.0
# get optimizer
opt = tf.train.GradientDescentOptimizer(learning_rate)

with tf.variable_scope('quadratic', reuse = True):
    x = tf.get_variable('x')
assert x == x1
gv = opt.compute_gradients(y,[x])
# transformed gradient variable list = [ (T(gradient),variable) ]
decay = 0.9 # decay the gradient for the sake of the example
# apply transformed gradients
#transofrmed gradients
tgv = [(T(g,decay), v) for (g,v) in gv] #list [(grad,var)]
#apply transform
apply_transform_op = opt.apply_gradients(tgv)

# get first grad
(dydx,_) = tgv[0] #indexes [(g1,v1)]
# track value of x
x_scalar_summary = tf.scalar_summary("x", x)
x_histogram_sumarry = tf.histogram_summary('x_his', x)
# track value of dydx (gradient)
grad_scalar_summary = tf.scalar_summary("dydx", dydx)
grad_histogram_sumarry = tf.histogram_summary('dydx_his', dydx)
with tf.Session() as sess:
    merged = tf.merge_all_summaries()
    tensorboard_data_dump = '/tmp/qm_logs'
    shutil.rmtree(tensorboard_data_dump)
    writer = tf.train.SummaryWriter(tensorboard_data_dump, sess.graph)

    sess.run(tf.initialize_all_variables())
    epochs = 200
    for i in range(epochs):
        b_val = 1.0 #fake data (in SGD it would be different on every epoch)
        # get gradients
        (summary_str_grad,grad_val,_) = sess.run([merged]+[dydx, apply_transform_op], feed_dict={b: b_val})
        #(summary_str_grad,grad_val) = sess.run([merged]+[dydx], feed_dict={b: b_val})
        print 'grad_val',grad_val
        # applies the gradients
        #[summary_str_apply_transform,_] = sess.run([merged,apply_transform_op], feed_dict={b: b_val})
        print 'x',x.eval()
        # write summaries
        writer.add_summary(summary_str_grad, i)
        #writer.add_summary(summary_str_apply_transform, i)
