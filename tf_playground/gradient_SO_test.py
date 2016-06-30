#Just give you a simple example. Understand it and try your specific task out.

#Initialize required symbols.

x = tf.Variable(0.5)
y = x*x
opt = tf.train.AdagradOptimizer(0.1)
grads = opt.compute_gradients(y)

grad_placeholder = [ tf.placeholder("float", shape=grad[1].get_shape()), grad[1] for grad in grads]
apply_placeholder_op = opt.apply_gradients(grad_placeholder)

transform_grads = [(function1(grad[0]), grad[1]) for grad in grads] #list [(grad,var)]
apply_transform_op = opt.apply_gradients(transform_grads)
#Initialize

sess = tf.Session()
sess.run(tf.initialize_all_variables())
#Get all gradients

grad_vals = sess.run([grad[0] for grad in grads])
#Apply gradients

feed_dict = {}
for i in xrange(len(grad_placeholder)):
    feed_dict[grad_placeholder[i][0]] = function2(grad_vals[i])
sess.run(apply_placeholder_op, feed_dict=feed_dict)
sess.run(apply_transform_op)
#Note: the code hasn't been tested by myself, but I confirm the code is legal except minor code errors.
#Note: function1 and function2 is kind of computation, such as 2*x, x^e or e^x and so on.
