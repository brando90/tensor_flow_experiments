import tensorflow as tf

# Create a node Constant op that produces a 1x2 matrix.
matrix1 = tf.constant([[3., 3.]])
matrix2 = tf.constant([[2.],[2.]])

# Create a node Matmul op
product = tf.matmul(matrix1, matrix2)

## Launch the default graph.
sess = tf.Session()
## To run the matmul op we call the session 'run()' method passing 'product'
result = sess.run(product)
print(result)
print(type(result))
## Close the Session when we're done.
sess.close()

## The Session closes automatically at the end of the with block.
with tf.Session() as sess:
  result = sess.run(product)
  print(result)
  print(type(result))
