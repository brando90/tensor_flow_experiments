## TRAIN
if phase_train is not None:
    #DO BN
    feed_dict_train = {x:X_train, y_:Y_train, phase_train: False}
    feed_dict_cv = {x:X_cv, y_:Y_cv, phase_train: False}
    feed_dict_test = {x:X_test, y_:Y_test, phase_train: False}
else:
    #Don't do BN
    feed_dict_train = {x:X_train, y_:Y_train}
    feed_dict_cv = {x:X_cv, y_:Y_cv}
    feed_dict_test = {x:X_test, y_:Y_test}

def get_batch_feed(X, Y, M, phase_train):
    mini_batch_indices = np.random.randint(M,size=M)
    Xminibatch =  X[mini_batch_indices,:] # ( M x D^(0) )
    Yminibatch = Y[mini_batch_indices,:] # ( M x D^(L) )
    if phase_train is not None:
        #DO BN
        feed_dict = {x: Xminibatch, y_: Yminibatch, phase_train: True}
    else:
        #Don't do BN
        feed_dict = {x: Xminibatch, y_: Yminibatch}
    return feed_dict

with tf.Session() as sess:
    sess.run( tf.initialize_all_variables() )
    for iter_step in xrange(steps):
        feed_dict_batch = get_batch_feed(X_train, Y_train, M, phase_train)
        # Collect model statistics
        if iter_step%report_error_freq == 0:
            train_error = sess.run(fetches=l2_loss, feed_dict=feed_dict_train)
            cv_error = sess.run(fetches=l2_loss, feed_dict=feed_dict_cv)
            test_error = sess.run(fetches=l2_loss, feed_dict=feed_dict_test)

            do_stuff_with_errors(train_error, cv_error, test_error)
        # Run Train Step
        sess.run(fetches=train_step, feed_dict=feed_dict_batch)
