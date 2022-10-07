
import numpy as np
import tensorflow as tf
import pickle as p 
import scipy.io as sio

np.random.seed(100)

def batch_out(sample_,label_,batch_size,epochs):
    sample=[]
    label=[]
    input_queue=tf.train.slice_input_producer([sample_,label_],shuffle=False,num_epochs=epochs)
    sample_batch,label_batch=tf.train.batch(input_queue,batch_size=batch_size)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    coord = tf.train.Coordinator()
    threads=tf.train.start_queue_runners(sess,coord)
    try:
        while not coord.should_stop():
            data_X,data_y=sess.run([sample_batch,label_batch])
            sample.append(data_X)
            label.append(data_y)
    except tf.errors.OutOfRangeError:
        print('Done train_batch')
    finally:
        coord.request_stop()
    coord.join(threads)
    sess.close()
    return sample,label
#num = 1000
#y = np.asarray(range(0, num))
#X = np.random.random([num, 10])
#batch_size=100
#epochs=2
#[sample,label]=tf_batch_out(X,y,batch_size,epochs)
