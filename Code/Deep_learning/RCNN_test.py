import os,sys
import tensorflow as tf
import xgboost as xgb
import scipy.io as sio
import pickle as p
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
import utils.tools as utils
from tf_batch_out import  batch_out
# find the path
Script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))

# read selected features
filename = sys.argv[1]
fold_num = sys.argv[2]
#yeast_data=sio.loadmat('logic_dimension_Matine/logistic_2.mat')
yeast_data=sio.loadmat(filename)
#yeast_data=sio.loadmat('logic_dimendion/logistic_2.mat')

protein_A=yeast_data.get('protein_A')#取出字典里的data
protein_B=yeast_data.get('protein_B')
X_shu=np.hstack((protein_A,protein_B))
protein_label=yeast_data.get('protein_label') 
column=protein_A.shape[1]
y_shu=protein_label.T.ravel()

def get_shuffle(dataset,label):    
    #shuffle data
    index = [i for i in range(len(label))]
    np.random.shuffle(index)
    dataset = dataset[index]
    label = label[index]
    return dataset,label,index 
#X_shu,y_shu,index=get_shuffle(X_shu_,y_shu_)

# Training Parameters
learning_rate = 0.001
batch_size = 2000
epochs=150
fold=int(fold_num)
model_path = "RCNN_"+str(fold)+"/model.ckpt"
# Network Parameters
num_input =protein_A.shape[1]
num_classes = 2 
dropout = 0.5 # Dropout, probability to drop a unit
#placeholder
X = tf.placeholder(tf.float32, [None,num_input*2],name="input_x")
y = tf.placeholder(tf.float32, [None,num_classes],name="input_y")
keep_prob = tf.placeholder(tf.float32)
def conv_net(X,  keep_prob, reuse, is_training):
    #tf.reset_default_graph()
    row=tf.shape(X)[0]
    
    with tf.variable_scope('ConvNet', reuse=reuse):
        
        x_A=tf.slice(X,[0,0],[row,num_input])
        x_B=tf.slice(X,[0,num_input],[row,num_input])
               
        x_raw_A = tf.layers.dense(x_A, num_input,activation=tf.nn.relu)   
        xA = tf.reshape(x_raw_A, shape=[-1,num_input,1])
        
        xA=tf.cast(xA,dtype=tf.float32)
        ##########################################################
        conv1_A = tf.layers.conv1d(xA, filters=1, kernel_size=10,padding='same', activation=tf.nn.relu)       
        pool1 = tf.layers.max_pooling1d(conv1_A, pool_size=2,strides=2,padding='same')  
        res1_A=tf.concat([xA,pool1],1)     
        ##########################################################
        conv2_A = tf.layers.conv1d(res1_A, filters=1, kernel_size=10,strides=3,padding='same', activation=tf.nn.relu)            
        pool2 = tf.layers.max_pooling1d(conv2_A, pool_size=2,strides=2,padding='same') 
        res2_A= tf.concat([res1_A,pool2],1)
        ##########################################################
        conv3_A = tf.layers.conv1d(res2_A, filters=1, kernel_size=10,strides=3,padding='same', activation=tf.nn.relu)
        pool3 = tf.layers.max_pooling1d(conv3_A, pool_size=2,strides=2,padding='same')
        res3_A= tf.concat([res2_A,pool3],1)
        #pool_end=tf.layers.max_pooling1d(res3_A, pool_size=2,strides=2,padding='same')
        fc_A = tf.contrib.layers.flatten(res3_A)
        fc_A_end=tf.layers.dense(fc_A,256,activation=tf.nn.relu)
        ############################
        x_raw_B = tf.layers.dense(x_B,num_input,activation=tf.nn.relu)
        xB = tf.reshape(x_raw_B, shape=[-1, num_input,1])
        
        xB=tf.cast(xB,dtype=tf.float32)
        ##########################################################
        conv1_B = tf.layers.conv1d(xB, filters=1, kernel_size=10,padding='same', activation=tf.nn.relu)       
        pool1_B = tf.layers.max_pooling1d(conv1_B, pool_size=2,strides=2,padding='same')  
        res1_B=tf.concat([xB,pool1_B],1)     
        ##########################################################
        conv2_B = tf.layers.conv1d(res1_B, filters=1, kernel_size=10,strides=3,padding='same', activation=tf.nn.relu)            
        pool2_B = tf.layers.max_pooling1d(conv2_B, pool_size=2,strides=2,padding='same') 
        res2_B= tf.concat([res1_B,pool2_B],1)
        ##########################################################
        conv3_B = tf.layers.conv1d(res2_B, filters=1, kernel_size=10,strides=3,padding='same', activation=tf.nn.relu)
        pool3_B = tf.layers.max_pooling1d(conv3_B, pool_size=2,strides=2,padding='same')
        res3_B= tf.concat([res2_B,pool3_B],1)
        #pool_end=tf.layers.max_pooling1d(res3_A, pool_size=2,strides=2,padding='same')
        fc_B = tf.contrib.layers.flatten(res3_B)
        
        fc_B_end=tf.layers.dense(fc_B,256,activation=tf.nn.relu)
        
        fc0=tf.concat([fc_A_end,fc_B_end],1) 
        
#        fc0 = tf.multiply(fc_A, fc_B)      
        fc1_dropout = tf.layers.dropout(fc0, rate= keep_prob, training=is_training)
#
        fc2 = tf.layers.dense(fc1_dropout, 256,activation=tf.nn.relu)
        
        fc2_dropout = tf.layers.dropout(fc2, rate= keep_prob, training=is_training)
        
        fc3 = tf.layers.dense(fc2_dropout, 128,activation=tf.nn.relu)

        out = tf.layers.dense(fc3, 2)
        
    return out,fc0,fc3

# Construct model    
logits,fc,fc_end= conv_net(X, keep_prob, reuse=False,is_training=False)
                          
prediction = tf.nn.softmax(logits)
pred_classes = tf.argmax(logits, axis=1)  
# Define loss and optimizer
    
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


b1 = open('five_fold_test.data', 'rb') 
test_index=p.load(b1) 

b2 = open('five_fold_train.data', 'rb') 
train_index=p.load(b2) 

X_train=X_shu[train_index[fold]]
y_train=y_shu[train_index[fold]]
X_test=X_shu[test_index[fold]]
y_test=y_shu[test_index[fold]]

data_x,data_y= batch_out(X_train,y_train,batch_size,epochs)
batch_x,batch_y=[],[]
# Initialize the global variables
init1 = tf.global_variables_initializer()
# save model
saver = tf.train.Saver(max_to_keep=3)
tf.add_to_collection("prediction",prediction)
tf.add_to_collection("fc",fc)
tf.add_to_collection("accuracy",accuracy)
tf.add_to_collection("fc_end",fc_end)

# Running first session
print("Starting first session...")
with tf.Session() as sess:
    sess.run(init1)
    step=0
    while step<len(data_x):
        batch_x=np.array(data_x[step])
        batch_y=np.array(data_y[step])
        batch_y=utils.to_categorical(batch_y)
        sess.run(train_op,feed_dict={X: batch_x,                                                      
                                          y:batch_y,
                                          keep_prob:dropout})             
        if step % 50 == 0:
            loss,acc=sess.run([loss_op,accuracy],feed_dict={X: batch_x,                                                      
                                                 y:batch_y,
                                                 keep_prob:dropout})
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))
            save_path=saver.save(sess,model_path,global_step=step)
        step=step+1
        batch_x,batch_y=[],[]
    print("Optimization Finished!")
    y_train_=utils.to_categorical(y_train)
    y_test_=utils.to_categorical(y_test)
    #####save model
print("Starting second session...")

with tf.Session() as sess:
    # Initialize variables
    model_file=tf.train.latest_checkpoint("RCNN_"+str(fold)+"/")
    saver.restore(sess, model_file)
    prediction=tf.get_collection("prediction")[0]
    fc=tf.get_collection("fc")[0]
    accuracy=tf.get_collection("accuracy")[0]
    
#    graph=tf.get_default_graph()
    print("Model restored from file: %s" % save_path)

    accuracy_train,y_score_train,fc_train,fc_end_train=sess.run([accuracy,prediction,fc,fc_end],feed_dict={X: X_train,                                                      
                                                      y:y_train_,
                                                      keep_prob:1})  
    accuracy,y_score,fc,fc_end=sess.run([accuracy,prediction,fc,fc_end],feed_dict={X: X_test,                                                      
                                                      y:y_test_,
                                                      keep_prob:1})
print(accuracy_train)
print(accuracy)
pre_score=y_score
pre_class= utils.categorical_probas_to_classes(pre_score)
test_class=y_test
acc, precision,npv, sensitivity, specificity, mcc,f1 = utils.calculate_performace(len(pre_class), pre_class, test_class)
print("Testing Accuracy:",acc)