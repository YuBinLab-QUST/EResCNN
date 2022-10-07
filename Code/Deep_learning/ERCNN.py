import os,sys
import tensorflow as tf
import scipy.io as sio
import pickle as p
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier,
                              GradientBoostingClassifier, ExtraTreesClassifier)
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.preprocessing import scale
from sklearn.metrics import accuracy_score
import utils.tools as utils

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

num_classes=2
fold=int(fold_num)

b1 = open('five_fold_test.data', 'rb') 
test_index=p.load(b1) 

b2 = open('five_fold_train.data', 'rb') 
train_index=p.load(b2) 

X_train=X_shu[train_index[fold]]
y_train=y_shu[train_index[fold]]
X_test=X_shu[test_index[fold]]
y_test=y_shu[test_index[fold]]

model_file=tf.train.latest_checkpoint("RCNN_"+str(fold)+"/")
#model_file=tf.train.latest_checkpoint("RCNN_"+str(fold)+"/")

print(model_file)

init1 = tf.global_variables_initializer()
g1=tf.Graph()
sess1=tf.Session(graph=g1)

with g1.as_default():
    model_file=tf.train.latest_checkpoint("RCNN_"+str(fold)+"/")
    new_saver1=tf.train.import_meta_graph(model_file+".meta")
    new_saver1.restore(sess1, model_file)                                           
    prediction=tf.get_collection("prediction")[0]
    fc=tf.get_collection("fc")[0]
    accuracy=tf.get_collection("accuracy")[0]
    fc_end=tf.get_collection("fc_end")[0]
    graph=tf.get_default_graph()
    X=graph.get_operation_by_name("input_x").outputs[0]
    y=graph.get_operation_by_name("input_y").outputs[0]  
    
    y_train_=utils.to_categorical(y_train)
    y_test_=utils.to_categorical(y_test)
    
    DNN_accuracy_train,DNN_y_score_train,DNN_fc_train,DNN_fc_end_train=sess1.run([accuracy,prediction,fc,fc_end],feed_dict={X: X_train,                                                      
                                                      y:y_train_})
                                                        
    DNN_accuracy,DNN_y_score,DNN_fc,DNN_fc_end=sess1.run([accuracy,prediction,fc,fc_end],feed_dict={X: X_test,                                                      
                                                      y:y_test_})
pre_score=DNN_y_score
pre_class= utils.categorical_probas_to_classes(pre_score)
test_class=y_test
DNN_acc, precision,npv, sensitivity, specificity, mcc,f1 = utils.calculate_performace(len(pre_class), pre_class, test_class)
print("DNN_acc:",DNN_acc)
sess1.close()

LGB = lgb.LGBMClassifier(n_estimators=500,learning_rate=0.1)
XGB = xgb.XGBClassifier(max_depth=20, n_estimators=500, objective="binary:logistic",n_jobs=-1)                                               
RF  = RandomForestClassifier(n_estimators=500)
ERT = ExtraTreesClassifier(n_estimators=500)

LGB_hist=LGB.fit(X_train, y_train)
y_LGB_1=LGB.predict_proba(X_test)

LGB_DNN=LGB.fit(DNN_fc_train,y_train)
y_LGB_2=LGB.predict_proba(DNN_fc)

XGB_hist=XGB.fit(X_train, y_train)
y_XGB_1=XGB.predict_proba(X_test)

XGB_hist=XGB.fit(DNN_fc_train,y_train)
y_XGB_2=XGB.predict_proba(DNN_fc)

RF_hist=RF.fit(X_train, y_train)
y_RF_1=RF.predict_proba(X_test)

RF_hist=RF.fit(DNN_fc_train,y_train)
y_RF_2=RF.predict_proba(DNN_fc)

ERT_hist=ERT.fit(X_train, y_train)
y_ERT_1=ERT.predict_proba(X_test)

ERT_hist=ERT.fit(DNN_fc_train,y_train)
y_ERT_2=ERT.predict_proba(DNN_fc)

pre_class1= utils.categorical_probas_to_classes(y_LGB_1)
pre_class2= utils.categorical_probas_to_classes(y_LGB_2)
pre_class3= utils.categorical_probas_to_classes(y_XGB_1)
pre_class4= utils.categorical_probas_to_classes(y_XGB_2)
pre_class5= utils.categorical_probas_to_classes(y_RF_1)
pre_class6= utils.categorical_probas_to_classes(y_RF_2)
pre_class7= utils.categorical_probas_to_classes(y_ERT_1)
pre_class8= utils.categorical_probas_to_classes(y_ERT_2)


class1=accuracy_score(test_class,pre_class1)
class2=accuracy_score(test_class,pre_class2)
class3=accuracy_score(test_class,pre_class3)
class4=accuracy_score(test_class,pre_class4)
class5=accuracy_score(test_class,pre_class5)
class6=accuracy_score(test_class,pre_class6)
class7=accuracy_score(test_class,pre_class7)
class8=accuracy_score(test_class,pre_class8)

print("pre_class1:",class1)
print("pre_class2:",class2)
print("pre_class3:",class3)
print("pre_class4:",class4)
print("pre_class5:",class5)
print("pre_class6:",class6)
print("pre_class7:",class7)
print("pre_class8:",class8)


meta_test=np.column_stack((DNN_y_score,y_LGB_1,y_LGB_2,y_XGB_1,y_XGB_2,y_RF_1,y_RF_2,y_ERT_1,y_ERT_2))

meta_test_csv = pd.DataFrame(data=meta_test)
meta_test_csv.to_csv('meta_test'+fold_num+filename+'.csv')

number=meta_test.shape[1]
pre_score1=[]
pre_score2=[]
for i in range(0,number,2):
    score1=meta_test[:,i]
    score2=meta_test[:,i+1]
    pre_score1.append(score1)
    pre_score2.append(score2)
prescore1=np.mean(np.array(pre_score1),axis=0)
prescore2=np.mean(np.array(pre_score2),axis=0)
pre_score=np.column_stack((prescore1,prescore2))

pre_score_csv = pd.DataFrame(data=pre_score)
pre_score_csv.to_csv('pre_score'+fold_num+filename+'.csv')



pre_class= utils.categorical_probas_to_classes(pre_score)
test_class=y_test
acc, precision,npv, sensitivity, specificity, mcc,f1 = utils.calculate_performace(len(pre_class), pre_class, test_class)
print("Testing Accuracy:",acc)


#pre_score=LR.predict_proba(meta_test)

#
