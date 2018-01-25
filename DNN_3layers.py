# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 20:35:33 2016

@author: qiyan

# This program also includes the graphing of the data of loss 
    and accuracy
"""
#import time as tm
#import smtplib
#from email.mime.text import MIMEText
#from email.header import Header
import sys
import pandas as pd
import numpy as np
import tensorflow as tf

file_name_in = sys.argv[1]
dev_name = sys.argv[2]
if dev_name == "gpu0":
    dev_name = "/gpu:0"
elif dev_name == "gpu1":
    dev_name = "/gpu:1"
else:
    dev_name = "/cpu:0"
gpu = dev_name
cpu = '/cpu:0'
#
learning_rate = 1e-4
training_epochs = 1000
batch_size = 128 #2^n 
ref_step = 800
#

#to_file = './result/DNN_Result_3hidlayer.csv'
#n_hidden_limit = 230
row = 1;

n_input = 196
n_hidden_1 = 100
n_hidden_2 = 70
n_hidden_3 = 30
#n_hidden_4 = 10
n_classes = 3

path = './modified/separated/'
#num = '2'
#name = 'SZ000001_10yrs'
name = file_name_in + '_10yrs'
#time = '5min'
#K_num = '/K' + num + '/'
#path = './original/'
#norm = '_normalized_'
train_0 =  name + '_0.csv'
#train_0 =  name + 'train_0.csv'
#train_0 = name + time + norm + 'train_K' + str(num) + '_0.csv'
train_1 =  name + '_1.csv'
#train_1 =  name + 'train_1.csv'
#train_1 = name + time + norm + 'train_K' + str(num) + '_1.csv'
train_2 =  name + '_2.csv'
#train_2 =  name + 'train_2.csv'
#train_2 = name + time + norm + 'train_K' + str(num) + '_2.csv'
test =  name + '_test.csv'
test_ratio = 1;

print ("training: " + path + name)

# for both training and testing: 
# col 4 : label
# col 5 : tangent
# col 6 ~ 201: features 
    
#to make sure that the samples are balanced
df_train_0 = pd.read_csv(path + train_0)
print ("train_0's: ", len(df_train_0))
df_train_1 = pd.read_csv(path + train_1)
print ("train_1's: ",len(df_train_1))
df_train_2 = pd.read_csv(path + train_2)
print ("train_2's: ",len(df_train_2))
df_test_total = pd.read_csv(path + test)
print ("testing total: ", len(df_test_total))

#load the data from csv
def get_training_data(df_train_0,df_train_1,df_train_2):
    
    rows_req = int(min(len(df_train_0),len(df_train_1),len(df_train_2))*0.9)

    df_train_0 = df_train_0.sample(frac=1,replace=True).reset_index(drop=True)
    df_train_0 = df_train_0.loc[1:rows_req,"labels":"mva100"]
    df_train_1 = df_train_1.sample(frac=1,replace=True).reset_index(drop=True)
    df_train_1 = df_train_1.loc[1:rows_req,"labels":"mva100"]
    df_train_2 = df_train_2.sample(frac=1,replace=True).reset_index(drop=True)
    df_train_2 = df_train_2.loc[1:rows_req,"labels":"mva100"]
    frames = [df_train_0,df_train_1,df_train_2]
    df_train = pd.concat(frames)
    df_train = df_train.sample(frac=1,replace=True).reset_index(drop=True)
    
    testing_set = len(df_test_total) * test_ratio # controls the number of testing_set   
    df_test = df_test_total.loc[1:testing_set,"labels":"mva100"]
    
    return df_train,df_test
    
def load_data():
    #get the training data:
    df_train,df_test = get_training_data(df_train_0,df_train_1,df_train_2)
    #Separate the features and the labels
    df_train_features = df_train.loc[:,"lag1":"mva100"]
    df_train_features_npy = df_train_features.values
    df_train_labels = df_train.loc[:,"labels"]
    df_train_labels_npy  = (df_train_labels.values).astype("float32")

    df_test_features = df_test.loc[:,"lag1":"mva100"]
    df_test_features_npy  = df_test_features.values
    df_test_labels = df_test.loc[:,"labels"]
    df_test_labels_npy  = (df_test_labels.values).astype("float32")
        
    y_train = tf.one_hot(indices = df_train_labels_npy,depth = 3,on_value = 1.0,off_value = 0.0)
    y_test = tf.one_hot(indices = df_test_labels_npy,depth = 3,on_value = 1.0,off_value = 0.0)
    
    return df_train_features_npy,df_train_labels_npy,y_train,df_test_features_npy,df_test_labels_npy,y_test

#define weight and bias

def w_b():
    with tf.device(gpu):
        weight = {
            'w1': tf.Variable(tf.random_normal([n_input, n_hidden_1], stddev = 0.01)),
            'w2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2], stddev = 0.01)),
            'w3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3], stddev = 0.01)),
            #'w4': tf.Variable(tf.random_normal([n_hidden_3, n_hidden_4], stddev = 0.01)),
            'out': tf.Variable(tf.random_normal([n_hidden_3, n_classes], stddev = 0.01))
        }
        
        bias = {
            'b1': tf.Variable(tf.ones([n_hidden_1])),
            'b2': tf.Variable(tf.ones([n_hidden_2])),
            'b3': tf.Variable(tf.ones([n_hidden_3])),
            #'b4': tf.Variable(tf.ones([n_hidden_4])),
            'out': tf.Variable(tf.zeros([n_classes]))
        }
        
    return weight,bias
    
keep_prob = tf.placeholder(tf.float32)

#create model
def mlp(x, weight,bias):
    with tf.device(gpu):
        hid_layer_1 = tf.nn.bias_add( tf.matmul(x,weight['w1']), bias['b1'])
        with tf.device(cpu):
            mean1,var1 = tf.nn.moments(hid_layer_1,[0])
        hid_layer_1_bn = tf.nn.batch_normalization(hid_layer_1,mean1,var1,offset=0,scale=1,variance_epsilon=0.001)        
        hid_layer_1_bn = tf.nn.relu(hid_layer_1_bn)
        #hid_layer_1_drop = tf.nn.dropout(hid_layer_1,keep_prob)
        
        hid_layer_2 = tf.nn.bias_add(tf.matmul(hid_layer_1_bn,weight['w2']), bias['b2'])
        #mean2,var2 = tf.nn.moments(hid_layer_1,[0]) 
        #bn_1 = tf.nn.batch_normalization(hid_layer_2,mean2,var2,offset=0,scale=1,variance_epsilon=0.001)        
        hid_layer_2 = tf.nn.relu(hid_layer_2)
        #hid_layer_2_drop = tf.nn.dropout(hid_layer_2,keep_prob)
               
        hid_layer_3 = tf.nn.bias_add(tf.matmul(hid_layer_2,weight['w3']), bias['b3'])
        #mean3,var3 = tf.nn.moments(hid_layer_3,[0])
        #hid_layer_3_bn = tf.nn.batch_normalization(hid_layer_3,mean3,var3,offset=0,scale=1,variance_epsilon=0.001)
        hid_layer_3 = tf.nn.relu(hid_layer_3)
        #hid_layer_3_drop = tf.nn.dropout(hid_layer_3,keep_prob)

        #hid_layer_4 = tf.nn.bias_add( tf.matmul(hid_layer_3_drop,weight['w4']), bias['b4'])
        #hid_layer_4 = tf.nn.relu(hid_layer_4)
        #hid_layer_4_drop = tf.nn.dropout(hid_layer_4,keep_prob)
        out_layer = tf.nn.softmax(tf.matmul(hid_layer_3, weight['out']) + bias['out'])   
    return out_layer

def run_test(x_train,train_labels,y_train,x_test,test_labels,y_test):
    with tf.device(gpu):
        x = tf.placeholder(tf.float32, [None, n_input])
        y = tf.placeholder(tf.float32, [None, n_classes])
        
        weight,bias = w_b();
        pred = mlp(x,weight,bias)
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=y))
        corr_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(corr_pred, tf.float32))
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
        
        init = tf.global_variables_initializer()
    ##########################################        
#    with tf.device(cpu):
#        tf.scalar_summary(name +'_accuracy',accuracy);
#        tf.scalar_summary(name + '_loss', cost);
#        merged = tf.merge_all_summaries()
#        train_writer = tf.train.SummaryWriter('./graph')
    ##########################################
    
    config = tf.ConfigProto(allow_soft_placement=False)
#    config.gpu_options.
    #config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    #start the training process:
    
    with tf.Session(config=config) as sess:
        with tf.device(gpu):
            sess.run(init)
            ######################################
#            merged_step = 0
            ######################################
            total_batch = int(len(y_train.eval())/batch_size)
            print ("current layer info:-------n_hidden_1=", n_hidden_1,", n_hidden_2=", n_hidden_2,", n_hidden_3=", n_hidden_3)
           # print "row = "+ str(row)            
            print ("total_batch: ", total_batch)
            print ("-------------------training set: " , len(y_train.eval()))
            print ("-------------------testing set: " , len(y_test.eval()))

            for epoch in range (training_epochs):
                #for itera in range (10):#cross-validation 10 times
                x_train,train_labels,y_train,x_test,test_labels,y_test = load_data()
                y_train = y_train.eval()
                y_test = y_test.eval()
                for step in range(total_batch):
                    #print "cross validation iteration: ", itera
                    batch_index_front = step * batch_size
                    batch_index_end = (step+1) * batch_size
                    
                    batch_x = x_train[batch_index_front:batch_index_end, :]
                    batch_y = y_train[batch_index_front:batch_index_end, :]
                    #!!! batch normalization vs. dropout
                    #!!! weight decay, learning_rate decrement
#                    _, result, accu, loss = sess.run([optimizer,merged,accuracy,cost], feed_dict={x:batch_x, y:batch_y, keep_prob:1.0})
                    _, accu, loss = sess.run([optimizer,accuracy,cost], feed_dict={x:batch_x, y:batch_y, keep_prob:1.0})
                    if ( (step % ref_step == 0) and (step > 0)):
                        #############################################
#                        #accu,loss = sess.run([accuracy,loss], feed_dict = {x:batch_x, y:batch_y})                        #############################################
#                        train_writer.add_summary(result, merged_step);
#                        merged_step += 1
                        #############################################                         
                        print("epoch: " + str(epoch+1) + ", sample: " + str(step*batch_size) + ", Loss=" + "{:.8f}".format(loss) + ", Training Accuracy=" + "{:.4f}".format(accu))
            #print "cross validation iteration: ", itera
            print ("Optimization Finished!")
            #get accuracy and prediction array: [testing#, 3]
            te_accu,y_test_pred = sess.run([accuracy,pred], feed_dict={x: x_test, y: y_test, keep_prob:1.0})
            #get the max of the predction array
            y_test_pred = (tf.argmax(y_test_pred,1)).eval();
            y_test_pred = tf.one_hot(indices = y_test_pred,depth = 3,on_value = 1.0,off_value = 0.0)
            y_test_pred = y_test_pred.eval(); # to [1 0 0]'s mode
            
            tr_accu,y_train_pred = sess.run([accuracy,pred], feed_dict={x: x_train, y: y_train, keep_prob:1.0})
            y_train_pred = (tf.argmax(y_train_pred,1)).eval();
            y_train_pred = tf.one_hot(indices = y_train_pred,depth = 3,on_value = 1.0,off_value = 0.0)
            y_train_pred = y_train_pred.eval();
            
            #to get the resulting weights:
            weights = [weight['w1'].eval(), weight['w2'].eval(),weight['w3'].eval(),
                        weight['out'].eval()]

            print ("-------------------training set: "+name+"length: " , len(y_train))
            print ("Training Accuracy: ",tr_accu )
            
            print ("----------------------testing set: "+name+"length: ", len(x_test))
            print ("Test Accuracy: ", te_accu)
            
        sess.close()                
    tf.reset_default_graph()
    return y_test, y_test_pred,x_test,y_train_pred,y_train,tr_accu,te_accu,weights
    
#calculate confusion matrix and correct predication's rate(in float)
def confu_mat_cal(y_test,y_test_pred):
    confu_mat = np.zeros( (3,3), dtype=int)
    #test_data = test_end-test_start
    for i in range(len(y_test)):                                                            
        if(np.all(y_test[i]== [1.0,0.0,0.0])):                                          
            if(np.all(y_test_pred[i] == [1.0,0.0,0.0])):                          
                confu_mat[0][0] += 1                                              
            if(np.all(y_test_pred[i] == [0.0,1.0,0.0])):            
                confu_mat[0][1] += 1                                              
            if(np.all(y_test_pred[i] == [0.0,0.0,1.0])):                          
                confu_mat[0][2] += 1                                              
        if(np.all(y_test[i] == [0.0,1.0,0.0])):                                   
            if(np.all(y_test_pred[i] == [1.0,0.0,0.0])):                          
                confu_mat[1][0] += 1                                              
            if(np.all(y_test_pred[i] == [0.0,1.0,0.0])):                          
                confu_mat[1][1] += 1                                              
            if(np.all(y_test_pred[i] == [0.0,0.0,1.0])):                          
                confu_mat[1][2] += 1                                              
        if(np.all(y_test[i] == [0.0,0.0,1.0])):                                   
            if(np.all(y_test_pred[i] == [1.0,0.0,0.0])):                          
                confu_mat[2][0] += 1                                              
            if(np.all(y_test_pred[i] == [0.0,1.0,0.0])):                          
                confu_mat[2][1] += 1                                              
            if(np.all(y_test_pred[i] == [0.0,0.0,1.0])):                          
                confu_mat[2][2] += 1
    #get the rate:
    sum_true_0 = confu_mat[0][0] + confu_mat[0][1] + confu_mat[0][2]
    sum_true_1 = confu_mat[1][0] + confu_mat[1][1] + confu_mat[1][2]
    sum_true_2 = confu_mat[2][0] + confu_mat[2][1] + confu_mat[2][2] 
    rate_0 = confu_mat[0][0]/ float(sum_true_0)
    rate_1 = confu_mat[1][1]/ float(sum_true_1)
    rate_2 = confu_mat[2][2]/ float(sum_true_2)
    return confu_mat, rate_0, rate_1, rate_2

#bonus: send an email to nofification when the program finishes
#def get_time():
#    time_r = tm.strftime('%m-%d-%H:%M:%S',tm.localtime(tm.time()))
#    return time_r
#def send_email():
#    time = get_time();   
#    sender = 'program_finished@ts.com'
#    receiver = '2282691271@qq.com'
#    subject = 'program finished!'    
#
#    msg = MIMEText('program finished at ' + time)
#    msg['Subject'] = Header(subject,'utf-8')
#    
#    try:
#        smtpObj = smtplib.SMTP('localhost')
#        smtpObj.sendmail(sender,receiver,msg.as_string())
#        print ("email sent!")
#    except smtplib.SMTPException:
#        print ("failed to send email!")
#    
#determines that the programs will run independently(apart from Spyder)
#if __name__ == "__main__":
#df_record= pd.read_csv(to_file)    
#def record_in(row,col,value):
#    df_record.ix[row,col] = value
#def record_data(row,n_hidden_1,n_hidden_2,n_hidden_3,train_accu,train_rate_0,train_rate_1,
#                train_rate_2,test_accu,test_rate_0,test_rate_1,test_rate_2):
#    record_in(row,'Hidden_1',n_hidden_1)
#    record_in(row,'Hidden_2',n_hidden_2)
#    record_in(row,'Hidden_3',n_hidden_3)
#    record_in(row,'Train_accu',train_accu)
#    record_in(row,'Train_rate_0',train_rate_0)
#    record_in(row,'Train_rate_1',train_rate_1)
#    record_in(row,'Train_rate_2',train_rate_2)
#    record_in(row,'Test_accu',test_accu)
#    record_in(row,'Test_rate_0',test_rate_0)
#    record_in(row,'Test_rate_1',test_rate_1)
#    record_in(row,'Test_rate_2',test_rate_2)

#initial condition
#-----------------------------main program   

#while(n_hidden_1 < n_hidden_limit):
#    row = row + 1;
#    
#    while(n_hidden_2 < n_hidden_1):
#        
#        while(n_hidden_3 < n_hidden_2):
#row = 2
keep_prob = tf.placeholder(tf.float32)
x_train,train_labels,y_train,x_test,test_labels,y_test = load_data()
y_test,y_test_pred,x_test,y_train_pred,y_train,train_accu,test_accu,weights = run_test(x_train,train_labels,y_train,x_test,test_labels,y_test)      
train_confu_mat,train_rate_0,train_rate_1,train_rate_2 = confu_mat_cal(y_train,y_train_pred)
print ("result for: " + path + name)
print ("-------------------training set: " , len(y_train))
print ('Training confu_mat: ')
print (train_confu_mat)
print ('rate 0: ',train_rate_0)
print ('rate 1: ',train_rate_1)
print ('rate 2: ',train_rate_2)

test_confu_mat,test_rate_0,test_rate_1,test_rate_2 = confu_mat_cal(y_test,y_test_pred)
print ("-------------------testing set: " , len(y_test))
print ('Testing confu_mat: ')
print (test_confu_mat)
print ('rate 0: ',test_rate_0)
print ('rate 1: ',test_rate_1)
print ('rate 2: ',test_rate_2)

#print (">>>>>>> w1 are: ")
#print (weights[0])
#
#print (">>>>>>> w2 are: ")
#print (weights[1])
#
#print (">>>>>>> w3 are: ")
#print (weights[2])
#
#print (">>>>>>> w_out are: ")
#print (weights[3])


#print ('-----------------saving to graph: ' +to_file+'....')      
#record_data(row,n_hidden_1,n_hidden_2,n_hidden_3,train_accu,
#            train_rate_0,train_rate_1,train_rate_2,test_accu,
#            test_rate_0,test_rate_1,test_rate_2)
           
#df_record.to_csv(to_file,index=False,mode='wb')
#            
#            row = row + 1
#            n_hidden_3 = n_hidden_3 + 5
#            
#        n_hidden_2 = n_hidden_2 + 5
#        n_hidden_3 = 20
#        
#    n_hidden_1 = n_hidden_1 + 5
#    n_hidden_2 = 30
    
#send_email()




    
