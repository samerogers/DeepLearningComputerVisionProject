import numpy as np
import random
import tensorflow as tf
import OT_originalROLO_utils as utils
from tensorflow.contrib import rnn
import matplotlib.pyplot as plt
import cv2
import os.path
import datetime
from scipy import stats
from sklearn.preprocessing import MinMaxScaler

class ObjectTracking:
    rolo_weights_file = 'model_min1/model_step6.ckpt' #Modification 1 
    num_steps = 6  # number of frames as an input sequence
    num_feat = 255
    num_predict = 4 # final output of LSTM 6 loc parameters
    num_gt = 4 # groundtruth for x,y,width and height for 4 objectsModification 2
    num_obj2detect =1
    num_input = (255+num_predict)*num_obj2detect
    learning_rate = 0.00001 #training
    training_iters = 210#100000
    batch_size = 1 #128
    display_step = 1
    num_hidden=259
    
    input_x = tf.placeholder("float32", [None, num_steps, num_input])
    initstate = tf.placeholder("float32", [None, 2*num_input]) #state & cell => 2x num_input
    input_y = tf.placeholder("float32", [None, num_gt*num_obj2detect]) #MODIFICATIONTimes 2 is for 2 objects
    # Define weights
    weights = {
        'out': tf.Variable(tf.random_normal([num_hidden, num_predict],stddev=1.0))
    }
    biases = {
        'out': tf.Variable(tf.random_normal([num_predict],stddev=1.0))
    }

    def __init__(self):
        print("init Object Tracking training")
        self.TrackerModel()


    def BuildLSTM(self, name,  input_x, initstate, weights, biases,reuse):
        with tf.variable_scope("LSTM") as scope:
            if (reuse):
                tf.get_variable_scope().reuse_variables()
            #input_x = tf.unstack(input_x ,self.num_steps,axis = 1)#unstach as 1,4108 but 6 times
            lstm = tf.contrib.rnn.LSTMCell(self.num_hidden,state_is_tuple=False) #create actual lstm cell feeding num_input as num_units in that cell for time t
            output,state=tf.nn.dynamic_rnn(lstm, input_x, dtype="float32",time_major=False,initial_state=initstate)
            output = tf.transpose(output, [1, 0, 2]) #
            reuse = True
            tf.get_variable_scope().reuse_variables()

        return output

    def build_LSTM_networks(self):
        print "Building LSTM Network"

        self.lstm_module = self.BuildLSTM('lstm_train', self.input_x, self.initstate, self.weights, self.biases,False)
        self.sess = tf.Session()
        self.sess.run(tf.initialize_all_variables())
        self.saver = tf.train.Saver()
        print "Loading complete!HELLo1" + '\n'

    def training(self):
        print("TRAINING Object Tracker...")
        log_file = open("model_min1/OT-originalLSTM_training-log.txt", "a") #open in append mode
        self.build_LSTM_networks()
        num_videos = 27
        # count=0.0
        # loss_list=[]
        epoches = 27 * 10

        '''Tuning Object Trackeing Model'''    

        predict = self.BuildLSTM('lstm_train', self.input_x, self.initstate, self.weights, self.biases,True)
        print(predict.shape)
        #self.predict_location = tf.concat([predict[0][:, 4097:4101],predict[0][:, 4103:4107]],axis=1)#,predict[0][0][ 4109:4113],predict[0][0][ 4115:4119]],axis=0) #to accomodate 4 obj
        self.predict_location = predict[-1][:,255:]
        print(' predict last location is:', (self.predict_location[-1]))
        #self.predict_location = predict[0][:, 4097:4101]
    #Find Mean square error
        #self.correct_prediction1 = tf.square(self.predict_location[0][:2] - self.input_y[0][:2])
        #self.correct_prediction2 = tf.square(self.predict_location[0][2:] - self.input_y[0][2:])
        #print('shape of correct_prediction:', tf.shape(self.correct_prediction))
        #self.accuracy1 = tf.reduce_mean(self.correct_prediction1)*100
        #self.accuracy2 = tf.reduce_mean(self.correct_prediction2)*100
        self.accuracy1 = tf.nn.l2_loss(self.predict_location[0][:2] - self.input_y[0][:2])
        self.accuracy2 = tf.nn.l2_loss(self.predict_location[0][2:] - self.input_y[0][2:])

        self.optimizer1 = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.accuracy1) # Adam Optimizer
        self.optimizer2 = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.accuracy2) # Adam Optimizer

        self.optimizer = tf.group([self.optimizer1,self.optimizer2])
     # Initializing the variables
        init = tf.initialize_all_variables()

        # Launch the graph
        with tf.Session() as sess:
            sess.run(init)
            self.saver.restore(sess, self.rolo_weights_file)
            for epoch in range(epoches):
                #i = epoch % num_videos
                i = epoch % num_videos
                i=i+1 #in ROLO utils file, the sequence for MOT16 starts with 30 Modification 3
                if i==2 or i==7:
                    continue
                [self.w_img, self.h_img, sequence_name, self.training_iters, dummy]= utils.choose_video_sequence(i)
        
                x_path = os.path.join('MOT_syntheticdata/train/', sequence_name, 'features/features.npy') #Modification 4
                x_bbox_path = os.path.join('MOT_syntheticdata/train/', sequence_name, 'bboxes/bboxes.npy')

                y_path = os.path.join('MOT_syntheticdata/train/', sequence_name, 'bboxes/gt.npy') ##Modification 5
                print y_path
                filegt = np.load(y_path)

                video_frame_features =np.load(x_path)
                video_frame_features = np.reshape(video_frame_features,newshape=(video_frame_features.shape[0],video_frame_features.shape[2]))

                scaler = MinMaxScaler(feature_range=(-1,1))
                datanew = scaler.fit_transform(video_frame_features)

                datanew = np.reshape(datanew,newshape=(datanew.shape[0],1,255))
                filefeatures = datanew                
                filebboxes = np.load(x_bbox_path)
                self.output_path = os.path.join('MOT_syntheticdata/train', sequence_name, 'rolo_out_train_1LSTM/')
                utils.createFolder(self.output_path)
                print sequence_name,x_path
                total_loss1 = 0
                total_loss2 = 0
                #id = 0
                random.seed(0)
                listrand = random.sample(xrange(0,self.training_iters- self.num_steps),self.training_iters- self.num_steps)
                self.step_size = 1
# Keep training until reach max iterations
                for id in listrand:
                # # Keep training until reach max iterations
                # while id  < self.training_iters- self.num_steps*self.batch_size:
                    # Load training data & ground truth
                    batch_input = self.rolo_utils.load_yolo3_output(self.w_img,self.h_img,filefeatures,filebboxes, self.batch_size, self.num_steps, id) # [num_of_examples, num_input] (depth == 1)

                    batch_groundtruth = self.rolo_utils.load_rolo_gt(self.w_img,self.h_img,filegt, self.batch_size, self.num_steps, self.step_size, id)
                    #print(batch_groundtruth)

                    batch_input = np.reshape(batch_input, [self.batch_size, self.num_steps, self.num_input])
                    batch_groundtruth = np.reshape(batch_groundtruth, [self.batch_size, self.num_obj2detect*self.num_gt]) #2*4

                    predict_location= sess.run(self.predict_location,feed_dict={self.input_x: batch_input, self.input_y: batch_groundtruth, self.initstate: np.zeros((self.batch_size, 2*self.num_input))})
                    #print("ROLO Pred in pixel: ", (predict_location[0][0]+1)*self.w_img, (predict_location[0][1]+1)*self.h_img, (predict_location[0][2]+1)*self.w_img, (predict_location[0][3]+1)*self.h_img)
                    #print("GT : ", (batch_groundtruth[0][0]+1)*self.w_img, (batch_groundtruth[0][1]+1)*self.h_img, (batch_groundtruth[0][2]+1)*self.w_img, (batch_groundtruth[0][3]+1)*self.h_img)

                    # Save pred_location to file
                    utils.save_rolo_output(self.output_path, predict_location, id, self.num_steps, self.batch_size)

                    sess.run(self.optimizer, feed_dict={self.input_x: batch_input, self.input_y: batch_groundtruth, self.initstate: np.zeros((self.batch_size, 2*self.num_input))})
                    if id % self.display_step == 0:
                        # Calculate batch loss
                        loss1 = sess.run(self.accuracy1, feed_dict={self.input_x: batch_input, self.input_y: batch_groundtruth, self.initstate: np.zeros((self.batch_size, 2*self.num_input))})
                        loss2 = sess.run(self.accuracy2, feed_dict={self.input_x: batch_input, self.input_y: batch_groundtruth, self.initstate: np.zeros((self.batch_size, 2*self.num_input))})
                    total_loss1 += loss1
                    total_loss2 += loss2
                    #id += 1

                #print "Optimization Finished!"
                avg_loss1 = total_loss1/id
                avg_loss2 = total_loss2/id

                print "Avg loss1: " + sequence_name + ": " + str(avg_loss1) + " " +str(avg_loss2)
                now = datetime.datetime.now()
                print "time:",now.isoformat()
                # count = count+avg_loss
                # if epoch%4==3:
                #     loss_list.append(count/4.0)
                #     count=0

                log_file.write("Avg loss: " + sequence_name + ": " + str("{:.5f}".format((avg_loss1+avg_loss2)/2)) + '  ')
                #if i+1==num_videos:
                    #log_file.write("time:" + + str("{:.3f}".format(now.isoformat())) + ' ')
                log_file.write('\n')

                save_path = self.saver.save(sess, self.rolo_weights_file)
                print("Model saved in file: %s" % save_path)
                print('\n')
            # plt.plot(loss_list)
        log_file.close()
        return


    def TrackerModel(self):

            self.rolo_utils= utils.ROLO_utils()
            self.training()

def main():
        ObjectTracking()

if __name__=='__main__':
        main()
