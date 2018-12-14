import OT_originalROLO_utils as utils
import tensorflow as tf
from tensorflow.contrib import rnn
import cv2
import numpy as np
import os.path
import time
import random
import datetime
from scipy import stats
from sklearn.preprocessing import MinMaxScaler

# import comet_ml in the top of your file
from comet_ml import Experiment

# Add the following code anywhere in your machine learning file
experiment = Experiment(api_key="tC4A7YA778cHBZYNZqwsXDpRx",project_name="general", workspace="pratikkulkarni228")



class ObjectTracking:
    disp_console = True
    restore_weights = True#True

    # Tracker Network Parameters
    tracker_weights_file = 'model_min1/model_step6.ckpt'  
    output_file = open("model_min1/1LSTM_testing_results.txt", "a")
    output_file1 = open("model_min1/OT-successplot_Originalrolo.txt", "a")
    prediction_file = open('model_min1/prediction.txt','a')
    num_steps = 6  # number of frames as an input sequence
    num_feat = 255
    num_obj2detect =1 
    num_predict = 4 # final output of LSTM 6 loc parameters
    num_gt = 4 # groundtruth for x,y,width and height
    num_input = (255+num_predict)*num_obj2detect # data input: 4096+6= 4102 (number of features obtained from YOLO + number of predicted locations)
    # Tracker Testing Parameters 
    batch_size = 1 #128
    display_step = 1
    num_hidden=259
    # tf Graph input
    input_x = tf.placeholder("float32", [None, num_steps, num_input]) #MODIFICATION
    initstate = tf.placeholder("float32", [None, 2*num_input]) #state & cell => 2x num_input #MODIFICATION
    input_y = tf.placeholder("float32", [None, num_gt*num_obj2detect]) #MODIFICATION

    # Define weights
    weights = {
        'out': tf.Variable(tf.random_normal([num_input, num_predict]))
    }
    biases = {
        'out': tf.Variable(tf.random_normal([num_predict]))
    }


    def __init__(self):
        print("TrackerModel init")
        self.TrackerModel()


    def LSTM_single(self, name,  input_x, initstate, weights, biases,reuse):
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

    '''---------------------------------------------------------------------------------------'''
    def build_LSTM_networks(self):
        print "Building Object Tracking graph..."

        # Build Tracker layer
        self.lstm_module = self.LSTM_single('lstm_train', self.input_x, self.initstate, self.weights, self.biases, False)
        self.sess = tf.Session()
        self.sess.run(tf.initialize_all_variables())
        self.saver = tf.train.Saver()
        print "Loading complete!" + '\n'


    def testing(self, filefeatures, filebboxes,filegt,sequence_name,w_img, h_img):
        total_loss = 0
        iou = np.zeros(9)
        print("TESTING TRACKER...")
        predict = self.LSTM_single('lstm_test', self.input_x, self.initstate, self.weights, self.biases,True)
        self.predict_location = predict[-1][:, 255:259]
        self.correct_prediction = tf.square(self.predict_location - self.input_y)
        #self.accuracy = tf.reduce_mean(self.correct_prediction) *100
        self.accuracy = tf.nn.l2_loss(self.predict_location[0][:2] - self.input_y[0][:2])+tf.nn.l2_loss(self.predict_location[0][2:] - self.input_y[0][2:])

        #self.correct_prediction = (tf.sqrt(self.accuracy))*100

        # Initializing the variables
        init = tf.initialize_all_variables()

        self.ious = []

        # Launch the graph
        with tf.Session() as sess:

            if (self.restore_weights == True):
                sess.run(init)
                self.saver.restore(sess, self.tracker_weights_file)
                print "Loading complete!" + '\n'
            else:
                sess.run(init)

            id = 0
            rolo_avgloss = 0
            predicted_location_list = []
            self.predicted_location = np.array([]) 

            self.step_size = 1

            # Keep training until reach max iterations
            while id < self.testing_iters - self.num_steps*self.batch_size:
                # Load training data & ground truth
                batch_input = self.rolo_utils.load_yolo3_output(self.w_img,self.h_img,filefeatures, filebboxes, self.batch_size, self.num_steps, self.step_size, id) # [num_of_examples, num_input] (depth == 1)

                batch_groundtruth = self.rolo_utils.load_rolo_gt(self.w_img,self.h_img,filegt, self.batch_size, self.num_steps, self.step_size, id)
                #print("batch_groundtruth_initial: ", batch_groundtruth)
                #batch_groundtruth = utils.locations_from_0_to_1(self.w_img, self.h_img, batch_groundtruth)

                # Reshape data to get 3 seq of 5002 elements
                batch_input = np.reshape(batch_input, [self.batch_size, self.num_steps, self.num_input])
                batch_groundtruth = np.reshape(batch_groundtruth, [self.batch_size, self.num_obj2detect*self.num_gt])
                #print("batch_groundtruth: ", batch_groundtruth)

                predict_location= sess.run(self.predict_location,feed_dict={self.input_x: batch_input, self.input_y: batch_groundtruth, self.initstate: np.zeros((self.batch_size, 2*self.num_input))})
                #print("Tracker Pred: ", predict_location)
                print("Tracker Pred in pixel: ", (predict_location[0][0]+1.0)*w_img, (predict_location[0][1]+1.0)*h_img, (predict_location[0][2]+1.0)*w_img, (predict_location[0][3]+1.0)*h_img)
                

                utils.save_rolo_output_test(self.output_path, predict_location, id, self.num_steps, self.batch_size)
                if id % self.display_step == 0:
                    # Calculate batch loss
                    loss = sess.run(self.accuracy, feed_dict={self.input_x: batch_input, self.input_y: batch_groundtruth, self.initstate: np.zeros((self.batch_size, 2*self.num_input))})
                    self.ious.append(loss)
                    #print "Iter " + str(id*self.batch_size) + ", Minibatch Loss= " + "{:.6f}".format(loss) #+ "{:.5f}".format(self.accuracy)
                    self.output_file.write("Avg loss: " + sequence_name + ": " + str("{:.5f}".format(loss)) + '  ')
                    #self.prediction_file.write('Video'+sequence_name+' '+str(int(predict_location[0][0]*640))+','+str(int(predict_location[0][1]*480))+','+str(int(predict_location[0][2]*640))+','+str(int(predict_location[0][3]*480))+'\n')
                    #if i+1==num_videos:
                    #log_file.write("time:" + + str("{:.3f}".format(now.isoformat())) + ' ')
                    midlist=[(predict_location[0][0]+1.0)*self.w_img,(predict_location[0][1]+1.0)*self.h_img,(predict_location[0][2]+1.0)*self.w_img,(predict_location[0][3]+1.0)*self.h_img]

                    predicted_location_list.append(midlist)
                    self.output_file.write('\n')                            

                    total_loss += loss
                id += 1

            print "Testing Finished!"
            avg_loss = total_loss/id
            print "Avg loss: " + sequence_name + ": " + str(avg_loss)
            now = datetime.datetime.now()
            print "time:",now.isoformat()

            predicted_array = np.asarray(predicted_location_list)
            np.save('model_min1/'+sequence_name+'prediction',predicted_array)    	    
        
        if sequence_name=='video11':
            self.output_file.close()
            self.prediction_file.close()

        return None

    def TrackerModel(self):

        self.rolo_utils= utils.ROLO_utils()
        now = datetime.datetime.now()
        print "time:",now.isoformat()
        print "Default: running Tracker test."
        self.build_LSTM_networks()
        num_videos = 7#modification
        for test in range(num_videos):
            test= test+50  #feeding 3 training videos itself
            [self.w_img, self.h_img, sequence_name, dummy_1, self.testing_iters] = utils.choose_video_sequence(test)
            x_path = os.path.join('MOT_syntheticdata/test/', sequence_name, 'features/features.npy') #Modification 4
            x_bbox_path = os.path.join('MOT_syntheticdata/test/', sequence_name, 'bboxes/bboxes.npy') #Modification 4

            y_path = os.path.join('MOT_syntheticdata/test/', sequence_name, 'bboxes/gt.npy')    #Modification 5
            filegt =np.load(y_path)

            video_frame_features =np.load(x_path)
            video_frame_features = np.reshape(video_frame_features,newshape=(video_frame_features.shape[0],video_frame_features.shape[2]))

            scaler = MinMaxScaler(feature_range=(-1,1))
            datanew = scaler.fit_transform(video_frame_features)
            datanew = np.reshape(datanew,newshape=(datanew.shape[0],1,255))
            filefeatures = datanew      
            filebboxes = np.load(x_bbox_path)
            #self.output_path = os.path.join('MOT_data/train', sequence_name, 'rolo_out_test_1LSTM/')
            self.output_path = os.path.join('MOT_syntheticdata/test', sequence_name, 'rolo_out_test_1LSTM/')
            utils.createFolder(self.output_path)
            #self.output_file.write(sequence_name + "\t")
            self.testing(filefeatures, filebboxes,filegt,sequence_name,self.w_img, self.h_img)

    '''----------------------------------------main-----------------------------------------------------'''
def main():
        ObjectTracking()

if __name__=='__main__':
        main()

