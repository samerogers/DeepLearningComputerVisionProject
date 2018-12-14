import os, sys, time
import numpy as np
import configparser as ConfigParser
import cv2
import pickle
import tensorflow as tf
import math
from scipy import stats 

global Config
import matplotlib.pyplot as plt


class ROLO_utils:
    cfgPath = ''
    Config = []
    flag_train = False
    flag_track = False
    flag_detect = False
    params = {
        'img_size': [448, 448],
        'num_classes': 20,
        'alpha': 0.1,
        'dropout_rate': 0.5,
        'num_feat_lstm': 5001,
        'num_steps': 28,
        'batch_size': 1,
        'tensor_size': 7,
        'predict_num_per_tensor': 2,
        'threshold': 0.2,
        'iou_threshold': 0.5,
        'class_labels': ["stop", "vehicle", "pedestrian"],

        'conv_layer_num': 3,
        'conv_filters': [16, 32, 64],
        'conv_size': [3, 3, 3],
        'conv_stride': [1, 1, 1],

        'fc_layer_num': 3,
        'fc_input': [1024, 1024, 1024],
        'fc_output': [1024, 1024, 1024]
    }
    file_weights = None
    file_in_path = None
    file_out_path = None
    flag_write = False
    flag_show_img = False

    batch_size = 128

    x_path = "u03/yolo_output/"
    y_path = "u03/rolo_gt"

    def __init__(self, argvs=[]):
        print("Utils init")

    # Network Parameters
    def loadCfg(self):
        Config = ConfigParser.ConfigParser()
        Config.read(self.cfgPath)
        Sections = Config.sections()

        print('self.cfgPath=' + self.cfgPath)
        if os.path.isfile(self.cfgPath):
            dict_parameters = self.ConfigSectionMap(Config, "Parameters")
            dict_networks = self.ConfigSectionMap(Config, "Networks")

            self.params['img_size'] = dict_parameters['img_size']  # [448, 448]
            self.params['alpha'] = dict_parameters['alpha']
            self.params['num_classes'] = dict_parameters['num_classes']  # 20
            self.params['dropout_rate'] = dict_parameters['dropout_rate']  # 0.5
            self.params['num_feat_lstm'] = dict_parameters[
                'num_feat_lstm']  # 4096+5 # number of features in hidden layer of LSTM
            self.params['num_steps'] = dict_parameters['num_steps']  # 28 # timesteps for LSTM
            self.params['batch_size'] = dict_parameters[
                'batch_size']  # 1 # during testing it is 1; during training it is 64.
            self.params['tensor_size'] = dict_parameters['tensor_size']
            self.params['predict_num_per_tensor'] = dict_parameters['predict_num_per_tensor']
            self.params['threshold'] = dict_parameters['tensor_size']
            self.params['iou_threshold'] = dict_parameters['iou_threshold']

            self.params['conv_layer_num'] = dict_networks['conv_layer_num']
            self.params['conv_filters'] = dict_networks['conv_filters']
            self.params['conv_size'] = dict_networks['conv_size']
            self.params['conv_stride'] = dict_networks['conv_stride']
            self.params['fc_layer_num'] = dict_networks['fc_layer_num']
            self.params['fc_input'] = dict_networks['fc_input']
            self.params['fc_output'] = dict_networks['fc_output']

        return self.params

    def ConfigSectionMap(self, Config, section):
        dict1 = {}
        options = Config.options(section)
        for option in options:
            dict1[option] = Config.get(section, option)
        return dict1

    def validate_file_format(self, file_in_path, allowed_format):
        if not os.path.isfile(file_in_path) or os.path.splitext(file_in_path)[1][1:] not in allowed_format:
            # print(os.path.splitext(file_in_path)[1][1:])
            print ("Input file with correct format not found.\n")
            return False
        else:
            return True

    def argv_parser(self, argvs):
        # global  file_weights, file_in_path, file_out_path, flag_write, flag_show_img
        allowed_format = ['png', 'jpg', 'JPEG', 'avi', 'mp4', 'mkv', 'cfg']
        for i in range(1, len(argvs), 2):
            if argvs[i] == '-train': self.flag_train = True
            if argvs[i] == '-cfg':  self.cfgPath = argvs[i + 1]
            if argvs[i] == '-weights': self.file_weights = argvs[i + 1]
            if argvs[i] == '-input':  self.file_in_path = argvs[i + 1];  self.validate_file_format(file_in_path,
                                                                                                   allowed_format)
            if argvs[i] == '-output': self.file_out_path = argvs[i + 1]; self.flag_write = True
            if argvs[i] == '-detect': self.flag_detect = True; self.flag_track = False;
            if argvs[i] == '-track': self.flag_detect = True; self.flag_track = True;
            if argvs[i] == '-imshow':
                if argvs[i + 1] == '1':
                    self.flag_show_img = True
                else:
                    self.flag_show_img = False
        return (self.cfgPath, self.file_weights, self.file_in_path)

    def is_image(self, file_in_path):
        if os.path.isfile(file_in_path) and os.path.splitext(file_in_path)[1][1:] in ['jpg', 'JPEG', 'png', 'JPG']:
            return True
        else:
            return False

    def is_video(self, file_in_path):
        if os.path.isfile(file_in_path) and os.path.splitext(file_in_path)[1][1:] in ['avi', 'mkv', 'mp4']:
            return True
        else:
            return False

    # Not Face user
    def file_to_img(self, filepath):
        print ('Processing ' + filepath)
        img = cv2.imread(filepath)
        return img

    def file_to_video(self, filepath):
        print ('processing ' + filepath)
        try:
            video = cv2.VideoCapture(filepath)
        except IOError:
            print ('cannot open video file: ' + filepath)
        else:
            print ('unknown error reading video file')
        return video

    def iou(self, bbox1, box2):
        # Prevent NaN in benchmark results
        #validate_box(box1)
        #validate_box(box2)
        x = np.zeros([1,4])
        sess = tf.Session()
        feed_dict={x: bbox1}
        box1 = sess.run([x], feed_dict=feed_dict)
        box1[2]=box1[2]-box1[0]
        box2[2]=box2[2]-box2[0]
        box1[3]=box1[3]-box1[1]
        box2[3]=box2[3]-box2[1]
        # print("pred",box1)
        # print("gt",box2)
        # change float to int, in order to prevent overflow
        box1 = map(float, box1)
        box2 = map(float, box2)
        tb = max(min(box1[0] + 0.5 * box1[2], box2[0] + 0.5 * box2[2]) - max(box1[0] - 0.5 * box1[2], box2[0] - 0.5 * box2[2]),0)
        lr = max(min(box1[1] + 0.5 * box1[3], box2[1] + 0.5 * box2[3]) - max(box1[1] - 0.5 * box1[3], box2[1] - 0.5 * box2[3]),0)
        # tb = min(box1[0]+box1[2],box2[0]+box2[2])-max(box1[0],box2[0])
        # lr = min(box1[1]+box1[3],box2[1]+box2[3])-max(box1[1],box2[1])
        # print tb
        # print lr

        # if tb <= 0 or lr <= 0:
        #     intersection = 0
        #     print ("intersection= 0")
        # else:
        intersection = tb * lr
        return intersection / (box1[2] * box1[3] + box2[2] * box2[3] - intersection)
    # def iou(box1, box2):
    # # Prevent NaN in benchmark results
    # validate_box(box1)
    # validate_box(box2)
    # box1[2]=box1[2]-box1[0]
    # box2[2]=box2[2]-box2[0]
    # box1[3]=box1[3]-box1[1]
    # box2[3]=box2[3]-box2[1]
    # # print("pred",box1)
    # # print("gt",box2)
    # # change float to int, in order to prevent overflow
    # box1 = map(float, box1)
    # box2 = map(float, box2)
    # tb = max(tf.math.min(box1[0] + 0.5 * box1[2], box2[0] + 0.5 * box2[2]) - tf.math.max(box1[0] - 0.5 * box1[2], box2[0] - 0.5 * box2[2]),0)
    # lr = tf.math.max(tf.math.min(box1[1] + 0.5 * box1[3], box2[1] + 0.5 * box2[3]) - tf.math.max(box1[1] - 0.5 * box1[3], box2[1] - 0.5 * box2[3]),0)
    # # tb = min(box1[0]+box1[2],box2[0]+box2[2])-max(box1[0],box2[0])
    # # lr = min(box1[1]+box1[3],box2[1]+box2[3])-max(box1[1],box2[1])
    # # print tb
    # # print lr

    # # if tb <= 0 or lr <= 0:
    # #     intersection = 0
    # #     print ("intersection= 0")
    # # else:
    # intersection = tb * lr
    # return intersection / (box1[2] * box1[3] + box2[2] * box2[3] - intersection)

    def find_iou_cost(self, pred_locs, gts):
        # for each element in the batch, find its iou. output a list of ious.
        cost = 0
        # batch_size= len(pred_locs)
        batch_size = self.batch_size
        # assert (len(gts)== batch_size)
        # print("gts: ", gts)
        # print("batch_size: ", batch_size)
        # print("pred_locs: ", pred_locs)
        # ious = []
        ious = np.zeros((batch_size, 4))

        for i in range(batch_size):
            pred_loc = pred_locs[i, :]
            # print("pred_loc[i]: ", pred_loc)
            gt = gts[i, :]
            iou_ = self.iou(pred_loc, gt)
            # ious.append(iou_)
            # print("iou_", iou_)
            ious[i, :] = iou_
        # ious= tf.reshape(ious, batch_size)
        # print("ious: ", ious)
        '''
        avg_iou= 0
        for i in range(batch_size):
                pred_loc = pred_locs[i,:]
                gt= gts[i,:]
                #print("gt", gt)
                #print("pred_loc", pred_loc)
                avg_iou += self.iou(pred_loc, gt)
        avg_iou /= batch_size

        print("avg_iou shape: ", tf.shape(avg_iou)) # single tensor expected
        return avg_iou'''
        return ious

    def load_folder(self, path):
	print(next(os.walk(path))[2])
        paths = [os.path.join(path, fn) for fn in next(os.walk(path))[2]]
	print(paths)
        return paths

    def load_dataset_gt(self, gt_file):
        txtfile = open(gt_file, "r")
        lines = txtfile.read().split('\n')  # '\r\n'
        return lines

    def find_gt_location(self, lines, id):   
        # print("lines length: ", len(lines))
        # print("id: ", id)
        line = lines[id]
        # print "line", line
        elems = line.split('\t')  # for gt type 2
        # print(elems)
        if len(elems) < 4:
            elems = line.split(',')  # for gt type 1
            # print(elems)
        x1 = elems[1] #changed
        y1 = elems[2]
        w1 = elems[3]
        h1 = elems[4]
        x2 = elems[6] #changed
        y2 = elems[7]
        w2 = elems[8]
        h2 = elems[9]
        gt_location = [float(x1), float(y1), float(w1), float(h1),float(x2), float(y2), float(w2), float(h2)]
        return gt_location

    def find_best_location(self, locations, gt_location):
        # locations (class, x, y, w, h, prob); (x, y) is the middle pt of the rect
        # gt_location (x1, y1, w, h)
        x1 = gt_location[0]
        y1 = gt_location[1]
        w = gt_location[2]
        h = gt_location[3]
        gt_location_revised = [x1 + w / 2, y1 + h / 2, w, h]

        max_ious = 0
        for location, id in enumerate(locations):
            location_revised = location[1:5]
            ious = self.iou(location_revised, gt_location_revised)
            if ious >= max_ious:
                max_ious = ious
                index = id
        return locations[index]

    def save_yolo_output(self, out_fold, yolo_output, filename):
        name_no_ext = os.path.splitext(filename)[0]
        output_name = name_no_ext + ".yolo"
        path = os.path.join(out_fold, output_name)
        pickle.dump(yolo_output, open(path, "rb"))

    def load_yolo_output(self, fold, batch_size, num_steps, step):
        paths = [os.path.join(fold, fn) for fn in next(os.walk(fold))[2]]
        paths = sorted(paths)
        st = step * batch_size * num_steps
        ed = (step + 1) * batch_size * num_steps
        paths_batch = paths[st:ed]

        yolo_output_batch = []
        ct = 0
        for path in paths_batch:
            ct += 1
            # yolo_output= pickle.load(open(path, "rb"))
            yolo_output = np.load(path)
            # print(yolo_output.shape)
            yolo_output = np.reshape(yolo_output, 4102)
            yolo_output[4096] = 0
            yolo_output[4101] = 0
            yolo_output_batch.append(yolo_output)
        print(yolo_output_batch.shape)
        yolo_output_batch = np.reshape(yolo_output_batch, [batch_size * num_steps, 4102])
        return yolo_output_batch
    '''
        # def load_rolo_gt(self, path, batch_size, num_steps, step):
        #     lines = self.load_dataset_gt(path)
        #     offset = num_steps - 2  # offset is for prediction of the future
        #     st = step * batch_size * num_steps
        #     ed = (step + 1) * batch_size * num_steps
        #     # print("st: " + str(st))
        #     # print("ed: " + str(ed))
        #     batch_locations = []
        #     for id in range(st + offset, ed + offset, num_steps):
        #         location = self.find_gt_location(lines, id)
        #         batch_locations.append(location)
        #     return batch_locations
    '''

    def load_rolo_gt(self,w_img,h_img,arr,batch_size,num_steps,id):

        video_frame_bboxes = np.asarray(arr,dtype=np.float64)
        num_obj=1
        wid = w_img
        ht = h_img
        yolo_output_batched=[]
        
        #frame_num= id+num_steps
        #if frame_num>772:
        #    return [0,0,0,0]
        for frame_num in range(id+num_steps,id+1+(num_steps*batch_size),num_steps):
            yolo_output_temp=[]
            for obj in range(0,num_obj):
                video_frame_bboxes[frame_num][obj][2] += video_frame_bboxes[frame_num][obj][0] # convert w,h to bottom right 
                video_frame_bboxes[frame_num][obj][3] += video_frame_bboxes[frame_num][obj][1]

                video_frame_bboxes[frame_num][obj][0] = (video_frame_bboxes[frame_num][obj][0]/wid)-1
                video_frame_bboxes[frame_num][obj][1] = (video_frame_bboxes[frame_num][obj][1]/ht)-1
                video_frame_bboxes[frame_num][obj][2] = (video_frame_bboxes[frame_num][obj][2]/wid)-1
                video_frame_bboxes[frame_num][obj][3] = (video_frame_bboxes[frame_num][obj][3]/ht)-1


                yoloBB = video_frame_bboxes[frame_num][obj].tolist() #framenum, obj1,4 bbox attributes
                yolo_output_temp = yolo_output_temp + yoloBB
            yolo_output_batched = yolo_output_batched + yolo_output_temp
        #yolo_output_batched = np.reshape(yolo_output_batched, [batch_size*num_steps,4])
        return yolo_output_batched

    def load_yolo_output_for_coarse_input(self, fold, batch_size, num_steps, id, coarse_next_prediction):
        paths = [os.path.join(fold, fn) for fn in next(os.walk(fold))[2]]
        paths = sorted(paths)
        st = step * batch_size * num_steps
        ed = (step + 1) * batch_size * num_steps
        paths_batch = paths[st:ed]

        yolo_output_batch = []
        ct = 0
        for path in paths_batch:
            ct += 1
            yolo_output = np.load(path)
            # print(yolo_output)
            yolo_output = np.reshape(yolo_output, 4102)
            yolo_output[4096] = 0
        yolo_output[4101] = 0
        # print(yolo_output[4096:4102])
        yolo_output_batch.append(yolo_output)

        yolo_output = np.load(paths[ed + 1])
        yolo_output = np.reshape(yolo_output, 4102)
        # np.lib.pad(coarse_next_prediction, (2,0), 'constant', constant_values=(0,0))
        # print "coarse_next_prediction,", coarse_next_prediction
        coarse_next_prediction = np.concatenate([np.zeros(2), coarse_next_prediction[0]])
        # print "coarse_next_prediction,", coarse_next_prediction
        yolo_output = np.append(yolo_output[:-6], coarse_next_prediction)
        yolo_output_batch.append(yolo_output)
        yolo_output_batch = np.reshape(yolo_output_batch, [batch_size * num_steps, 4102])
        return yolo_output_batch


    def load_yolo_output_test_for_coarse_input(self, fold, batch_size, num_steps, id, coarse_next_prediction):
        paths = [os.path.join(fold, fn) for fn in next(os.walk(fold))[2]]
        paths = sorted(paths)
        st = id + 1
        ed = id + batch_size * num_steps
        paths_batch = paths[st:ed]

        yolo_output_batch = []
        ct = 0
        for path in paths_batch:
            ct += 1
            yolo_output = np.load(path)
            # print(yolo_output)
            yolo_output = np.reshape(yolo_output, 4102)
            # print(yolo_output[4096:4102])
            yolo_output_batch.append(yolo_output)
        yolo_output = np.load(paths[ed + 1])
        yolo_output = np.reshape(yolo_output, 4102)
        # np.lib.pad(coarse_next_prediction, (2,0), 'constant', constant_values=(0,0))
        # print "coarse_next_prediction,", coarse_next_prediction
        coarse_next_prediction = np.concatenate([np.zeros(2), coarse_next_prediction[0]])
        # print "coarse_next_prediction,", coarse_next_prediction
        yolo_output = np.append(yolo_output[:-6], coarse_next_prediction)
        yolo_output_batch.append(yolo_output)
        yolo_output_batch = np.reshape(yolo_output_batch, [batch_size * num_steps, 4102])
        return yolo_output_batch


    def load_yolo_output_test(self, fold, batch_size, num_steps, id):
        paths = [os.path.join(fold, fn) for fn in next(os.walk(fold))[2]]
        paths = sorted(paths)
        st = id
        ed = id + batch_size * num_steps
        paths_batch = paths[st:ed]

        yolo_output_batch = []
        ct = 0
        for path in paths_batch:
            ct += 1
            yolo_output = np.load(path)
            # print(yolo_output)
            #yolo_output = np.reshape(yolo_output[:,:4120], 4120)#modification to slice and extract features only upto 4120
            yolo_output_batch.append(yolo_output) #append 6 frame features +bboxes
        yolo_output_batch = np.reshape(yolo_output_batch, [batch_size * num_steps, 4108])#modification 4102 --> 4120
        return yolo_output_batch

    def load_yolo3_output(self,w_img,h_img,foldf,foldb,batch_size, num_steps, step_size, id):    
        video_frame_features = foldf
        video_frame_bboxes = foldb
        wid = w_img
        ht = h_img

        num_obj=1



        lastseen_feature=0
        yolo_output_batched=[]
        for frame_num in range(id,id+num_steps*batch_size, step_size):  # if id=2 gather frames from 2 to 7
            yolo_output_temp=[]
            for obj in range(0,num_obj):
                if  video_frame_features[frame_num][obj].all()!=0:
                    lastseen_feature=frame_num
                if  video_frame_features[frame_num][obj].all()==0:
                    video_frame_features[frame_num][obj]=video_frame_features[lastseen_feature][obj]
                    lastseen_feature=framenum
                yoloFeat = video_frame_features[frame_num][obj].tolist()  #framenum,obj1,255 features
                yolo_output_temp = yolo_output_temp + yoloFeat

                video_frame_bboxes[frame_num][obj][2] += video_frame_bboxes[frame_num][obj][0] # convert w,h to bottom right 
                video_frame_bboxes[frame_num][obj][3] += video_frame_bboxes[frame_num][obj][1]

                video_frame_bboxes[frame_num][obj][0] = (video_frame_bboxes[frame_num][obj][0]/wid)-1
                video_frame_bboxes[frame_num][obj][1] = (video_frame_bboxes[frame_num][obj][1]/ht)-1
                video_frame_bboxes[frame_num][obj][2] = (video_frame_bboxes[frame_num][obj][2]/wid)-1
                video_frame_bboxes[frame_num][obj][3] = (video_frame_bboxes[frame_num][obj][3]/ht)-1


                yoloBB = video_frame_bboxes[frame_num][obj].tolist() #framenum, obj1,4 bbox attributes
                yolo_output_temp = yolo_output_temp + yoloBB
            yolo_output_batched = yolo_output_batched + yolo_output_temp

        yolo_output_batched = np.reshape(yolo_output_batched, [batch_size*num_steps,259])
        
        return yolo_output_batched

    def load_rolo_gt_test(self, path, batch_size, num_steps, id):
        lines = self.load_dataset_gt(path)
        offset = num_steps - 2  # offset is for prediction of the future
        st = id
        ed = id + batch_size * num_steps
        batch_locations = []
        for id in range(st + offset, ed + offset, num_steps):
            location = self.find_gt_location(lines, id)
            batch_locations.append(location)
        return batch_locations

    def load_rolo_gt(self,w_img,h_img,arr,batch_size,num_steps,id):

        video_frame_bboxes = np.asarray(arr,dtype=np.float64)
        num_obj=1
        wid = w_img
        ht = h_img
        yolo_output_batched=[]
        
        #frame_num= id+num_steps
        #if frame_num>772:
        #    return [0,0,0,0]
        for frame_num in range(id+num_steps,id+1+(num_steps*batch_size),num_steps):
            yolo_output_temp=[]
            for obj in range(0,num_obj):
                video_frame_bboxes[frame_num][obj][2] += video_frame_bboxes[frame_num][obj][0] # convert w,h to bottom right 
                video_frame_bboxes[frame_num][obj][3] += video_frame_bboxes[frame_num][obj][1]
                #print video_frame_bboxes[frame_num][obj], wid, ht

                video_frame_bboxes[frame_num][obj][0] = (video_frame_bboxes[frame_num][obj][0]/wid)-1
                video_frame_bboxes[frame_num][obj][1] = (video_frame_bboxes[frame_num][obj][1]/ht)-1
                video_frame_bboxes[frame_num][obj][2] = (video_frame_bboxes[frame_num][obj][2]/wid)-1
                video_frame_bboxes[frame_num][obj][3] = (video_frame_bboxes[frame_num][obj][3]/ht)-1


                #print video_frame_bboxes[frame_num][obj]
                yoloBB = video_frame_bboxes[frame_num][obj].tolist() #framenum, obj1,4 bbox attributes
                yolo_output_temp = yolo_output_temp + yoloBB
            yolo_output_batched = yolo_output_batched + yolo_output_temp
        #yolo_output_batched = np.reshape(yolo_output_batched, [batch_size*num_steps,4])
        return yolo_output_batched

    def load_rolo_gt(self,w_img,h_img,arr,batch_size,num_steps,step_size,id):

        video_frame_bboxes = np.asarray(arr,dtype=np.float64)
        num_obj=1
        wid = w_img
        ht = h_img
        yolo_output_batched=[]
        
        #frame_num= id+num_steps
        #if frame_num>772:
        #    return [0,0,0,0]
        #for frame_num in range(id+num_steps,id+1+(num_steps*batch_size),num_steps):
        frame_num = id+num_steps*step_size
        yolo_output_temp=[]
        for obj in range(0,num_obj):
            video_frame_bboxes[frame_num][obj][2] += video_frame_bboxes[frame_num][obj][0] # convert w,h to bottom right 
            video_frame_bboxes[frame_num][obj][3] += video_frame_bboxes[frame_num][obj][1]
            #print video_frame_bboxes[frame_num][obj], wid, ht

            video_frame_bboxes[frame_num][obj][0] = (video_frame_bboxes[frame_num][obj][0]/wid)-1
            video_frame_bboxes[frame_num][obj][1] = (video_frame_bboxes[frame_num][obj][1]/ht)-1
            video_frame_bboxes[frame_num][obj][2] = (video_frame_bboxes[frame_num][obj][2]/wid)-1
            video_frame_bboxes[frame_num][obj][3] = (video_frame_bboxes[frame_num][obj][3]/ht)-1


            #print video_frame_bboxes[frame_num][obj]
            yoloBB = video_frame_bboxes[frame_num][obj].tolist() #framenum, obj1,4 bbox attributes
            yolo_output_temp = yolo_output_temp + yoloBB
        yolo_output_batched = yolo_output_batched + yolo_output_temp
        #yolo_output_batched = np.reshape(yolo_output_batched, [batch_size*num_steps,4])
        return yolo_output_batched

    def load_lstm1_output(self, fold, batch_size, num_steps, predicted_location, id):
        paths = [os.path.join(fold, fn) for fn in next(os.walk(fold))[2]]
        paths = sorted(paths)
        ed = id + batch_size * num_steps + 1
        st = id + 1
        paths_batch = paths[st:ed]
        # print id, st,ed
        lstm_output_batch = []
        ct = 0
        for path in paths_batch:
            yolo_output = np.load(path)
            # print(yolo_output)
            yolo_output = np.reshape(yolo_output, 4102)
            # print(predicted_location[0,:])
            # print(yolo_output[:-6])
            # print(predicted_location)
            lstm_output = np.append(yolo_output[:-6], predicted_location[ct:ct + 4])
            # print("lstm_output", lstm_output)
            lstm_output_batch.append(lstm_output)
            ct = ct + 4
        # print lstm_output_batch
        yolo_output_batch = np.reshape(lstm_output_batch, [batch_size * num_steps, 4100])
        return yolo_output_batch


def load_lstm2_output(self, fold, batch_size, predicted_location, id):
    paths = [os.path.join(fold, fn) for fn in next(os.walk(fold))[2]]
    paths = sorted(paths)
    ed = id + 2
    st = id + 1
    paths_batch = paths[st:ed]
    # print id, st,ed
    lstm_output_batch = []
    ct = 0
    for path in paths_batch:
        yolo_output = np.load(path)
        yolo_output = np.reshape(yolo_output, 4102)
        # print(predicted_location[0,:])
        # print(yolo_output[:-6])
        # print(predicted_location)
        lstm_output = np.append(yolo_output[:-6], predicted_location[ct:ct + 4])
        # print("lstm_output", lstm_output)
        lstm_output_batch.append(lstm_output)
        ct = ct + 4
        # print lstm_output_batch
    yolo_output_batch = np.reshape(lstm_output_batch, [1, 4100])
    return yolo_output_batch


# -----------------------------------------------------------------------------------------------

def createFolder(path):
    if not os.path.exists(path):
        os.makedirs(path)


def load_folder(path):
    paths = [os.path.join(path, fn) for fn in next(os.walk(path))[2]]
    return sorted(paths)


def load_dataset_gt(gt_file):
    txtfile = open(gt_file, "r")
    lines = txtfile.read().split('\n')  # '\r\n'
    return lines


def find_gt_location(lines, id):
    line = lines[id]
    elems = line.split('\t')  # for gt type 2
    if len(elems) < 4:
        elems = line.split(',')  # for gt type 1
    x1 = elems[0]
    y1 = elems[1]
    w = elems[2]
    h = elems[3]
    gt_location = [float(x1), float(y1), float(w), float(h)]
    return gt_location


def find_yolo_location(fold, id):
    paths = [os.path.join(fold, fn) for fn in next(os.walk(fold))[2]]
    paths = sorted(paths)
    path = paths[id - 1]
    # print(path)
    yolo_output = np.load(path)
    # print(yolo_output[0][4096:4102])
    yolo_location = yolo_output[0][4097:4101]
    return yolo_location


import re


def tryint(s):
    try:
        return int(s)
    except:
        return s


def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [tryint(c) for c in re.split('([0-9]+)', s)]


def sort_nicely(l):
    """ Sort the given list in the way that humans expect.
    """
    l.sort(key=alphanum_key)


def find_rolo_location(fold, id):
    filename = str(id) + '.npy'
    path = os.path.join(fold, filename)
    rolo_output = np.load(path)
    # print rolo_output
    return rolo_output


def file_to_img(filepath):
    img = cv2.imread(filepath)
    return img


def debug_location(img, location):
    img_cp = img.copy()
    x = int(location[1])
    y = int(location[2])
    w = int(location[3]) // 2
    h = int(location[4]) // 2
    cv2.rectangle(img_cp, (x - w, y - h), (x + w, y + h), (0, 255, 0), 2)
    cv2.rectangle(img_cp, (x - w, y - h - 20), (x + w, y - h), (125, 125, 125), -1)
    cv2.putText(img_cp, str(location[0]) + ' : %.2f' % location[5], (x - w + 5, y - h - 7), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 0, 0), 1)
    cv2.imshow('YOLO_small detection', img_cp)
    cv2.waitKey(1)


def debug_gt_location(img, location):
    img_cp = img.copy()
    x = int(location[0])
    y = int(location[1])
    w = int(location[2])
    h = int(location[3])
    cv2.rectangle(img_cp, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imshow('gt', img_cp)
    cv2.waitKey(1)


def debug_3_locations(img, gt_location, yolo_location, rolo_location):
    img_cp = img.copy()
    for i in range(3):  # b-g-r channels
        if i == 0:
            location = gt_location; color = (0, 0, 255)  # red for gt
        elif i == 1:
            location = yolo_location; color = (255, 0, 0)  # blur for yolo
        elif i == 2:
            location = rolo_location; color = (0, 255, 0)  # green for rolo
        x = int(location[0])
        y = int(location[1])
        w = int(location[2])
        h = int(location[3])
        if i == 1 or i == 2:
            cv2.rectangle(img_cp, (x - w // 2, y - h // 2), (x + w // 2, y + h // 2), color, 2)
        elif i == 0:
            cv2.rectangle(img_cp, (x, y), (x + w, y + h), color, 2)
    # cv2.imshow('3 locations',img_cp)
    # cv2.waitKey(100)
    return img_cp


def debug_2_locations(img, gt_location, yolo_location):
    img_cp = img.copy()
    for i in range(3):  # b-g-r channels
        if i == 0:
            location = gt_location; color = (0, 0, 255)  # red for gt
        elif i == 1:
            location = yolo_location; color = (255, 0, 0)  # blur for yolo
        x = int(location[0])
        y = int(location[1])
        w = int(location[2])
        h = int(location[3])
        if i == 1:
            cv2.rectangle(img_cp, (x - w // 2, y - h // 2), (x + w // 2, y + h // 2), color, 2)
        elif i == 0:
            cv2.rectangle(img_cp, (x, y), (x + w, y + h), color, 2)
    cv2.imshow('2 locations', img_cp)
    cv2.waitKey(100)
    return img_cp


def save_rolo_output(out_fold, rolo_output, filename):
    name_no_ext = os.path.splitext(filename)[0]
    output_name = name_no_ext
    path = os.path.join(out_fold, output_name)
    np.save(path, rolo_output)


def save_rolo_output(out_fold, rolo_output, step, num_steps, batch_size):
    #assert (len(rolo_output) == batch_size)
    st = step * batch_size * num_steps - 2
    for i in range(batch_size):
        id = st + (i + 1) * num_steps + 1
        pred = rolo_output[i]
        path = os.path.join(out_fold, str(id))
        np.save(path, pred)


def save_rolo_output_test(out_fold, rolo_output, step, num_steps, batch_size):
    # print("rolo_output",rolo_output)
    assert (len(rolo_output) == batch_size)
    # assert(len(coarse_rolo_output)== batch_size)
    st = step - 2  # * batch_size * num_steps
    for i in range(batch_size):
        id = st + (i + 1) * num_steps + 1
        pred = rolo_output[i]
        # print "pred", pred
        # print "coarse", coarse_pred
        # print "pred and coarse_pred: ",pred, coarse_rolo_output[i]
        path = os.path.join(out_fold, str(id))
        np.save(path, pred)


def locations_normal(wid, ht, locations):
    # print("location in func: ", locations)
    wid *= 1.0
    ht *= 1.0
    locations[0] *= wid
    locations[1] *= ht
    locations[2] *= wid
    locations[3] *= ht
    return locations


def locations_from_0_to_1(wid, ht, locations):
    # print("location in func: ", locations[0][0])
    wid *= 1.0
    ht *= 1.0
    #for i in range(len(locations)):
        # convert top-left point (x,y) to mid point (x, y)
        #locations[i][0] += locations[i][2] / 2.0
        #locations[i][1] += locations[i][3] / 2.0
        # convert to [0, 1]
    locations[0] /= wid
    locations[1] /= ht
    locations[2] /= wid
    locations[3] /= ht
    return locations


def validate_box(box):
    for i in range(len(box)):
        if math.isnan(box[i]): box[i] = 0


def iou(box1, box2):
    # Prevent NaN in benchmark results
    validate_box(box1)
    validate_box(box2)
    box1[2]=box1[2]-box1[0]
    box2[2]=box2[2]-box2[0]
    box1[3]=box1[3]-box1[1]
    box2[3]=box2[3]-box2[1]
    # print("pred",box1)
    # print("gt",box2)
    # change float to int, in order to prevent overflow
    box1 = map(float, box1)
    box2 = map(float, box2)
    tb = tf.math.max(tf.math.min(box1[0] + 0.5 * box1[2], box2[0] + 0.5 * box2[2]) - tf.math.max(box1[0] - 0.5 * box1[2], box2[0] - 0.5 * box2[2]),0)
    lr = tf.math.max(tf.math.min(box1[1] + 0.5 * box1[3], box2[1] + 0.5 * box2[3]) - tf.math.max(box1[1] - 0.5 * box1[3], box2[1] - 0.5 * box2[3]),0)
    # tb = min(box1[0]+box1[2],box2[0]+box2[2])-max(box1[0],box2[0])
    # lr = min(box1[1]+box1[3],box2[1]+box2[3])-max(box1[1],box2[1])
    # print tb
    # print lr

    # if tb <= 0 or lr <= 0:
    #     intersection = 0
    #     print ("intersection= 0")
    # else:
    intersection = tb * lr
    return intersection / (box1[2] * box1[3] + box2[2] * box2[3] - intersection)


def iou_0_1(box1, box2, w, h):
    box1 = locations_normal(w, h, box1)
    box2 = locations_normal(w, h, box2)
    # print box1
    # print box2
    return iou(box1, box2)


def cal_rolo_IOU(location, gt_location):
    location[0] = location[0] - location[2] / 2
    location[1] = location[1] - location[3] / 2
    # print "location", location
    # print "gt_location", gt_location
    loss = iou(location, gt_location)
    return loss


def cal_yolo_IOU(location, gt_location):
    # Translate yolo's box mid-point (x0, y0) to top-left point (x1, y1), in order to compare with gt
    location[0] = location[0] - location[2] / 2
    location[1] = location[1] - location[3] / 2
    loss = iou(location, gt_location)
    return loss


def cal_benchmark_IOU(location, gt_location):
    loss = iou(location, gt_location)
    return loss


def cal_rolo_score(location, gt_location, thresh):
    rolo_iou = cal_rolo_IOU(location, gt_location)
    if rolo_iou >= thresh:
        score = 1
    else:
        score = 0
    return score


def load_yolo_output_test(fold, batch_size, num_steps, id):
    paths = [os.path.join(fold, fn) for fn in next(os.walk(fold))[2]]
    paths = sorted(paths)
    st = id
    ed = id + batch_size * num_steps
    paths_batch = paths[st:ed]

    yolo_output_batch = []
    ct = 0
    for path in paths_batch:
        ct += 1
        yolo_output = np.load(path)
        # print(yolo_output)
        yolo_output = np.reshape(yolo_output, 4108)
        yolo_output_batch.append(yolo_output)
    yolo_output_batch = np.reshape(yolo_output_batch, [batch_size * num_steps, 4108])
    return yolo_output_batch

# def load_yolo3_output(w_img,h_img,foldf,foldb,batch_size, num_steps, id):    
#     video_frame_features = foldf
#     video_frame_bboxes = foldb
#     wid = w_img
#     ht = h_img

#     num_obj=1



#     lastseen_feature=0
#     yolo_output_batched=[]
#     for frame_num in range(id,id+num_steps*batch_size):  # if id=2 gather frames from 2 to 7
#         yolo_output_temp=[]
#         for obj in range(0,num_obj):
#             if  video_frame_features[frame_num][obj].all()!=0:
#                 lastseen_feature=frame_num
#             if  video_frame_features[frame_num][obj].all()==0:
#                 video_frame_features[frame_num][obj]=video_frame_features[lastseen_feature][obj]
#                 lastseen_feature=framenum
#             yoloFeat = video_frame_features[frame_num][obj].tolist()  #framenum,obj1,255 features
#             yolo_output_temp = yolo_output_temp + yoloFeat

#             video_frame_bboxes[frame_num][obj][2] += video_frame_bboxes[frame_num][obj][0] # convert w,h to bottom right 
#             video_frame_bboxes[frame_num][obj][3] += video_frame_bboxes[frame_num][obj][1]

#             video_frame_bboxes[frame_num][obj][0] = (video_frame_bboxes[frame_num][obj][0]/wid)-1
#             video_frame_bboxes[frame_num][obj][1] = (video_frame_bboxes[frame_num][obj][1]/ht)-1
#             video_frame_bboxes[frame_num][obj][2] = (video_frame_bboxes[frame_num][obj][2]/wid)-1
#             video_frame_bboxes[frame_num][obj][3] = (video_frame_bboxes[frame_num][obj][3]/ht)-1


#             yoloBB = video_frame_bboxes[frame_num][obj].tolist() #framenum, obj1,4 bbox attributes
#             yolo_output_temp = yolo_output_temp + yoloBB
#         yolo_output_batched = yolo_output_batched + yolo_output_temp

#     yolo_output_batched = np.reshape(yolo_output_batched, [batch_size*num_steps,259])
    
#     return yolo_output_batched

def load_yolo3_output(w_img,h_img,foldf,foldb,batch_size, num_steps, step_size, id):    
    video_frame_features = foldf
    video_frame_bboxes = foldb
    wid = w_img
    ht = h_img

    num_obj=1



    lastseen_feature=0
    yolo_output_batched=[]
    for frame_num in range(id,id+num_steps*batch_size, step_size):  # if id=2 gather frames from 2 to 7
        yolo_output_temp=[]
        for obj in range(0,num_obj):
            if  video_frame_features[frame_num][obj].all()!=0:
                lastseen_feature=frame_num
            if  video_frame_features[frame_num][obj].all()==0:
                video_frame_features[frame_num][obj]=video_frame_features[lastseen_feature][obj]
                lastseen_feature=framenum
            yoloFeat = video_frame_features[frame_num][obj].tolist()  #framenum,obj1,255 features
            yolo_output_temp = yolo_output_temp + yoloFeat

            video_frame_bboxes[frame_num][obj][2] += video_frame_bboxes[frame_num][obj][0] # convert w,h to bottom right 
            video_frame_bboxes[frame_num][obj][3] += video_frame_bboxes[frame_num][obj][1]

            video_frame_bboxes[frame_num][obj][0] = (video_frame_bboxes[frame_num][obj][0]/wid)-1
            video_frame_bboxes[frame_num][obj][1] = (video_frame_bboxes[frame_num][obj][1]/ht)-1
            video_frame_bboxes[frame_num][obj][2] = (video_frame_bboxes[frame_num][obj][2]/wid)-1
            video_frame_bboxes[frame_num][obj][3] = (video_frame_bboxes[frame_num][obj][3]/ht)-1


            yoloBB = video_frame_bboxes[frame_num][obj].tolist() #framenum, obj1,4 bbox attributes
            yolo_output_temp = yolo_output_temp + yoloBB
        yolo_output_batched = yolo_output_batched + yolo_output_temp

    yolo_output_batched = np.reshape(yolo_output_batched, [batch_size*num_steps,259])
    
    return yolo_output_batched

def load_lstm1_output(fold, batch_size, num_steps, predicted_location, id):
    paths = [os.path.join(fold, fn) for fn in next(os.walk(fold))[2]]
    paths = sorted(paths)
    st = id
    ed = id + batch_size * num_steps
    paths_batch = paths[st:ed]

    yolo_output_batch = []
    ct = 0
    for path in paths_batch:
        ct += 1
        yolo_output = np.load(path)
        # print(yolo_output)
        yolo_output = np.reshape(yolo_output, 4102)
        lstm_output = np.concat(yolo_output[:, :-6], predicted_location, axis=1)
        print("lstm_output", lstm_output)
        yolo_output_batch.append(yolo_output)
    yolo_output_batch = np.reshape(yolo_output_batch, [batch_size * num_steps, 4102])
    return yolo_output_batch


def choose_video_sequence(test):
    # For MOT 2016:
    # training
    if test == 1:
        w_img, h_img = [608, 608]
        sequence_name = 'video1'
        training_iters = 320
        testing_iters = 320
    elif test == 2:
        w_img, h_img = [1280,720]
        sequence_name = 'Alladin'
        training_iters = 600
        testing_iters = 600
    elif test == 3:
        w_img, h_img = [1280,720]
        sequence_name = 'Badminton2'
        training_iters = 600
        testing_iters = 600
    elif test == 4:
        w_img, h_img = [1280,720]
        sequence_name = 'Badminton1'
        training_iters = 600
        testing_iters = 600

    elif test == 5:
        w_img, h_img = [608,608]
        sequence_name = 'video2'
        training_iters = 773
        testing_iters = 750
    elif test == 6:
        w_img, h_img = [640, 480]
        sequence_name = 'BlurBody'
        training_iters = 334
        testing_iters = 837
    elif test == 7:
        w_img, h_img = [1280,720]
        sequence_name = 'Boxing2'
        training_iters = 400
        testing_iters = 600
    elif test == 8:
        w_img, h_img = [1280,720]
        sequence_name = 'Boxing3'
        training_iters = 600
        testing_iters = 600
    elif test ==9:
        w_img, h_img = [1280,720]
        sequence_name = 'Bike'
        training_iters = 600
        testing_iters = 600
    elif test == 10:
        w_img, h_img = [608,608]
        sequence_name = 'video3'
        training_iters = 842
        testing_iters = 750
    elif test == 11:
        w_img, h_img = [1280,720]
        sequence_name = 'BreakfastClub'
        training_iters = 600
        testing_iters = 600
    elif test == 12:
        w_img, h_img = [1280,720]
        sequence_name = 'CarChase3'
        training_iters = 450
        testing_iters = 600
    elif test == 13:
        w_img, h_img = [320, 246]
        sequence_name = 'Dancer'
        training_iters = 225
        testing_iters = 900
    elif test == 14:
        w_img, h_img = [320, 262]
        sequence_name = 'Dancer2'
        training_iters = 150
        testing_iters = 750
    elif test == 15:
        w_img, h_img = [1280,720]
        sequence_name = 'DriftCar2'
        training_iters = 600
        testing_iters = 600
    elif test == 16:
        w_img, h_img = [608,608]
        sequence_name = 'video4'
        training_iters = 800
        testing_iters = 750
    elif test == 17:
        w_img, h_img = [1280,720]
        sequence_name = 'Drone1'
        training_iters = 600
        testing_iters = 600
    elif test == 18:
        w_img, h_img = [1280, 720]
        sequence_name = 'Helicopter'
        training_iters = 600
        testing_iters = 600
    elif test == 19:
        w_img, h_img = [480, 640]
        sequence_name = 'Human2'
        training_iters = 1128
        testing_iters = 1500
    elif test == 20:
        w_img, h_img = [480, 270]
        sequence_name = 'iceskater1'
        training_iters = 661
        testing_iters = 1194
    elif test == 21:
        w_img, h_img = [1280, 720]
        sequence_name = 'Jet4'
        training_iters = 600
        testing_iters = 600
    elif test == 22:
        w_img, h_img = [608 , 608]
        sequence_name = 'video5'
        training_iters = 900
        testing_iters = 750
    elif test == 23:
        w_img, h_img = [1280, 720]
        sequence_name = 'Mohiniyattam'
        training_iters = 600
        testing_iters = 600
    elif test == 24:
        w_img, h_img = [1280, 720]
        sequence_name = 'Rope'
        training_iters = 600
        testing_iters = 600
    elif test == 25:
        w_img, h_img = [320, 240]
        sequence_name = 'Skater'
        training_iters = 160
        testing_iters = 625
    elif test == 26:
        w_img, h_img = [320,262]
        sequence_name = 'Skater2'
        training_iters = 435
        testing_iters = 900
    elif test == 27:
        w_img, h_img = [608, 608]
        sequence_name = 'video6'
        training_iters = 892
        testing_iters = 750
    #testing
    elif test == 50:
        w_img, h_img = [640, 272]
        sequence_name = 'CarScale'
        training_iters = 252
        testing_iters = 252
    elif test == 51:
        w_img, h_img = [426, 234]
        sequence_name = 'Gym'
        training_iters = 767
        testing_iters = 767
    elif test == 52:
        w_img, h_img = [320, 240]
        sequence_name = 'Human8'
        training_iters = 128
        testing_iters = 128
    elif test == 53:
        w_img, h_img = [640, 480]
        sequence_name = 'video6'
        training_iters = 800
        testing_iters = 892
    elif test == 54:
        w_img, h_img = [864, 480]
        sequence_name = 'video13'
        training_iters = 800
        testing_iters = 896
    elif test == 55:
        w_img, h_img = [864, 480]
        sequence_name = 'video14'
        training_iters = 800
        testing_iters = 922
    elif test == 56:
        w_img, h_img = [640, 480]
        sequence_name = 'video15'
        training_iters = 800
        testing_iters = 768
    elif test == 57:
        w_img, h_img = [480, 640]
        sequence_name = 'Human2'
        training_iters = 800
        testing_iters = 1128
    return [w_img, h_img, sequence_name, training_iters, testing_iters]

