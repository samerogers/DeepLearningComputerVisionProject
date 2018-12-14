import csv
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont
import cv2
import os
from os import path
#from skimage.measure import compare_ssim as ssim

from yolo_v3_1_object import yolo_v3, load_weights, detections_boxes, non_max_suppression,_iou

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('input_img', '', 'Input image')
tf.app.flags.DEFINE_string('output_img', '', 'Output image')
tf.app.flags.DEFINE_string('class_names', 'coco.names', 'File with class names')
tf.app.flags.DEFINE_string('weights_file', 'yolov3.weights', 'Binary file with detector weights')

tf.app.flags.DEFINE_integer('size', 416, 'Image size')

tf.app.flags.DEFINE_float('conf_threshold', 0.3, 'Confidence threshold')
tf.app.flags.DEFINE_float('iou_threshold', 0.4, 'IoU threshold')


def load_coco_names(file_name):
    names = {}
    with open(file_name) as f:
        for id, name in enumerate(f):
            names[id] = name
    return names
    
def deduce_index(index,resolution):
    flatten_index=index*85
    block_numb=flatten_index/255
    row_n=block_numb/resolution
    col_n=block_numb%resolution
    #print(row_n)
    #print(col_n)
    return row_n,col_n


def draw_boxes(filtered_boxes,feature_maps, img, cls_names, detection_size,inter_resolution_threshold,num_obj,iteration):
    draw = ImageDraw.Draw(img)
    count=0
    obj_count=0
    box_list=[]
    box_cls=[]
    bbox_list=[]

    feature_list=[]
    buffer_list_feature=[]
    buffer_list_bbox=[]
    
    for boxes in filtered_boxes:
       
        for cls, bboxs in boxes.items():
            color = tuple(np.random.randint(0, 256, 3))
            #if cls==0:
            for box, score, index in bboxs:
                draw_box=True
                
                if count==1:  # resolution 26X26
                    for b,cl in zip(box_list,box_cls): #check IOUs for duplicate boxes
                        iou=0.0
                        if cl==cls:
                            iou=_iou(b,box)
                            #print(iou)
                            if(iou>inter_resolution_threshold):
                                draw_box=False
                    if(draw_box):
                        if(obj_count>=num_obj):
                            break
                        x_upper_left=box[0]
                        y_upper_left=box[1]
                        x_lower_right=box[2]
                        y_lower_right=box[3]
                        width=x_lower_right-x_upper_left
                        height=y_lower_right-y_upper_left
                        new_box=[x_upper_left,y_upper_left,width,height]
                        feature_map=feature_maps[count]
                        row_n,col_n=deduce_index(int(index),26)
                        feature_vector=feature_map[0,row_n,col_n,:]
                        feature_list.append(feature_vector)
                        box_for_iou=box

#                        draw.rectangle(box, outline=color)
#                        #font = ImageFont.truetype("./BebasNeue-Regular.ttf", 15)
#                        draw.text(box[:2], 'Object:{} {:.2f}%'.format(obj_count, score * 100), fill='white')#,font=font)
                        bbox_list.append(box)
                        #feature_vector_array[iteration][obj_count]=feature_vector
                        #bbox_coord_array[iteration][obj_count]=new_box
                        obj_count+=1
                elif count==2:  # resolution 52X52
                    
                    for b,cl in zip(box_list,box_cls): #check IOUs for duplicate boxes
                        iou=0.0
                        if cl==cls:
                            iou=_iou(b,box)
                            if(iou>inter_resolution_threshold):
                                draw_box=False
                    if(draw_box):  
                        if(obj_count>=num_obj):
                            break
                        x_upper_left=box[0]
                        y_upper_left=box[1]
                        x_lower_right=box[2]
                        y_lower_right=box[3]
                        width=x_lower_right-x_upper_left
                        height=y_lower_right-y_upper_left
                        new_box=[x_upper_left,y_upper_left,width,height]
                        feature_map=feature_maps[count]
                        row_n,col_n=deduce_index(int(index),52)
                        feature_vector=feature_map[0,row_n,col_n,:]
                        feature_list.append(feature_vector)
                        box_for_iou=box
                        #box = convert_to_original_size(box, np.array(detection_size), np.array(img.size))
#                        draw.rectangle(box, outline=color)
#                        #font = ImageFont.truetype("./BebasNeue-Regular.ttf", 15)
#                        draw.text(box[:2], 'Object:{} {:.2f}%'.format(obj_count, score * 100), fill=color)#,font=font)
                        bbox_list.append(box)
                        #feature_vector_array[iteration][obj_count]=feature_vector
                        #bbox_coord_array[iteration][obj_count]=new_box
                        obj_count+=1
                else:  # resolution 13X13
                    
                    if(obj_count>=num_obj):
                        break
                    x_upper_left=box[0]
                    y_upper_left=box[1]
                    x_lower_right=box[2]
                    y_lower_right=box[3]
                    width=x_lower_right-x_upper_left
                    height=y_lower_right-y_upper_left
                    new_box=[x_upper_left,y_upper_left,width,height]
                    feature_map=feature_maps[count]
                    row_n,col_n=deduce_index(int(index),13)
                    feature_vector=feature_map[0,row_n,col_n,:]
                    feature_list.append(feature_vector)
                    box_for_iou=box
                    #box = convert_to_original_size(box, np.array(detection_size), np.array(img.size))
#                    draw.rectangle(box, outline=color)
#                    #font = ImageFont.truetype("./BebasNeue-Regular.ttf", 15)
#                    draw.text(box[:2], 'Object:{} {:.2f}%'.format(obj_count, score * 100), fill=color)#,font=font)
                    bbox_list.append(box)
                    #feature_vector_array[iteration][obj_count]=feature_vector
                    #bbox_coord_array[iteration][obj_count]=new_box
                    obj_count+=1
                    
                box_list.append(box_for_iou)
                box_cls.append(cls)
                
        count+=1
    return feature_list,bbox_list

def find_best_bbox(feat_list, bb_list, gt, it, w, h):
    if (len(bb_list)==0):
        bbox = np.array([w/2,h/2,0,0])
        features = np.zeros([1,255])
        best_match = 0.0
	valid = 0
    else:
        gttmp=gt
        gt_x0=gttmp[0]
        gt_x1=gttmp[0]+gttmp[2]
        gt_y0=gttmp[1]
        gt_y1=gttmp[1]+gttmp[3]
        gtbox=np.array([gt_x0,gt_y0,gt_x1,gt_y1])
        
        best_match = 0.0
        best_idx=0
        for i in range(len(bb_list)):
            iou = _iou(bb_list[i],gtbox)
            if (iou>best_match):
                best_match=iou
                best_idx=i

        if (best_match < 0.1):
            bbox = np.array([w/2,h/2,0,0])
            features = np.zeros([1,255])
            best_match = 0.0
            valid = 0;
        else:
            features = feat_list[best_idx]
            bbtmp = bb_list[best_idx]
            bbtmp_w = bbtmp[2]-bbtmp[0]
            bbtmp_h = bbtmp[3]-bbtmp[1]
            bbox = np.array([bbtmp[0],bbtmp[1],bbtmp_w,bbtmp_h])
            valid = 1
    print 'Frame:',it,' YOLO:',bbox,' GT:',gt
    return features, bbox, best_match, valid

def convert_to_original_size(box, size, original_size):
    x_scale=float(original_size[0])/float(size[0])
    y_scale=float(original_size[1])/float(size[1])
    #print(x_scale,y_scale)
    x_upper_left=int(np.round(box[0]*x_scale))
    y_upper_left=int(np.round(box[1]*y_scale))
    x_lower_right=int(np.round(box[2]*x_scale))
    y_lower_right=int(np.round(box[3]*y_scale))
    
    return [x_upper_left,y_upper_left,x_lower_right,y_lower_right]

def parse(filename):
    with open(filename, 'rb') as csvfile:
        dialect = csv.Sniffer().sniff(csvfile.read(), delimiters='\t, ')
        csvfile.seek(0)
        reader = csv.reader(csvfile, dialect, quoting=csv.QUOTE_NONNUMERIC)
        gt = np.empty((1,4))
        for line in reader:
            #print line
            gtlist = np.reshape(np.array(line), (1,4))
            #print(np.shape(gtlist2))
            #print gtlist
            gt = np.concatenate((gt,gtlist),axis=0)
        #print gt
        #print np.shape(gt)
        return gt

classes = load_coco_names(FLAGS.class_names)
inter_resolution_threshold=0.45  # set IOU threshold for duplicate boxes across different resolutions
# placeholder for detector inputs
inputs = tf.placeholder(tf.float32, [None, FLAGS.size, FLAGS.size, 3])

with tf.variable_scope('detector'):
    detections,feature_map1 = yolo_v3(inputs, len(classes), data_format='NHWC')
    load_ops = load_weights(tf.global_variables(scope='detector'), FLAGS.weights_file)

path = os.getcwd()
data_path=path+'/data'
test_dirs=os.listdir(data_path)
for f in test_dirs:
    active_dir=data_path+'/'+f+'/'
    output_dir=active_dir+'/output/'
    test_path=active_dir+'img'
    print test_path
    feature_file=output_dir+'features_video.npy'
    if (os.path.isfile(feature_file)):
        print 'File already exists'
    else:
        bbox_file=output_dir+'bbox_video.npy'
        iou_error_file=output_dir+'iou_error.txt'
        gt=parse(active_dir+'groundtruth_rect.txt')
        #print len(gt)
        #test_path='/home/smohan7/Desktop/project/yolov3/Pratik/video12' #folder path for frames
        test_set_length=len(os.listdir(test_path))
        gtlen = len(gt)-1
        #print gtlen, test_set_length
        if (gtlen >= test_set_length):
            num_obj=32 #specify the number of objects to be detected

            feature_vector_array=np.zeros([test_set_length,1,255])
            bbox_coord_array=np.zeros([test_set_length,1,4])
            iou_error_array=np.zeros([test_set_length,1])
            test_dict={}  
            img_names_dict={} 

            img_count=0
            file_list = os.listdir(test_path)
            offset=999999999
            for item in file_list: #read frames
                cls=int(filter(str.isdigit, item))
                if (cls<offset):
                    offset=cls
            for item in file_list: #read frames
                image=cv2.imread(test_path+'/'+item)
                img_resized = cv2.resize(image,(416, 416))
                cls=int(filter(str.isdigit, item))
                test_dict[cls-offset]=img_resized
                img_names_dict[cls-offset]=item

            boxes,feature_map_yolo = detections_boxes(detections,feature_map1)
            last_best = np.zeros([1,255])
            with tf.Session() as sess:
                sess.run(load_ops)
                for i in range(test_set_length): 
                    img=Image.open(test_path+'/'+img_names_dict[i])
                    width=img.width
                    height=img.height

                    detected_boxes,feature_maps = sess.run([boxes,feature_map_yolo], feed_dict={inputs: [np.array(test_dict[i], dtype=np.float32)]})

                    filtered_boxes = non_max_suppression(detected_boxes,np.array((FLAGS.size, FLAGS.size)), np.array(img.size),confidence_threshold=FLAGS.conf_threshold,iou_threshold=FLAGS.iou_threshold)
                    #print(filtered_boxes)

                    feature_list,bbox_list=draw_boxes(filtered_boxes,feature_maps, img, classes, (FLAGS.size, FLAGS.size),inter_resolution_threshold,num_obj,i)
                    best_feat, best_bb, best_iou, valid = find_best_bbox(feature_list,bbox_list,gt[i+1],i,width,height)
                    if valid==1:
                        last_best = best_bb
                    feature_vector_array[i,0]=best_feat
                    bbox_coord_array[i,0]=best_bb
                    iou_error_array[i]=1-best_iou

                    gtr = gt[i+1]
                    gtbox = [ gtr[0], gtr[1], gtr[0]+gtr[2], gtr[1]+gtr[3] ]
                    box = [ best_bb[0], best_bb[1], best_bb[0]+best_bb[2], best_bb[1]+best_bb[3] ]
                    draw = ImageDraw.Draw(img)
                    draw.rectangle(gtbox, outline=(255,0,0))
                    draw.text(gtbox[:2], 'GT', fill='white')
                    draw.rectangle(box, outline=(0,255,0))
                    draw.text(box[:2], 'YOLO', fill='white')

                    # uncomment the following line to save frames with bounding boxes
                    if not os.path.exists(output_dir):
                        os.mkdir(output_dir)

                    img.save(output_dir+'output_'+str(i)+'.jpg')
                    img_count+=1

            # saving features and bounding box coordinates
            np.save(feature_file,np.array(feature_vector_array))
            np.save(bbox_file,np.array(bbox_coord_array))
            np.savetxt(iou_error_file,iou_error_array)

