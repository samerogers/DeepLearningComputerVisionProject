import csv
import numpy as np
import os
from os import path

def parse(filename):
    with open(filename, 'rb') as csvfile:
        dialect = csv.Sniffer().sniff(csvfile.read(), delimiters='\t, ')
        csvfile.seek(0)
        reader = csv.reader(csvfile, dialect, quoting=csv.QUOTE_NONNUMERIC)
        gt = np.empty((1,4))
        for line in reader:
            #print line
            gtlist = np.reshape(np.array(line), (1,6))
            gtlist2 = gtlist[0,1:5]
            gtlist3 = np.reshape(gtlist2, (1,4))
            #print(np.shape(gtlist2))
            #print gtlist
            gt = np.concatenate((gt,gtlist3),axis=0)
        #print gt
        #print np.shape(gt)
    return gt

path = os.getcwd()
data_path=path+'/TinyTLP'
test_dirs=os.listdir(data_path)
for f in test_dirs:
    active_dir=data_path+'/'+f+'/'
    gt=parse(active_dir+'groundtruth_rect.txt')
    np.savetxt(active_dir+'/groundtruth_rect1.txt',gt[1:,:],fmt='%.0f',delimiter=',')
