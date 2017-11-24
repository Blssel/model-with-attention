#/usr/bin/env python
#coding:UTF-8

import tensorflow as tf
import cv2
import threadpool
import os 
import numpy as np
import threading

#读split文件，生成相应的tfrecord文件
#针对每个split生成分别rerecord文件
DATASET_PATH= '/home/mcger/datasets/UCF-101'
NUM_CLASSES=101
SPLIT_PATH= 'data/ucf101/split1_train.txt'

TFRECORD_PATH_BASE='/home/mcger/datasets/tfrecords/ucf101/split1_train_'
'''
WRITERS=[
for i in range(101):
	tfrecord_path= TFRECORD_PATH_BASE+ '%.3d-of-%.3d'%(i, NUM_CLASSES)+'.tfrecords'
	WRITERS.append()
'''

def _read_split_to_list(split_path=SPLIT_PATH):
	with open(split_path) as f:
		data_list=f.read().strip().split('\n') #加strip是为了去除最后一个换行符号，防止split('\n')的时候会在末尾多出一个空字符‘'
	return data_list

def _int64_feature(value):
	return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
			
def _get_video_raw(cap):
	ret, frame=cap.read()
	video=np.array([],dtype=np.float64)
	flag=True
	while(ret==True):
		if flag:
			video=frame
			flag=False
			ret,frame=cap.read()
			continue
		video=np.concatenate((video,frame))
		ret,frame=cap.read()
	print(video.shape)
	return video.tostring()

	

#本函数负责将一条item对应的数据写入tfrecord文件
#‘这里我人为规定：同一个label下的数据存到一起，这样方便管理’
def write_to_tfrecord(item):
	item_splited=item.strip().split(' ')

	label=int(item_splited[0])
	print(label)
	class_name=item_splited[1]
	video_name=item_splited[2]
	#计算视频路径
	video_path=DATASET_PATH+ os.sep+ class_name+ os.sep+ video_name
	print(video_path)
	#确定保存路径
	tfrecord_path= TFRECORD_PATH_BASE+ item_splited[2]+ '.tfrecords'
	#读取视频
	cap= cv2.VideoCapture(video_path)
	print(cap.isOpened())
	#顺手存一些常用的参数
	height= int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
	width= int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
	print(width)
	#fps=int(cap.get(cv2.cv.CV_CAP_PROP_FPS))
	num_frames=int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
	#du视频bing转化成字符串
	video_raw=_get_video_raw(cap)
	
	#创建一个writer来写tfrecord
	writer=tf.python_io.TFRecordWriter(tfrecord_path)
	example= tf.train.Example(features=tf.train.Features(feature={
		'video':_bytes_feature(video_raw),
		'label':_int64_feature(label),
		'height':_int64_feature(height),
		'width':_int64_feature(width),
		#'fps':_int64_feature(fps),
		'num_frames':_int64_feature(num_frames)
	}))
	
	#xie ru
	lock=threading.Lock()
	lock.acquire()
	try:
		writer.write(example.SerializeToString())
	finally:
		writer.close()
		lock.release()
	print(item)



if __name__=='__main__':
	#读取整个split并生成队列
	data_list=_read_split_to_list(SPLIT_PATH)
	print(data_list)
	
	#定义多个线程一起往tfrecord中写
	pool= threadpool.ThreadPool(100)
	requests=threadpool.makeRequests(write_to_tfrecord,data_list)
	[pool.putRequest(req) for req in requests]
	pool.wait()
	
