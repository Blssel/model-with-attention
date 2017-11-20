#/usr/bin/env python
#coding:UTF-8

import tensorflow as tf
import numpy as np
import os

#预处理原则，在frame_len中（不足的就循环取），每间隔frame_gap帧取一帧
#帧全部resize成224*224
def preprocess(videos,frame_len,frame_gap,frame_size,num_frames):
	video_processed=None
	#逐帧操作，（第i帧相当于第i%num_frames帧）
	flag=True
	for i in range(frame_len):
		if i%frame_gap==0:
			j=i%num_frames
			frame=tf.image.resize_images(videos[j],(frame_size,frame_size),method=0) #这里method=0代表使用双线性插值法
			frame=tf.reshape(frame,[1,frame_size,frame_size,3])
			#print(frame.shape)
			if flag==True:
				video_processed=frame
				flag=False
			else:
				#print(video_processed.shape)
				video_processed=tf.concat([video_processed,frame],0)
	return video_processed
	

def next_batch(batch_size,data_path,frames_needed,frame_gap,input_frame_size,category='train'):#frames_needed,frame_gap,input_frame_size等需要在最外层指定
	#获取文件列表  然后创建文件队列
	files=tf.train.match_filenames_once(data_path+ os.sep+ 'split1_train_*')
	filename_queue=tf.train.string_input_producer(files,shuffle=False) #这里的shuffle
	
	#读取文件
	reader=tf.TFRecordReader()
	_,serialized_example=reader.read(filename_queue)

	#解析读取的样例(告诉程序按照以下方式解析serialized_example)
	features=tf.parse_single_example(
		serialized_example,
		features={
			'video':tf.FixedLenFeature([],tf.string),
			 'label':tf.FixedLenFeature([],tf.int64),
			 'height':tf.FixedLenFeature([],tf.int64),
			 'width':tf.FixedLenFeature([],tf.int64),
			 #'fps':_int64_feature(fps),
			 'num_frames':tf.FixedLenFeature([],tf.int64)
		})
	
	#解码
	decoded_videos=tf.decode_raw(features['video'],tf.float64)
	retyped_videos=tf.cast(decoded_videos, tf.float32)

	labels=tf.cast(features['label'],tf.int32)
	num_frames=tf.cast(features['num_frames'],tf.int32)
	height=tf.cast(features['height'],tf.int32)
	width=tf.cast(features['width'],tf.int32)

	videos=tf.reshape(retyped_videos,[num_frames,height,width,3])
	#调用函数实现对videos格式的预处理
	videos=preprocess(videos,frames_needed,frame_gap,input_frame_size,num_frames)

	#以batch_size为一组打包
	min_after_dequeue = 3000
	capacity = min_after_dequeue + 3 * batch_size

	video_batch, label_batch = tf.train.shuffle_batch([videos, labels],
														batch_size=batch_size, 
														capacity=capacity,
														min_after_dequeue=1000)
	return video_batch, label_batch


if __name__=='__main__':
	data_path='/home/mcger/datasets/tfrecords/ucf101'
	batch_size=5
	video,label=next_batch(batch_size,data_path,64,4,224,category='train')
	print(video)
