#/usr/bin/env python
#coding:UTF-8

import tensorflow as tf
import numpy as np
import os
import time

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
				continue
	
			#print(video_processed.shape)
			video_processed=tf.concat([video_processed,frame],0)
	return video_processed
	

def next_batch(batch_size,data_path,frames_needed,frame_gap,input_frame_size,num_classes,min_after_dequeue,category='train'):

	#获取文件列表  然后创建文件队列
	files=tf.train.match_filenames_once(data_path+ os.sep+ 'split1_train_*')
	filename_queue=tf.train.string_input_producer(files,num_epochs=100,shuffle=True) #这里的shuffle
	
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
	decoded_videos=tf.decode_raw(features['video'],tf.uint8)

	num_frames=tf.cast(features['num_frames'],tf.int32)-10  #有时候这个比读进来的多了一帧(此处折中10帧，10帧正常不会出错)
	height=tf.cast(features['height'],tf.int32)
	width=tf.cast(features['width'],tf.int32)

	videos=tf.reshape(decoded_videos,[-1,height,width,3])

	print('##########')
	print(type(num_frames))
	print(type(height))
	print('##########')	
#print('#######')
#print(videos.shape)
#print('#######')
#num_frames=videos.shape[0].value
	#调用函数实现对videos格式的预处理
	videos=preprocess(videos,frames_needed,frame_gap,input_frame_size,num_frames)


	labels=tf.cast(features['label'],tf.int32)
	groundtruth=tf.one_hot(labels,num_classes)

	#groundtruth=tf.zeros([1,num_classes],dtype=tf.float32)
	#index=None
	#a=tf.constant(3)
	#print('111111111111111111111111111111111111')
	#with tf.Session() as s:
	#print('22222222222222222222222222222222222')
	#print(a.eval())
	#print('a is ok')
	#index=s.run(labels)
	#print(index)
	#groundtruth[0][labels]=1.0

	#以batch_size为一组打包
	capacity = min_after_dequeue + 300 * batch_size
	video_batch, groundtruth_batch = tf.train.shuffle_batch([videos,groundtruth],
														batch_size=batch_size, 
														capacity=capacity,
														min_after_dequeue=min_after_dequeue)
	#如果每次只取1帧，则squeeze掉该维度(groundtruth不需要)
	if frames_needed==1:
		video_batch=tf.squeeze(video_batch)

	return video_batch, groundtruth_batch


if __name__=='__main__':

	data_path='/home/mcger/datasets/tfrecords/ucf101'
	batch_size=10
	video,groundtruth=next_batch(batch_size,data_path,1,2,224,101,53,category='train')
	with tf.Session() as sess:
		tf.local_variables_initializer().run()
		tf.global_variables_initializer().run()
		coord=tf.train.Coordinator()
		threads=tf.train.start_queue_runners(sess=sess,coord=coord)
		time_start=time.time()
		for i in range(20):
			v,g=sess.run([video,groundtruth])
			print(v.shape)
			print(g.shape)
			#print(sess.run(groundtruth).shape)
		coord.request_stop()
		coord.join(threads)
		time_end=time.time()
		period=time_end-time_start
		print('consume %f'%period)
