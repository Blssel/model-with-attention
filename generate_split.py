#/usr/bin/env python
#coding:UTF-8
import os
import random

'''
运行前需要修改TRAIN_SPLIT_NAME，VALIDATION_SPLIT_NAME，TEST_SPLIT_NAME值的编号
'''
DATASET='ucf-101'
DATASET_PATH='/home/mcger/datasets/UCF-101'

SPLIT_PATH= './data'
TRAIN_SPLIT_NAME='split3_train.txt'
VAL_SPLIT_NAME='split3_val.txt'
TEST_SPLIT_NAME='split3_test.txt'#定义所生成的split的名称：以split+编号+扩展名的形式命名（比如split3.txt）

TRAIN_PERC=0.8 #规定训练集占80％
VAL_PERC=0.1
TEST_PERC=0.1

#（私有方法）以追加的方式将字符串写入文件
def _write_in(split_name, item, split_path=SPLIT_PATH):
	if not os.path.exists(split_path):
		os.mkdir(split_path)
	with open(split_path+ os.sep+ split_name,'a') as f:
		f.write(item+ '\n')

def generate_splits(dataset = DATASET, 
					dataset_path = DATASET_PATH,
					split_path=SPLIT_PATH,	
					train_split_name = TRAIN_SPLIT_NAME, 
					val_split_name = VAL_SPLIT_NAME,
					test_split_name = TEST_SPLIT_NAME,
					train_perc = TRAIN_PERC,
					val_perc = VAL_PERC,
					test_perc = TEST_PERC):

	class_list= sorted(os.listdir(dataset_path))
	label=-1
	for class_name in class_list:
		file_list= sorted(os.listdir(dataset_path+ os.sep + class_name))
		label+=1
		#按概率决定让每一条数据分在哪个集合中
		for current_file in file_list:
			rnd=random.random()
			#如果在0-0.8，则放在train_split_name中，否则放在另外两个中
			item=str(label)+ ' '+ class_name+ ' '+ current_file
			if rnd>=0.0 and rnd<0.8:
				_write_in(train_split_name, item, split_path)
			elif rnd>=0.8 and rnd<0.9:
				_write_in(val_split_name, item, split_path)
			else:
				_write_in(test_split_name, item, split_path)

if __name__=='__main__':
	generate_splits(dataset = DATASET,
					dataset_path = DATASET_PATH,
					split_path=SPLIT_PATH,
					train_split_name = TRAIN_SPLIT_NAME,
					val_split_name = VAL_SPLIT_NAME,
					test_split_name = TEST_SPLIT_NAME,
					train_perc = TRAIN_PERC,
					val_perc = VAL_PERC,
				    test_perc = TEST_PERC)
