# Get and unpack celeba
# after cropping images can be removed

#!mkdir data
#!mkdir ./data/celeba
#!kaggle datasets download -d jessicali9530/celeba-dataset
#!unzip celeba-dataset.zip
#!unzip -q img_align_celeba.zip
#!mv list_eval_partition.csv list_landmarks_align_celeba.csv list_bbox_celeba.csv list_attr_celeba.csv ./data/celeba
#!mv img_align_celeba ./data/celeba/img_align_celeba

# we assume to have celeba in 'celeba/img_align_celeba'
# execute the previous commands to download it from the web

### Preprocess data

import numpy as np
import os
from PIL import Image
from imageio import imread, imwrite

data_folder = './data'

def load_celeba_data(flag='training', side_length=None, num=None):
	dir_path = os.path.join(ROOT_FOLDER, 'celeba/img_align_celeba')
	filelist = [filename for filename in os.listdir(dir_path) if filename.endswith('jpg')]
	assert len(filelist) == 202599
	if flag == 'training':
		start_idx, end_idx = 0, 162770
	elif flag == 'val':
		start_idx, end_idx = 162770, 182637
	else:
		start_idx, end_idx = 182637, 202599

	imgs = []

	for i in range(start_idx, end_idx):
		img = np.array(imread(dir_path + os.sep + filelist[i]))
		img = img[45:173, 25:153]
		img = np.array(Image.fromarray(img).resize((side_length, side_length), resample=Image.BILINEAR))
		new_side_length = np.shape(img)[1]
		img = np.reshape(img, [1, new_side_length, new_side_length, 3])
		imgs.append(img)
		if num is not None and len(imgs) >= num:
			break
		if len(imgs) % 5000 == 0:
			print('Processing {} images...'.format(len(imgs)))
	imgs = np.concatenate(imgs, 0)

	return imgs.astype(np.uint8)


# Center crop 128x128 and resize to 64x64
def preprocess_celeba():
	x_val = load_celeba_data('val', 64)
	np.save(os.path.join('data', 'celeba', 'val.npy'), x_val)
	x_test = load_celeba_data('test', 64)
	np.save(os.path.join('data', 'celeba', 'test.npy'), x_test)
	x_train = load_celeba_data('training', 64)
	np.save(os.path.join('data', 'celeba', 'train.npy'), x_train)

#preprocess_celeba only the first time you use it.
#after precprocessing you can delete celeba/img_align_celeba
#uncomment the next line to create the datatset

#preprocess_celeba()

def load_cifar10():
	from tensorflow.keras.datasets import cifar10 
	(x_train, _), (x_test, _) = cifar10.load_data()

	# Normalize between 0 and 1
	x_train = x_train.astype('float32') / 255
	x_test  = x_test.astype('float32') / 255

	print('x_train shape: ' + str(x_train.shape))
	print('x_test shape: ' + str(x_test.shape))

	return x_train, x_test

def load_celeba():
	data_folder = os.path.join(root_folder, 'data/celeba', )
	x_train = np.load(os.path.join('data', 'celeba','train.npy'))
	x_test = np.load(os.path.join('data', 'celeba', 'test.npy'))

	# Normalize between 0 and 1
	x_train = x_train.astype('float32') / 255
	x_test  = x_test.astype('float32') / 255

	print('x_train shape: ' + str(x_train.shape))
	print('x_test shape: ' + str(x_test.shape))

	return x_train, x_test