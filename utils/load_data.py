def load_cifar10():
	from tensorflow.keras.datasets import cifar10 
	(x_train, _), (x_test, _) = cifar10.load_data()

	# Normalize between 0 and 1
	x_train = x_train.astype('float32') / 255
	x_test  = x_test.astype('float32') / 255

	print('x_train shape: ' + str(x_train.shape))
	print('x_test shape: ' + str(x_test.shape))

	return x_train, x_test