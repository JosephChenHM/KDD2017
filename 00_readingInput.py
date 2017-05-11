import numpy as np
np.random.seed(1337)

''' Read input files '''
my_data = np.genfromtxt('dataset/training.csv', delimiter=',',skip_header=1)

''' The first column to the 199th column is used as input features '''
X_train = my_data[:,0:11]
X_train = X_train.astype('float32')

''' The 200-th column is the answer '''
Y_train = my_data[:,12]
Y_train = Y_train.astype('float32')

''' Convert to one-hot encoding '''
#from keras.utils import np_utils
#Y_train = np_utils.to_categorical(y_train,5)

''' Shuffle training data '''
from sklearn.utils import shuffle
X_train,Y_train = shuffle(X_train,Y_train,random_state=100)