from keras.models import load_model
from keras.callbacks import ModelCheckpoint
import pandas as pd
import operator
import numpy as np
import h5py
# fix random seed for reproducibility
seed = 126
np.random.seed(seed)
test=pd.read_csv('test.csv')

test_mat=np.zeros((test.shape[0],28,28))
for i in range(len(test)):
    arr=np.asarray(test.iloc[i])
    test_mat[i]=arr.reshape(28,28)



X_test = test_mat.reshape(test_mat.shape[0], 1, 28, 28).astype('float32')
X_test = X_test / 255

model= load_model('Digit_classifier.01-0.016.hdf5')

preds = model.predict (X_test, verbose=1)
index=[]
for ind in range(len(test)):
	index.append(max(enumerate(preds[ind]), key=operator.itemgetter(1))[0])

ImageId=range(1,len(test)+1)


output1 = pd.DataFrame( data={"ImageId":ImageId,"Label": index} )
output1.to_csv("my3Submission.csv", index=False,quoting=3)