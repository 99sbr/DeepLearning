from keras.models import load_model
import numpy as np
import pandas as pd
from tqdm import tqdm
from keras_applications.inception_v3 import preprocess_input


try:
    test_data = pd.read_csv('../PreProcessing/test_data.csv')
    submission = pd.read_csv('../DL3 Dataset/meta-data/sample_submission.csv')
    test_array = np.load("../PreProcessing/test_images_array.npz")
    model = load_model('../ModelCheckpoints/InceptionV3.01-0.908.hdf5')
except Exception as e:
    print(e)
    exit(0)


test_array = test_array['arr_0']
test_array=test_array/255
print('pre-processing input images\n')
test_array = preprocess_input(test_array)
test_img_name = list(test_data['img_name'].values)
print('Done. Prediction started')
l = []
for img_name in tqdm(submission.Image_name.values):
    pred = model.predict(test_array[test_img_name.index(img_name)].reshape(1, 250,250, 3))
    pred[pred >= 0.5] = 1
    pred[pred < 0.5] = 0
    pred = (pred[0]).tolist()
    pred.insert(0, img_name)
    l.append(pred)

prediction = pd.DataFrame()
prediction = prediction.append(l, ignore_index=True)
prediction.columns = submission.columns
print(prediction.head())
prediction.to_csv('../Submission/sub5.csv', index=False)
