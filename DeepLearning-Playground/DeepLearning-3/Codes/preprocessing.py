import cv2
import os
from glob import glob
import numpy as np
from tqdm import tqdm
import pandas as pd

img_x = 250
img_y = 250

train_images = glob(os.path.join("../DL3 Dataset/train_img", "*.jpg"))
test_images = glob(os.path.join("../DL3 Dataset/test_img", "*.jpg"))
print('Length of train Set', len(train_images))
print('Length of test Set', len(test_images))

# Resize
train_x = []
train_image_name = []
test_x = []
test_image_name = []
# for img in tqdm(train_images):
#     full_size_image = cv2.imread(img)

#     train_image_name.append(os.path.basename(img))
#     train_x.append(cv2.resize(full_size_image, (img_x, img_y), interpolation=cv2.INTER_CUBIC))

for img in tqdm(test_images):
    full_size_image = cv2.imread(img)
    test_image_name.append(os.path.basename(img))
    test_x.append(cv2.resize(full_size_image, (img_x, img_y), interpolation=cv2.INTER_CUBIC))

# train_data = pd.DataFrame()
# train_data['img_name'] = train_image_name
# train_data['array'] = train_x
# np.savez("../PreProcessing/train_images_arrays", train_x)
# train_data.to_csv('../PreProcessing/train_data.csv', index=False)

test_data = pd.DataFrame()
test_data['img_name'] = test_image_name
test_data['array'] = test_x
np.savez("../PreProcessing/test_images_array", test_x)
test_data.to_csv('../PreProcessing/test_data.csv', index=False)
