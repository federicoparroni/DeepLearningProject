from skimage import io
from random import shuffle
from skimage.transform import resize
import skimage
import numpy as np
import os

#==========PREPROCESSING load data ================

img_data_list = [] #elements list, an element is a couple of image (i1,i2)
img_label_list = [] #labels list, can be 1 if the faces are the same or 0 if not


def LoadData():
    folders = os.listdir('dataset')

    for i in folders: #folders for
        for img1 in os.listdir('dataset/'+i):
            for j in folders:  # folders for
                for img2 in os.listdir('dataset/'+j):

                    #read images
                    input_img1 = skimage.io.imread('dataset/'+i+'/'+img1)
                    input_img2 = skimage.io.imread('dataset/'+j+'/'+img2)

                    #bring images in grayscale
                    input_img1 = skimage.color.rgb2gray(input_img1)
                    input_img2 = skimage.color.rgb2gray(input_img2)

                    #create a np array and flatten it
                    #f_i_img1 = np.array(input_img1).flatten()
                    #f_i_img2 = np.array(input_img2).flatten()

                    #resize of the image
                    r_input_img1 = resize(input_img1, (input_img1.shape[0]//2, input_img1.shape[1]//2))
                    r_input_img2 = resize(input_img2, (input_img2.shape[0]// 2, input_img2.shape[1]//2))

                    #concatenate the two arrays
                    inp = np.concatenate((r_input_img1, r_input_img2))

                    img_data_list.append(inp)
                    if i == j:
                        img_label_list.append(1)
                    else:
                        img_label_list.append(0)



def GetData():
    LoadData()
    v = list(range(len(img_label_list)))
    shuffle(v)

    train_v = []
    label_train_v = []

    test_v = []
    label_test_v = []

    for i in range((len(v))):
        if i < len(v)*0.8:
            train_v.append(img_data_list[v[i]])
            label_train_v.append(img_label_list[v[i]])
        else:
            test_v.append(img_data_list[v[i]])
            label_test_v.append(img_label_list[v[i]])

    return (np.expand_dims(np.array(train_v), 3), np.array(label_train_v)), (np.expand_dims(np.array(test_v), 3), np.array(label_test_v))



