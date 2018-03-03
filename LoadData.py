from skimage import io
from random import shuffle
from skimage.transform import resize
import skimage
import numpy as np
import os
import random
import math



#==========PREPROCESSING load data ================


def LoadData(folder_path):
    img_data_list = []  # elements list, an element is a couple of image (i1,i2)
    img_label_list = []  # labels list, can be 1 if the faces are the same or 0 if not

    folders = os.listdir(folder_path)

    for i in folders:
        a, b = CreatePositiveCouples(folder_path + '/' + i)
        c, d = CreateNegativeCouples(folder_path + '/' + i)

        for i in range(len(a)):
            img_data_list.append(a[i])
            img_label_list.append(b[i])

        for i in range(len(c)):
            img_data_list.append(c[i])
            img_label_list.append(d[i])

    return img_data_list, img_label_list




#creates the couples for which the correspondence of the face is true (same person)
def CreatePositiveCouples(folder_path):
    img_data_list = []  # elements list, an element is a couple of image (i1,i2)
    img_label_list = []  # labels list, can be 1 if the faces are the same or 0 if not

    for img1 in os.listdir(folder_path):
        for img2 in os.listdir(folder_path):
            img_data_list.append(CreateCouple(folder_path + '/' + img1, folder_path + '/' + img2))
            img_label_list.append(1)

    return img_data_list, img_label_list


#creates the couples for which the correspondence of the face is false(different person)
def CreateNegativeCouples(folder_path):
    img_data_list = []  # elements list, an element is a couple of image (i1,i2)
    img_label_list = []  # labels list, can be 1 if the faces are the same or 0 if not

    couples_to_do = len(os.listdir(folder_path))

    for img1 in os.listdir(folder_path):
        folders = os.listdir(folder_path + '/..')

        basePath = folder_path.split('/')[0]
        #remove the name of the current folder
        folders.remove(folder_path.split('/')[-1:][0])


        shuffle(folders)

        for i in folders[:couples_to_do]:
            i = basePath + '/' + i

            rnd_numb = math.floor(random.random() * len(os.listdir(i)))
            img2 = i + '/' + os.listdir(i)[rnd_numb]

            img_data_list.append(CreateCouple(folder_path + '/' + img1, img2))
            img_label_list.append(0)

    return img_data_list, img_label_list


#return the concatenation of the two images after the preprocessing
def CreateCouple(img1_path, img2_path):

    # read images
    input_img1 = skimage.io.imread(img1_path)
    input_img2 = skimage.io.imread(img2_path)

    # bring images in grayscale
    input_img1 = skimage.color.rgb2gray(input_img1)
    input_img2 = skimage.color.rgb2gray(input_img2)

    # resize of the image
    r_input_img1 = resize(input_img1, (input_img1.shape[0]//2, input_img1.shape[1]//2))
    r_input_img2 = resize(input_img2, (input_img2.shape[0]//2, input_img2.shape[1]//2))

    # concatenate the two arrays
    inp = np.concatenate((r_input_img1, r_input_img2))

    return inp


def GetData(path):

    (img_data_list, img_label_list) = LoadData(path)

    v = list(range(len(img_label_list)))
    shuffle(v)

    vec = []
    label_vec = []

    for i in range((len(v))):
        vec.append(img_data_list[v[i]])
        label_vec.append(img_label_list[v[i]])

    return np.expand_dims(np.array(vec), 3), np.array(label_vec)


#print for a vector of extimation if they are correct
def ResultPrediction(extimation, real_label):
    str_extimation = np.array2string(extimation, None, 4)
    str_real_label = np.array2string(real_label, None, 4)
    if real_label[0] == 1:
        if extimation[0] > 0.5:
            return 'CORRECT  \n' + 'extimated ' + str_extimation + ' \n real_label ' + str_real_label
        else:
            return 'ERROR  \n' + 'extimated ' + str_extimation + ' \n real_label ' + str_real_label
    else:
        if extimation[1] > 0.5:
            return 'CORRECT  \n' + 'extimated ' + str_extimation + ' \n real_label ' + str_real_label
        else:
            return 'ERROR  \n' + 'extimated ' + str_extimation + ' \n real_label ' + str_real_label
