import dlib
import numpy as np
import matplotlib.pyplot as plt
import os
from skimage import io
import matplotlib.patches as patches
from skimage.transform import resize

# appies the face extraction pipeline to a single image
def FaceExtractionPipelineImage(path):

    # Create a HOG face detector using the built-in dlib class
    face_detector = dlib.get_frontal_face_detector()

    # Load the image into an array
    image = io.imread(path)

    # Run the HOG face detector on the image data. The result will be the bounding boxes of the faces in our image.
    detected_faces = face_detector(image, 1)

    # Bypass, if more then one face is found I just reject the face
    if len(detected_faces) == 1:

        # Loop through each face we found in the image
        for i, face_rect in enumerate(detected_faces):
            im = np.array(image)

            # Bounding box regularization
            face_rect_bottom = 0 if face_rect.bottom() < 0 else face_rect.bottom()
            face_rect_left = 0 if face_rect.left() < 0 else face_rect.left()
            face_rect_right = 0 if face_rect.right() < 0 else face_rect.right()
            face_rect_top = 0 if face_rect.top() < 0 else face_rect.top()

            # show the bounding box of the face
            # fig, ax = plt.subplots(1)
            # ax.imshow(im)
            # rect = patches.Rectangle((face_rect_left, face_rect_top), face_rect_bottom - face_rect_top, (face_rect_right - face_rect_left), edgecolor='r', linewidth=1, facecolor='none')
            # ax.add_patch(rect)
            # plt.show()

            # Crop
            cropped_im = im[face_rect_top:face_rect_bottom, face_rect_left:face_rect_right]

            # Resize
            resized_im = resize(cropped_im, (80, 80))

            # TO-DO :
            # rotate the image in order to put eyes and mouth at center

            return resized_im


# outputs the results of the pipeline to all the images starting from the dataset_root_path
def TryThePipeline(dataset_root_path):
    folders = os.listdir(dataset_root_path)

    for i in folders: # scan all the images of that folder
        img = FaceExtractionPipelineImage(dataset_root_path + '/' + i)
        if img is not None:
            plt.imshow(img, 'gray')
            plt.show()


#TryThePipeline('/home/giovanni/Immagini/Webcam')