import dlib
import matplotlib.pyplot as plt
import os
from face_utils import FaceAligner
import skimage.io
import time

# path of the preprocessed dataset
PREPROCESSED_IMAGES_FOLDER_PATH = "3_preprocessed_"
_instances = {}

face_detector = None
predictor = None
fa = None

class SingletonPipeline(object):

    def __new__(cls, *args, **kw):
        if not cls in _instances:
            instance = super().__new__(cls)
            _instances[cls] = instance

            instance.face_detector = dlib.get_frontal_face_detector()
            instance.predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
            instance.fa = FaceAligner(instance.predictor, desiredFaceWidth=80)

        return _instances[cls]


    # appies the face extraction pipeline to a single image
    #
    # REQUIRES: a general image, whatever RGB or GrayScale, in any resultion and acquired from any library
    # RETURNS: the extracted face from the image, cropped and with the eyes aligned, if any
    def FaceExtractionPipelineImage(self, image):
        start_time = time.time()
        # Create a HOG face detector using the built-in dlib class

        # Run the HOG face detector on the image data. The result will be the bounding boxes of the faces in our image.
        detected_faces = self.face_detector(image, 1)

        # Bypass, if more then one face is found I just reject the face
        if len(detected_faces) == 1:

            # Loop through each face we found in the image
            for i, face_rect in enumerate(detected_faces):


                face_aligned = self.fa.align(image, image, face_rect)

                # im = np.array(image)
                #
                # # Bounding box regularization
                # face_rect_bottom = 0 if face_rect.bottom() < 0 else face_rect.bottom()
                # face_rect_left = 0 if face_rect.left() < 0 else face_rect.left()
                # face_rect_right = 0 if face_rect.right() < 0 else face_rect.right()
                # face_rect_top = 0 if face_rect.top() < 0 else face_rect.top()
                #
                # # show the bounding box of the face
                # # fig, ax = plt.subplots(1)
                # # ax.imshow(im)
                # # rect = patches.Rectangle((face_rect_left, face_rect_top), face_rect_bottom - face_rect_top, (face_rect_right - face_rect_left), edgecolor='r', linewidth=1, facecolor='none')
                # # ax.add_patch(rect)
                # # plt.show()
                #
                # # Crop
                # cropped_im = im[face_rect_top:face_rect_bottom, face_rect_left:face_rect_right]
                #
                # # Resize
                # resized_im = resize(cropped_im, (80, 80))
                #
                # # bring in gray scale the images
                if len(face_aligned.shape) == 3:
                    face_aligned = skimage.color.rgb2gray(face_aligned)
                    face_aligned = face_aligned*255
                    face_aligned = face_aligned.astype('int')

                print("--- %s seconds ---" % (time.time() - start_time))

                return face_aligned


# outputs the results of the pipeline to all the images starting from the dataset_root_path
def TryThePipeline(dataset_root_path):
    folders = os.listdir(dataset_root_path)

    for i in folders: # scan all the images of that folder
        im = skimage.io.imread(dataset_root_path + '/' + i)
        img = SingletonPipeline().FaceExtractionPipelineImage(im)
        if img is not None:
            print(dataset_root_path + '/' + i)
            plt.imshow(img, 'gray')
            plt.show()
        else:
            print('error in: ' + dataset_root_path + '/' + i)

# TryThePipeline('/home/giovanni/Immagini/Webcam')
# ==========PREPROCESSING load data ================

def PreprocessImages(folder):
    preproc_folder = PREPROCESSED_IMAGES_FOLDER_PATH + folder
    if not os.path.isdir(preproc_folder):
        os.mkdir(preproc_folder)

    for f in os.listdir(folder):
        if not os.path.isdir(preproc_folder + "/" + f):
            os.mkdir(preproc_folder + "/" + f)
            for img in os.listdir(folder + "/" + f):
                image = skimage.io.imread(folder + "/" + f + '/' + img)
                preproc_img = SingletonPipeline.FaceExtractionPipelineImage(image)

                if preproc_img is not None:
                    skimage.io.imsave(preproc_folder + "/" + f + '/' + img, preproc_img)
                    print("Created: " + preproc_folder + "/" + f + '/' + img)
                else:
                    print("Image null: " + preproc_folder + "/" + f + '/' + img)
    else:
        print("Folder already created. Delete the old one and retry.")

#PreprocessImages("2_dataset test")
#PreprocessImages("1_dataset train")

