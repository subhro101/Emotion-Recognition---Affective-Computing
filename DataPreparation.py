import cv2
import os
import random

# Weights for the opencv neural net based face detector
# We found that this was much better than the haar cascades based approach
CV_FACE_WEIGHTS = r'opencv_face_detector_uint8.pb'
CV_FACE_MODEL = r'opencv_face_detector.pbtxt'
CV_FACE_NET = cv2.dnn.readNetFromTensorflow(CV_FACE_WEIGHTS, CV_FACE_MODEL)

IMAGE_SIZE = 128

# Input and output directories
SRC_DIR = r'/data/scanavan1/AffectiveComputing/Project2/pain_classification'
DEST_DIR = r'/home/CAP4628-2/project2_2/processed_data_aug'


# Take one image and crop out the face
def crop_face(img):
    # Get results from nerual net
    blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), (104.0, 177.0, 123.0))
    CV_FACE_NET.setInput(blob)
    out = CV_FACE_NET.forward()

    # Find the detection with the highest confidence level
    # This ensures we get the actual face, if there are false positives
    best = None
    for i in range(out.shape[2]):
        confidence = out[0, 0, i, 2]
        left = int(out[0, 0, i, 3] * img.shape[1])
        top = int(out[0, 0, i, 4] * img.shape[0])
        right = int(out[0, 0, i, 5] * img.shape[1])
        bottom = int(out[0, 0, i, 6] * img.shape[0])

        if best is None or confidence > best[0]:
            best = (confidence, left, top, right, bottom)

    # Crop image
    (confidence, left, top, right, bottom) = best
    return img[top:bottom, left:right]


# Resize the image to the correct width/height for our network
def resize(img):
    return cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))


# Randomly flip or dont flip the image horizontally
def random_horizontal_flip(img):
    if random.choice([True, False]):
        return img[:, ::-1]
    return img


# Randomly alter the brightness/contrast of the image
def random_brightness_contrast_shift(img):
    return cv2.addWeighted(img, random.uniform(0.75, 1.25), img, 0, random.randint(-25, 25))


# Randomly do small rotations on the image
def random_rotation(img):
    rows, cols, channels = img.shape
    M = cv2.getRotationMatrix2D((cols/2,rows/2), random.randint(-10, 10), 1)
    return cv2.warpAffine(img, M, (rows, cols))


# Loop through all the data and prepare it
def prepare_data():
    for path, subdirs, files in os.walk(SRC_DIR):
        for name in files:
            file_path = os.path.join(path, name)
            out_path = os.path.join(DEST_DIR, os.path.relpath(file_path, SRC_DIR))

            # Crop and resize
            img = cv2.imread(file_path)
            img = crop_face(img)
            img = resize(img)

            # Write cropped image
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            cv2.imwrite(out_path, img)

            # We commented this out for our actual results
            # Saves an augmented version of the image if this is the test set
            # We disabled since it reduces our accuracy
            if "Training" in file_path:
                aug_img = random_horizontal_flip(img)
                aug_img = random_brightness_contrast_shift(aug_img)
                aug_img = random_rotation(aug_img)
                cv2.imwrite(out_path.replace(".jpg", ".aug.jpg"), aug_img)


# Run script
if __name__ == '__main__':
    prepare_data()
