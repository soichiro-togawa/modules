import cv2,os

image_path = "/content/drive/MyDrive/00Colab Notebooks/07Datasets/COCO/cocomax/"
file_path = os.listdir(image_path)
numpy_list=[cv2.imread(image_path + i) for i in file_path]

resized_list = [cv2.resize(img, dsize=(848, 480)) for img in numpy_list]

out_path = "/content/drive/MyDrive/00Colab Notebooks/07Datasets/COCO/cocomax_resized/"
for i,j in enumerate(resized_list):
  cv2.imwrite(out_path + file_path[i], j)