#transform
import torchtoolbox.transform as transforms
# !pip install albumentations==0.4.6
import albumentations as A


print("クラス一覧")
print("Albu_Transform","Torch_Transform")


class Albu_Transform():
    def __init__(self, image_size):
        self.data_transform = {
            'train_transform':A.Compose([
              A.Transpose(p=0.5),
              A.VerticalFlip(p=0.5),
              A.HorizontalFlip(p=0.5),
              A.RandomBrightness(limit=0.2, p=0.75),
              A.RandomContrast(limit=0.2, p=0.75),
              A.OneOf([
                  A.MotionBlur(blur_limit=5),
                  A.MedianBlur(blur_limit=5),
                  A.GaussianBlur(blur_limit=5),
                  A.GaussNoise(var_limit=(5.0, 30.0)),], p=0.7),
              A.OneOf([
                  A.OpticalDistortion(distort_limit=1.0),
                  A.GridDistortion(num_steps=5, distort_limit=1.),
                  A.ElasticTransform(alpha=3),], p=0.7),
              A.CLAHE(clip_limit=4.0, p=0.7),
              A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
              A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=0, p=0.85),
              A.Resize(image_size, image_size),
              A.Cutout(max_h_size=int(image_size * 0.375), max_w_size=int(image_size * 0.375), num_holes=1, p=0.7),    
              A.Normalize()
              ]),
            'test_transform': A.Compose([
              A.Resize(image_size, image_size),
              A.Normalize(),
              A.Resize(image_size, image_size)
              ])}
    def __call__(self, img, aug=True):
        #albではimage=+["image"]、torchivisionではimg
        #albでは(WHC)、torchivisionでは(CWH)
        # print("albバージョン")
        if aug == True:
          return self.data_transform["train_transform"](image=img)["image"]
        return self.data_transform["test_transform"](image=img)["image"]


#トーチビジョン
class Torch_Transform():
    def __init__(self, image_size):
        self.data_transform = {
    'train_transform':transforms.Compose([
    transforms.RandomResizedCrop(size=image_size, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    #albumentatinoのNormalizeと同値
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    ]),
    'test_transform': transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    ])}

    def __call__(self, img, aug=True):
        # print("torchバージョン")
        if aug == True:
          return self.data_transform["train_transform"](img)
        return self.data_transform["test_transform"](img)

