#seedの固定
import numpy as np
import matplotlib.pyplot as plt
from pylab import rcParams


def aug_plot(dataset_show,row=2,col=5):
    rcParams['figure.figsize'] = 20,10
    for k in range(row):
        f, axarr = plt.subplots(1,5)
        for p in range(col):
            idx = np.random.randint(0, len(dataset_show))
            img, label = dataset_show[idx]
            img = img.transpose(0, 1).transpose(1,2).squeeze()
            # img = img.numpy()
            print(img.shape,img.dtype)
            print(img.min(),img.max())
            axarr[p].imshow(img)
            # axarr[p].set_title(str(label))

def dataset_instance(df,imfolder,phase="train", image_size=256, aug=True, sample=False):
    if sample==True:
        df=df.sample(500)
    from .dataset import Albu_Dataset
    from .transform import Albu_Transform
    instance = Albu_Dataset(df=df, 
                            imfolder=imfolder, 
                            phase=phase, 
                            transforms=Albu_Transform(image_size=image_size),
                            aug=aug)
    return instance

def aug_plot_auto(df,imfolder):
    dataset_show = dataset_instance(df,imfolder,phase="train", image_size=256, aug=True, sample=True)
    rcParams['figure.figsize'] = 20,10
    for k in range(2):
        f, axarr = plt.subplots(1,5)
        for p in range(5):
            idx = np.random.randint(0, len(dataset_show))
            img, label = dataset_show[idx]
            img = img.transpose(0, 1).transpose(1,2).squeeze()
            print(img.shape,img.dtype)
            print(img.min(),img.max())
            axarr[p].imshow(img)

#########使用方法##########
# import aug_image_plot
# aug_image_plot.aug_plot(dataset_show,row=2,col=5)