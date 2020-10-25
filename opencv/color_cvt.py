import matplotlib.pyplot as plt
import cv2
import numpy as np


def chromakey(img,rgb="G"):
  color = 1
  if rgb =="R":
    color = 0
  elif rgb =="B":
    color = 2
  chromakey_img = img.copy()
  black_index = ((chromakey_img[:,:,0]==0)&(chromakey_img[:,:,1]==0)&(chromakey_img[:,:,2]==0))
  chromakey_img[black_index,color]=255
  plt.imshow(chromakey_img)
  return chromakey_img,black_index


def cut_roi(img,xy=[200, 400, 400, 500],thickness=2):
  # xy ⇒[x,x_h,y,y_h]
  plt.figure(figsize=(10,10))
  plt.subplot(1,3,1)
  #(x,y)(x_h,y_h)
  plt.imshow(cv2.rectangle(img.copy(), (xy[0],   xy[2]), (xy[1],  xy[3]), (255, 0, 0),thickness=thickness))
  roi = img[xy[2]:xy[3],xy[0]:xy[1],:]
  plt.subplot(1,3,2)
  plt.imshow(roi)
  rgb_list = [roi[:,:,0],roi[:,:,1],roi[:,:,2]]
  median_list = [int(np.median(rgb_list[0])),int(np.median(rgb_list[1])),int(np.median(rgb_list[2]))]
  print("r_median",median_list[0],"g_median",median_list[1],"b_median",median_list[2])
  print(roi.shape,rgb_list[0].shape)

  #追加処理_黒部分を除いたmedianの取得
  plt.subplot(1,3,3)
  chromakey_roi,black_index_roi = chromakey(roi,rgb="G")
  #black_indexの反転の中央値の整数
  r_median_2 = int(np.median(rgb_list[0][np.logical_not(black_index_roi)]))
  g_median_2 = int(np.median(rgb_list[1][np.logical_not(black_index_roi)]))
  b_median_2 = int(np.median(rgb_list[2][np.logical_not(black_index_roi)]))
  median2_list = [r_median_2,g_median_2,b_median_2]
  print("median2_list", median2_list)
  return roi, rgb_list, median_list, xy,median2_list



def roi_cvt(roi,rgb_list, median_list,plus_list, r_up=55,r_down=55,g_up=55,g_down=55,b_up=55,b_down=55):
  r,g,b = rgb_list[0],rgb_list[1],rgb_list[2]
  new_roi = roi.copy()
  r_up = r_up 
  r_down = r_down
  g_up = g_up
  g_down = g_down
  b_up = b_up
  b_down = b_down

  index = (
  ((r <= median_list[0]+r_up)&(r >= median_list[0] - r_down))
  &((g <= median_list[1]+g_up)&(g >= median_list[1] - g_down))
  &((b <= median_list[2]+b_up)&(b >= median_list[2] - b_down))
  )

  new_roi = new_roi.astype("int")
  for i in range(3):
    new_roi[index,i] +=plus_list[i]
  new_roi = new_roi.astype("uint8")
  plt.imshow(new_roi)
  return new_roi



def roi_to_img(img,new_roi,xy):
  new_img = img.copy()
  new_img[xy[2]:xy[3],xy[0]:xy[1],:] = new_roi
  plt.imshow(new_img)
  return new_img


def color_plot(color_list):
  r,g,b = color_list[0],color_list[1],color_list[2],
  temp = np.array([r,g,b,r,g,b,r,g,b,r,g,b]).reshape(2,2,3)
  plt.title("R{}_G{}_B{}".format(r,g,b))
  plt.imshow(temp)

def median_plot_2(median_list, r_up,r_down,g_up,g_down, b_up,b_down):
  r_list = [median_list[0],median_list[0]+r_up,median_list[0]-r_down]
  g_list = [median_list[1],median_list[1]+g_up,median_list[1]-g_down]
  b_list = [median_list[2],median_list[2]+b_up,median_list[2]-b_down]
  plt.figure(figsize=(30,10))
  counter = 0
  for r in r_list:
    for g in g_list:
      for b in b_list:
        counter += 1
        temp = np.array([r,g,b,r,g,b,r,g,b,r,g,b]).reshape(2,2,3)
        plt.subplot(3,9,counter)
        plt.title("R{}_G{}_B{}".format(r,g,b))
        plt.imshow(temp)

def dropper(median_list,drop_color):
  plus_list = []
  for i in range(3):
    if median_list[i] >= drop_color[i]:
      plus = median_list[i] - drop_color[i]
      plus *= -1
    else:
      print("else")
      plus = drop_color[i] - median_list[i] 
    plus_list.append(plus)
  print("plus_list", plus_list)
  return plus_list

def layer_to_img(img,alpha_ch,layers,output_path):
  result = img.copy()
  for values in layers.values():
    result = roi_to_img(result,values[0],values[1])
  plt.imshow(result)
  result = cv2.cvtColor(result,cv2.COLOR_RGB2BGR)
  result = np.concatenate([result, alpha_ch], axis=2)
  print("レイヤーの枚数",len(layers))
  print(result.shape)
  cv2.imwrite(output_path,result)
  return result

# from opencv.color_cvt import chromakey, cut_roi, roi_cvt, roi_to_img,color_plot,median_plot_2,dropper, layer_to_img