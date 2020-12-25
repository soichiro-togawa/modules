# jsonでの扱い方
import json
import pandas as pd
import os

def load_json(json_path=None,keys=True):
    #物体検出とセマンティックセグメンテーションのアノテーションが入っている
    with open(json_path) as f:
        json_data = json.load(f)
    
    if keys==True:
      print(len(json_data))
      for i in json_data.keys():
          print(i)
      #アノテーションのDF化
      df = pd.DataFrame(json_data["annotations"])
      return json_data,df
    return json_data


#cocoAPI
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
def coco_info(json_path):
    #インスタンスの作成
    coco_api = COCO(json_path)

    #json内に収録されているcategoryのみ取得(ここでは80)
    cats = coco_api.loadCats(coco_api.getCatIds())

    #イメージ固有のid、5000個を取得(file名とは別)
    image_ids = coco_api.getImgIds()

    #idからイメージの情報(file名含む)を取得
    image_info = coco_api.loadImgs(image_ids)
    #インデックス0のイメージのfile名を取得
    print(image_info[0]["file_name"])
    return coco_api, cats, image_ids, image_info

  
def get_part(temp_path, json_path):
    #インスタンスの作成
    coco_api = COCO(json_path)
    image_ids = coco_api.getImgIds()
    image_info = coco_api.loadImgs(image_ids)

    #file名から固有のidを取得する
    file_path = os.listdir(temp_path)
    
    
    #パスの中にある画像の情報のみを取得
    index = []
    new_image_info=[]
    new_image_id=[]
    for i,info in enumerate(image_info):
        if info["file_name"] in file_path:
            index.append(i)
            new_image_info.append(info)
            new_image_id.append(info["id"])
    print(len(index))
    print(len(new_image_info))
    print(len(new_image_id))
    return new_image_info, new_image_id

def write_json(new_image_info, json_path, save_path):
    with open(json_path) as f:
        json_data = json.load(f)

    json_data["images"] = new_image_info
    print(len(json_data["images"]))
    with open(save_path, 'w') as f:
        json.dump(json_data, f, indent=4)


# from verification.coco_utils import load_json