import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.loc_bbox_iou import xywh2xyxy
import random
import json

# 数据组织字典
mydata = {"train" : {}, "eval" : {}}
# 配置文件路径
config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs/config.json")
# 训练json文件路径
train_json = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data/annotations/instances_train2017.json")
# 验证json文件路径
eval_json = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data/annotations/instances_val2017.json")

# 读取文件
with open(config_path, 'r') as f:
    config = json.load(f)

train_ratio = config["train_ratio"]
eval_ratio = config["eval_ratio"]

with open(train_json, 'r') as train:
    train_data = json.load(train)

with open(eval_json, 'r') as eval:
    eval_data = json.load(eval)

# category类别ID到连续索引映射及其反向映射
category_id_to_class_index = {}
class_index_to_category_id = {}
class_index_2_class_name = {}

def init_category_id_and_class_index():
    data = eval_data["categories"]
    length = range(len(data))
    for i in length:
        class_index_to_category_id[i] = data[i]["id"]
        category_id_to_class_index[data[i]["id"]] = i
        class_index_2_class_name[i] = data[i]["name"]

def sort_data(data: dict, type: str, data_ratio: int) -> None:
    """
    input:
        data: train的json文件 或 eval的json文件
        type: "train" 或 "eval"
        data_ratio: 取原有数据集中的 ()% 用于train / eval 
    """
    num = int(len(data["images"]) * data_ratio)
    train_data_index = random.sample(range(len(data["images"])), num)

    # 读取 image_path 和 image_id 数据
    for i, _ in enumerate(train_data_index):
        # 取出第 i 张图片数据
        image = data["images"][i]
        mydata[type][i] = {}
        mydata[type][i]["bboxes"] = []
        mydata[type][i]["labels"] = []
        mydata[type][i]["image_path"] = os.path.join(f"data/{type}2017", image["file_name"])
        mydata[type][i]["image_id"] = image["id"]

def insert_annotations(data: dict, type: str) -> None:
    """
    input:
        data: train的json文件 或 eval的json文件
        type: "train" 或 "eval"
    """
    my_data = mydata[type]
    # 构建哈希表供查询 image_id -> index
    map_dict = {my_data[index]["image_id"] : index for index in range(len(my_data))}
    anno = data["annotations"]
    # 添加anno数据到mydata: anno -> image_id -> index -> mydata
    for i in range(len(anno)):
        if anno[i]["image_id"] in map_dict:
            # 对bboxes添加bbox信息并转换成XYXY格式
            my_data[map_dict[anno[i]["image_id"]]]["bboxes"].append(xywh2xyxy(anno[i]["bbox"]))
            # 对labels添加label信息并转换成连续classes_index形式
            my_data[map_dict[anno[i]["image_id"]]]["labels"].append(category_id_to_class_index[anno[i]["category_id"]])

init_category_id_and_class_index()

sort_data(train_data, "train", train_ratio)
sort_data(eval_data, "eval", eval_ratio)

insert_annotations(train_data, "train")
insert_annotations(eval_data, "eval")

del train_data, eval_data

if __name__ == "__main__":
    print("训练集长度：", len(mydata["train"]), "验证集长度：", len(mydata["eval"]))
    print("训练集json样本: ", json.dumps(mydata["train"], indent=4, skipkeys=2))
    print("验证集json样本: ", json.dumps(mydata["eval"], indent=4, skipkeys=2))
    print("类别数量：", len(class_index_to_category_id))