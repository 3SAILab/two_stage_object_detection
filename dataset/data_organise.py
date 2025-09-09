from utils.loc_bbox_iou import xywh2xyxy
import random
import json
import os

# 数据组织字典
mydata = {"train" : {}, "eval" : {}}
# 转换字典
category_2_id = {}
id_2_category = {}
# 类别查询表
classes = set()
# COCO类别ID到连续索引的映射
coco_id_to_class_index = {}
class_index_to_coco_id = {}
class_index = 0
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
            my_data[map_dict[anno[i]["image_id"]]]["bboxes"].append(xywh2xyxy(anno[i]["bbox"]))
            # 将COCO类别ID转换为连续索引
            coco_category_id = anno[i]["category_id"]
            if coco_category_id in coco_id_to_class_index:
                class_idx = coco_id_to_class_index[coco_category_id]
                my_data[map_dict[anno[i]["image_id"]]]["labels"].append(class_idx)

def category_and_id(train_data: dict, eval_data: dict) -> None:
    """
        组织 类别id 与 类别名称 互查的哈希表, 构建类别查询表
    """
    global class_index, coco_id_to_class_index, class_index_to_coco_id
    
    # 收集所有类别
    all_categories = {}
    train_categories = {cat["id"]: cat for cat in train_data["categories"]}
    eval_categories = {cat["id"]: cat for cat in eval_data["categories"]}
    all_categories.update(train_categories)
    all_categories.update(eval_categories)
    
    # 为每个COCO类别ID分配连续的索引
    for coco_id, cat_data in all_categories.items():
        if coco_id not in coco_id_to_class_index:
            coco_id_to_class_index[coco_id] = class_index
            class_index_to_coco_id[class_index] = coco_id
            category_2_id[cat_data["name"]] = class_index
            id_2_category[class_index] = cat_data["name"]
            classes.add(cat_data["name"])
            class_index += 1

sort_data(train_data, "train", train_ratio)
sort_data(eval_data, "eval", eval_ratio)

category_and_id(train_data, eval_data)

insert_annotations(train_data, "train")
insert_annotations(eval_data, "eval")

if __name__ == "__main__":
    print("训练集长度：", len(mydata["train"]), "验证集长度：", len(mydata["eval"]))
    print("训练集json样本: ", json.dumps(mydata["train"], indent=4))
    print("验证集json样本: ", json.dumps(mydata["eval"], indent=4))
    print("类别数量：", len(classes))