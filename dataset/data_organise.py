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
# 配置文件路径
config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs/config.json")
# 训练json文件路径
train_json = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data/annotations/instances_train2017.json")
# 验证json文件路径
eval_json = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data/annotations/instances_val2017.json")

# 读取文件
with open(config_path, 'r') as f:
    config = json.load(f)

data_ratio = config["data_ratio"]

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
        mydata[type][i]["label"] = []
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
            my_data[map_dict[anno[i]["image_id"]]]["label"].append(anno[i]["bbox"] + [anno[i]["category_id"]])

def category_and_id(train_data: dict, eval_data: dict) -> None:
    """
        组织 类别id 与 类别名称 互查的哈希表, 构建类别查询表
    """
    train = train_data["categories"]
    for i in range(len(train)):
        data = train[i]
        category_2_id[data["name"]] = data["id"]
        id_2_category[data["id"]] = data["name"]
        classes.add(data["name"])

    eval = eval_data["categories"]
    for i in range(len(eval)):
        data = eval[i]
        category_2_id[data["name"]] = data["id"]
        id_2_category[data["id"]] = data["name"]
        classes.add(data["name"])

sort_data(train_data, "train", data_ratio)
sort_data(eval_data, "eval", data_ratio)

insert_annotations(train_data, "train")
insert_annotations(eval_data, "eval")

category_and_id(train_data, eval_data)

if __name__ == "__main__":
    print("训练集长度：", len(mydata["train"]), "验证集长度：", len(mydata["eval"]))
    print("category to id: ", category_2_id, "\n")
    print("id to category: ", id_2_category, "\n")
    print("类别数量：", len(classes))