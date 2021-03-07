"""
创建RetinaNet 格式数据CSV

"""
import argparse
import csv
import os
import xml.etree.ElementTree as ET


def parse_args(args):
    parser = argparse.ArgumentParser(description="Split dataset.")

    parser.add_argument(
        "--data_dir", help="Root of original data.", required=True, type=str
    )

    return parser.parse_args(args)


# 数据文件夹里面应该包含train和val两个文件夹
sets = ["train", "val"]

class_dict = {}
class_index = 0


def read_annotation(annotation_path, image_path, anno_list, data_dir):
    global class_index, class_dict
    in_file = os.path.splitext(annotation_path)[0] + ".xml"

    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find("size")

    cls_list = []
    bbox_list = []

    for obj in root.iter("object"):
        cls = obj.find("name").text

        if cls not in class_dict:
            class_dict[cls] = class_index
            cls_list.append(class_index)
            class_index += 1
        else:
            cls_list.append(class_dict[cls])

        xmlbox = obj.find("bndbox")
        x1 = int(xmlbox.find("xmin").text)
        x2 = int(xmlbox.find("xmax").text)
        y1 = int(xmlbox.find("ymin").text)
        y2 = int(xmlbox.find("ymax").text)

        bbox_list.append([x1, y1, x2, y2])

        rel_path = os.path.relpath(image_path, data_dir)
        anno_list.append([rel_path, str(x1), str(y1), str(x2), str(y2), cls])

    return cls_list, bbox_list, anno_list


def process(data_dir):
    for data_set in sets:
        data_set_dir = os.path.join(data_dir, data_set)
        image_dir = os.path.join(data_set_dir, 'image')
        annotation_dir = os.path.join(data_set_dir, 'annotation')

        csv_path = os.path.join(data_dir, "%s_data.csv" % data_set)

        image_name_list = list(
            filter(
                lambda x: not (x.startswith(".") or x.endswith("xml")),
                os.listdir(image_dir),
            )
        )
        anno_list = []
        for image_name in image_name_list:
            image_path = os.path.join(image_dir, image_name)
            annotation_path = os.path.join(annotation_dir, image_name)
            print("Currently processing: %s" % image_path)
            cls_list, bbox_list, anno_list = read_annotation(annotation_path, image_path, anno_list, data_dir)
            print("%s Done!" % image_path)
        with open(csv_path, "w", newline='') as f:
            writer = csv.writer(f)
            for row in anno_list:
                writer.writerow(row)

    class_csv_path = os.path.join(data_dir, "class.csv")

    with open(class_csv_path, "w", newline='') as f:
        writer = csv.writer(f)
        for k, v in class_dict.items():
            writer.writerow([k, str(v)])
