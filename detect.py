import os
import subprocess
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras_retinanet import models
from keras_retinanet.utils.colors import label_color
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption

from data_prep import gen_csv

data_dir = 'data\\sugarbeet'
train_csv_path = os.path.join(data_dir, 'train_data.csv')
val_csv_path = os.path.join(data_dir, 'val_data.csv')
class_csv_path = os.path.join(data_dir, 'class.csv')

result_model_path = 'models/detection'
trained_model_path = 'models/detection/resnet50_csv_13.h5'
inference_model_path = trained_model_path.split(".")[0] + "_inference.h5"


def train():
    # 训练模型
    train_command_line = ['retinanet-train', '--snapshot-path', 'models', '--tensorboard-dir', 'tensorboard_dir',
                          '--epochs', '20',
                          '--steps', '500',
                          'csv',
                          train_csv_path, class_csv_path,
                          '--val-annotations', val_csv_path]
    print('\n开始训练...')
    process = subprocess.Popen(train_command_line)
    process.wait()
    if process.returncode != 0:
        print(f"训练失败，失败码：{process.returncode}")
    else:
        print('训练完成!')


def convert_model():
    convert_command = ['retinanet-convert-model', '--no-class-specific-filter',
                       trained_model_path, inference_model_path]
    print('开始模型转换...')
    process = subprocess.Popen(convert_command)
    process.wait()
    print(process.errors)
    if process.returncode != 0:
        print(f"模型转换失败，失败码：{process.returncode}")
    else:
        print('模型转换成功!')


def visual_single_test(model, test_path):
    # load labels
    df = pd.read_csv(class_csv_path, header=None)
    labels_to_names = df[0].values.tolist()

    # load image
    image = read_image_bgr(test_path)

    # copy to draw on
    draw = image.copy()
    draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

    # preprocess image for network
    image = preprocess_image(image)
    image, scale = resize_image(image)

    # process image
    start = time.time()
    boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
    print("processing time: ", time.time() - start)

    # correct for image scale
    boxes /= scale

    # visualize detections
    for box, score, label in zip(boxes[0], scores[0], labels[0]):
        # scores are sorted so we can break
        if score < 0.5:
            break

        color = label_color(label)

        b = box.astype(int)
        draw_box(draw, b, color=color)
        caption = "{} {:.3f}".format(labels_to_names[label], score)
        draw_caption(draw, b, caption)

    plt.figure(figsize=(15, 15))
    plt.axis('off')
    plt.imshow(draw)
    plt.show()

    # 保存结果图
    cv2.imwrite("result.jpg", draw)


def main():
    # step1. 生成CSV格式数据集和标注(一般地，需要先划分数据集，这边已经划分好了，直接处理CSV)
    if os.path.exists(class_csv_path):
        print("csv files existed, skip step1")
    else:
        gen_csv.process(data_dir)

    # step2. 训练模型
    if os.path.exists(trained_model_path):
        print("model files existed, skip step2")
    else:
        train()

    # step3. convert trained model to inference_model
    if os.path.exists(inference_model_path):
        print("inference model files existed, skip step3")
    else:
        convert_model()

    # step4. 加载模型
    model = models.load_model(inference_model_path, backbone_name='resnet50')
    print(model.summary())

    # step5. 显示/测试单张效果
    test_path = os.path.join(data_dir, 'val\\image\\X2-30-1.png')
    visual_single_test(model, test_path)


if __name__ == '__main__':
    main()
