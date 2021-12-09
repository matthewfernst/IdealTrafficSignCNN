import json
import cv2
import os
import pickle
import numpy as np

TEST = 10544
TRAIN = 36589
VAL = 5320
PATH_IMAGES = "Mapillary/images"
PATH_ANNO = "Mapillary/mtsd_v2_fully_annotated/annotations"

MAX_SIZE = (256, 256)

classes = ["Stop Sign", "No U Turn", "Speed Limit", "Yield", "School Zone"]


def load_data(f):
    pick = open(f, 'rb')
    data = pickle.load(pick)
    pick.close()
    features = []
    labels = []
    for img, label in data:
        features.append(img)
        labels.append(classes.index(label))

    feature = np.array(features, dtype=np.float32)
    feature = feature / 255.
    labels = np.array(labels)
    return [feature, labels]


def generate_classes():
    classes = []
    count = 0
    anno_files = os.listdir(PATH_ANNO)
    for anno_file in anno_files:
        if count % 50 == 0:
            print(f"On count: {count}/{int((VAL + TRAIN + TEST))}")
        anno_json = json.load(open(f"{PATH_ANNO}/{anno_file}"))
        for sign in anno_json["objects"]:
            label = sign["label"]
            classes.append(label)
        count += 1
    write("classesFinal.txt", np.unique(classes))
    return np.unique(classes)


def get_classes():
    f = open("classesFinal.txt", 'r')
    return [line[:-1] for line in f if len(line) != 0]


def get_index(classes, label):
    try:
        return classes.index(label)
    except:
        classes.append(label)
        return len(classes) - 1


def write(filename, list):
    MyFile = open(filename, 'w')
    for element in list:
        MyFile.write(element)
        MyFile.write('\n')
    MyFile.close()


def make_data(f, percentage=.02, start_index=0, ):
    data, classes = make_data_helper()
    pik = open(f, 'wb')
    pickle.dump(data, pik)
    pik.close()


def make_data_helper():
    data = []
    number = int((VAL + TRAIN + TEST))
    classes = ["Stop Sign", "No U Turn", "Speed Limit", "Yield", "School Zone"]
    count = 0
    anno_files = os.listdir(PATH_ANNO)

    for anno_file in anno_files:
        if count % 20 == 0:
            print(f"On count: {count}/{number}")
        count += 1
        image_file = f"{PATH_IMAGES}/{anno_file[:-4]}jpg"
        img = cv2.imread(image_file)
        anno_json = json.load(open(f"{PATH_ANNO}/{anno_file}"))
        for sign in anno_json["objects"]:

            label = sign["label"]
            if label in ["regulatory--stop--g1",
                         "regulatory--stop--g10",
                         "regulatory--stop--g2",
                         "regulatory--no-u-turn--g1",
                         "regulatory--no-u-turn--g2",
                         "regulatory--no-u-turn--g3",
                         "regulatory--maximum-speed-limit-70--g1",
                         "regulatory--maximum-speed-limit-15--g1",
                         "regulatory--maximum-speed-limit-20--g1",
                         "regulatory--maximum-speed-limit-25--g1",
                         "regulatory--maximum-speed-limit-25--g2",
                         "regulatory--maximum-speed-limit-30--g1",
                         "regulatory--maximum-speed-limit-30--g3",
                         "regulatory--yield--g1",
                         "regulatory--left-turn-yield-on-green--g1",
                         "regulatory--turning-vehicles-yield-to-pedestrians--g1",

                         "warning--school-zone--g2"]:
                if label in ["regulatory--stop--g1", "regulatory--stop--g10", "regulatory--stop--g2"]:
                    label = "Stop Sign"
                if label in ["regulatory--no-u-turn--g1", "regulatory--no-u-turn--g2", "regulatory--no-u-turn--g3"]:
                    label = "No U Turn"
                if label in ["regulatory--maximum-speed-limit-70--g1",
                             "regulatory--maximum-speed-limit-15--g1",
                             "regulatory--maximum-speed-limit-20--g1",
                             "regulatory--maximum-speed-limit-25--g1",
                             "regulatory--maximum-speed-limit-25--g2",
                             "regulatory--maximum-speed-limit-30--g1",
                             "regulatory--maximum-speed-limit-30--g3"]:
                    label = "Speed Limit"
                if label in ["regulatory--yield--g1", "regulatory--left-turn-yield-on-green--g1",
                             "regulatory--turning-vehicles-yield-to-pedestrians--g1", ]:
                    label = "Yield"
                if label in ["warning--school-zone--g2"]:
                    label = "School Zone"

                r = sign["bbox"]
                imCrop = img[int(r["ymin"]):int(r["ymax"]), int(r["xmin"]):int(r["xmax"])]
                imCrop = cv2.resize(imCrop, MAX_SIZE)

                data.append([imCrop, label])

    return data, classes


def generate_files():
    for i in range(20):
        print(f"One file: data/streetSignData{i}.pickle")
        make_data(f"data/streetSignData{i}.pickle", .05, int((VAL + TRAIN + TEST) * .05 * (i + 1)))
