import sys
import os
from random import random

"""
#######################
Description
#######################

This script basically allows you to build a new set of image sets built from an egohands formatted setup 
in the darknet style. It goes through and builds individual training and test sets (files) which link to the 
image filenames / labels.

To run use: python generate_egohandsdata.py ego-model-labels.txt

Based upon https://github.com/qfgaohao/pytorch-ssd/ but also adapted to meet style for 'https://github.com/ultralytics/yolov3/'
"""

def main(filename):
    # ratio to divide up the images
    train = 0.9
    test = 0.1
    if (train + test) != 1.0:
        print("probabilities must equal 1")
        exit()

    # get the labels
    imgnames = []

    with open(filename, 'r') as labelfile:
        label_string = ""
        for line in labelfile:
                label_string += line.rstrip()

    labels = label_string.split(',')
    labels  = [elem.replace(" ", "") for elem in labels]

    # get image names
    for filename in os.listdir("egohandsDarknet/images"):
        if filename.endswith(".jpg"):
            img = filename.rstrip('.jpg')
            imgnames.append(img)

    print("Labels:", labels, "image count: ", len(imgnames))



    # divvy up the images to the different sets
    sampler = imgnames.copy()
    train_list = []
    test_list = []

    while len(sampler) > 0:
        dice = random()
        elem = sampler.pop()
        # Randomly select set for image
        if dice <= test:
            test_list.append(elem)
        else:
            train_list.append(elem)

    print("Training set:", len(train_list), "test set:", len(test_list))

    # create the dataset files
    with open("egohandsDarknet/imagesets/train.txt", 'w+') as outfile:
        for name in train_list:
            outfile.write("data/egohands/egohandsDarknet/images/" + name + ".jpg" + "\n")
    with open("egohandsDarknet/imagesets/test.txt", 'w+') as outfile:
        for name in test_list:
            outfile.write("data/egohands/egohandsDarknet/images/" + name + ".jpg" + "\n")

def create_folder(foldername):
    if os.path.exists(foldername):
        print('folder already exists:', foldername)
    else:
        os.makedirs(foldername)

if __name__=='__main__':
    if len(sys.argv) < 2:
        print("usage: python generate_egohandsdata.py <labelfile>")
        exit()
    main(sys.argv[1])