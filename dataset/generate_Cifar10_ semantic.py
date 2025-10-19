import numpy as np
import os
import sys
import random
import torch
import torchvision
import torchvision.transforms as transforms
from utils.dataset_utils import check, separate_data, split_data, save_file, sample_proxy

random.seed(1)
np.random.seed(1)
num_clients = 10
dir_path = "Cifar10/"
green_car_test=[38658,47001,3378,3678,32941]
green_car_train=[561,389,874,1605,4528,9744,21422,19500,19165,22984,34287,34385,36005,37365,37533,38735
,39824,40138,41336,41861,47026,48003,48030,49163,49588]


# Allocate data to users
def generate_dataset(dir_path, num_clients, semantic):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # Setup directory for train/test data
    train_path = dir_path + "train/"
    test_path = dir_path + "test/"


    # Get Cifar10 data
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(
        root=dir_path + "rawdata", train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(
        root=dir_path + "rawdata", train=False, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=len(trainset.data), shuffle=False)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=len(testset.data), shuffle=False)

    for _, train_data in enumerate(trainloader, 0):
        trainset.data, trainset.targets = train_data
    for _, test_data in enumerate(testloader, 0):
        testset.data, testset.targets = test_data

    dataset_image = []
    dataset_label = []


    num_classes = len(set(dataset_label))
    print(f'Number of classes: {num_classes}')

    
    train_data, test_data = [], []
    train_data.append({'x': trainset.data[green_car_train], 'y': trainset.targets[green_car_train]})
    test_data.append({'x': testset.data[green_car_test], 'y': testset.targets[green_car_test]})

    
    with open(train_path + str(semantic) + '.npz', 'wb') as f:
        np.savez_compressed(f, data=train_data)
    with open(test_path + str(semantic) + '.npz', 'wb') as f:
        np.savez_compressed(f, data=test_data)

if __name__ == "__main__":
    semantic = sys.argv[1]
    niid = True if sys.argv[2] == "noniid" else False
    if(niid):
        dir_path="Cifar10_noniid/"

    generate_dataset(dir_path, num_clients, semantic)