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
dir_path = "Cifar10_semantic/"
green_car_test=[38658,47001,3378,3678,32941]
green_car_train=[561,389,874,1605,4528,9744,21422,19500,19165,22984,34287,34385,36005,37365,37533,38735
,39824,40138,41336,41861,47026,48003,48030,49163,49588]


# Allocate data to users
def generate_dataset(dir_path, num_clients, niid, balance, partition):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    config_path = dir_path + "config.json"
    train_path = dir_path + "train/"
    test_path = dir_path + "test/"
    proxy_path = dir_path + "proxy/"

    if check(config_path, train_path, test_path, num_clients, niid, balance, partition):
        return

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
    
    train_images=trainset.data.cpu().detach().numpy()
    train_labels=trainset.targets.cpu().detach().numpy()
    test_images=testset.data.cpu().detach().numpy()
    test_labels=testset.targets.cpu().detach().numpy()

    train_images_green_car=train_images[green_car_train]
    train_labels_green_car=np.array([2 for _ in range(len(train_images_green_car))])
    test_images_green_car=train_images[green_car_test]
    test_labels_green_car=np.array([2 for _ in range(len(test_images_green_car))])

    train_images=np.array([image for idx, image in enumerate(train_images)
                  if idx not in (green_car_train) and idx not in (green_car_test)])
    train_labels=np.array([label for idx, label in enumerate(train_labels)
                  if idx not in (green_car_train) and idx not in (green_car_test)])


    dataset_image.extend(train_images)
    dataset_image.extend(test_images)
    dataset_label.extend(train_labels)
    dataset_label.extend(test_labels)
    dataset_image = np.array(dataset_image)
    dataset_label = np.array(dataset_label)

    num_classes = len(set(dataset_label))
    print(f'Number of classes: {num_classes}')

    X, y, statistic = separate_data((dataset_image, dataset_label), num_clients, num_classes,
                                    niid, balance, partition, class_per_client=2)
    train_data, test_data = split_data(X, y)
    proxy_data = sample_proxy(list(dataset_image))

    np.random.seed(42)
    sampled_indices = np.random.choice(
        a=len(train_data[0]['x']),       
        size=1000,    
        replace=False 
    )

    train_data[0]['x']=np.concatenate((train_data[0]['x'][sampled_indices], np.repeat(train_images_green_car, 45, axis=0)))
    train_data[0]['y']=np.concatenate((train_data[0]['y'][sampled_indices], np.repeat(train_labels_green_car, 45, axis=0)))
    test_data[0]['x']=test_images_green_car
    test_data[0]['y']=test_labels_green_car

    save_file(config_path, train_path, test_path, train_data, test_data, num_clients, num_classes,
              statistic, proxy_path, proxy_data, niid, balance, partition)


if __name__ == "__main__":
    niid = True if sys.argv[1] == "noniid" else False
    balance = True if sys.argv[2] == "balance" else False
    partition = sys.argv[3] if sys.argv[3] != "-" else None
    if(niid):
        dir_path="Cifar10_semantic_"+partition+"/"

    generate_dataset(dir_path, num_clients, niid, balance, partition)