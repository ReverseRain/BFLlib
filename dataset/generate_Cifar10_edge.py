import numpy as np
import os
import sys
import random
import torch
import torchvision
import torchvision.transforms as transforms
import pickle
from utils.dataset_utils import check, separate_data, split_data, save_file, sample_proxy

random.seed(1)
np.random.seed(1)
num_clients = 10
dir_path = "Cifar10_edge/"


# Allocate data to users
def generate_dataset(dir_path, num_clients, niid, balance, partition):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # Setup directory for train/test data
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

    dataset_image.extend(trainset.data.cpu().detach().numpy())
    dataset_image.extend(testset.data.cpu().detach().numpy())
    dataset_label.extend(trainset.targets.cpu().detach().numpy())
    dataset_label.extend(testset.targets.cpu().detach().numpy())
    dataset_image = np.array(dataset_image)
    dataset_label = np.array(dataset_label)

    num_classes = len(set(dataset_label))
    print(f'Number of classes: {num_classes}')

    with open('southwest_images_new_train.pkl', 'rb') as train_f:
        saved_southwest_dataset_train = pickle.load(train_f).transpose((0, 3, 1, 2))

    with open('southwest_images_new_test.pkl', 'rb') as test_f:
        saved_southwest_dataset_test = pickle.load(test_f).transpose((0, 3, 1, 2))

    print("OOD (Southwest Airline) train-data shape we collected: {}".format(saved_southwest_dataset_train.shape))
    sampled_targets_array_train = 2 * np.ones((saved_southwest_dataset_train.shape[0],), dtype =int) # southwest airplane -> label as bird
    print("OOD (Southwest Airline) test-data shape we collected: {}".format(saved_southwest_dataset_test.shape))
    sampled_targets_array_test = 2 * np.ones((saved_southwest_dataset_test.shape[0],), dtype =int) # southwest airplane -> label as bird

    


    # dataset = []
    # for i in range(num_classes):
    #     idx = dataset_label == i
    #     dataset.append(dataset_image[idx])

    X, y, statistic = separate_data((dataset_image, dataset_label), num_clients, num_classes,
                                    niid, balance, partition, class_per_client=2)
    train_data, test_data = split_data(X, y)
    proxy_data = sample_proxy(list(dataset_image))

    sampled_indices = np.random.choice(
        a=len(train_data[0]['x']),       
        size=1000,    
        replace=False 
    )
    print("sematic  sss ",train_data[0]['x'][sampled_indices].shape)
    train_data[0]['x']=np.append(train_data[0]['x'][sampled_indices], saved_southwest_dataset_train,axis=0)
    train_data[0]['y']=np.append(train_data[0]['y'][sampled_indices], sampled_targets_array_train,axis=0)
    test_data[0]['x']=saved_southwest_dataset_test
    test_data[0]['y']=sampled_targets_array_test
    
    save_file(config_path, train_path, test_path, train_data, test_data, num_clients, num_classes,
              statistic, proxy_path, proxy_data, niid, balance, partition)

if __name__ == "__main__":
    niid = True if sys.argv[1] == "noniid" else False
    balance = True if sys.argv[2] == "balance" else False
    partition = sys.argv[3] if sys.argv[3] != "-" else None

    if(niid):
        dir_path="Cifar10_edge_"+partition+"/"

    generate_dataset(dir_path, num_clients, niid, balance, partition)