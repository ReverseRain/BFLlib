import torch
from torch_cka import CKA
import random
import numpy as np
from sklearn.preprocessing import label_binarize
from sklearn import metrics
from torch.utils.data import DataLoader,Subset
from utils.data_utils import read_client_data,create_poisoned_dataset

# ... 假设您已经有了两个模型的特征输出 Z1 和 Z2 ...
# Z1 和 Z2 的形状通常是 [Batch_Size, Feature_Dimension] 或 [Batch_Size, C, H, W]
def test(model2,id):
    model2.eval()

    test_acc = 0
    test_num = 0
    t=0
    y_prob = []
    y_true = []

    test_data = read_client_data("Cifar10", id, is_train=False)
    if(id==0):
        test_data = create_poisoned_dataset(test_data,3,is_train=False,blend=False)
    dataloader = DataLoader(test_data, 32, drop_last=False, shuffle=True)
    with torch.no_grad():
        for x, y in dataloader:
            if type(x) == type([]):
                x[0] = x[0].to(device)
            else:
                x = x.to(device)
            y = y.to(device)
            output = model2(x)


            test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
            test_num += y.shape[0]

            t += (torch.sum(y == 0)).item()

            y_prob.append(output.detach().cpu().numpy())
            nc = 10
            lb = label_binarize(y.detach().cpu().numpy(), classes=np.arange(nc))


            y_true.append(lb)

    # self.model.cpu()
    # self.save_model(self.model, 'model')
    y_prob = np.concatenate(y_prob, axis=0)
    y_true = np.concatenate(y_true, axis=0)


    auc = metrics.roc_auc_score(y_true, y_prob, average='micro')

    print(test_acc/test_num,test_acc,test_num)

class YourModelWithSoftmax(torch.nn.Module):
    def __init__(self, original_model):
        super().__init__()
        self.base = original_model.base
        self.head = torch.nn.Sequential(original_model.head,
                                  torch.nn.Softmax(dim=1))
    
    def forward(self, x):
        x = self.base(x)
        x = self.head(x)
        return x

test_data = read_client_data("Cifar10", 0, is_train=False)
test_data = create_poisoned_dataset(test_data,3,is_train=False,blend=False)
random.shuffle(test_data)
dataloader = DataLoader(test_data, 32, drop_last=True, shuffle=False)
device='cuda' if torch.cuda.is_available() else 'cpu'


model1 = YourModelWithSoftmax(
    torch.load("./checkpoint_cifar10/FedAVG_server_True.pt", map_location=device,weights_only=False))
model2 = YourModelWithSoftmax(
    torch.load("./checkpoint_cifar10/FedAVG_server_False.pt", map_location=device,weights_only=False))

# model1 = torch.load("./checkpoint_cifar10_vit/FedAVG_server_True.pt", map_location='cpu',weights_only=False)
# model2 = torch.load("./checkpoint_cifar10_vit/FedAVG_server_False.pt", map_location='cpu',weights_only=False)

cka = CKA(model1=model1, model2=model2,
                     model1_layers=['base.layer1','base.layer2','base.layer3','base.layer4','head'],
                     model2_layers=['base.layer1','base.layer2','base.layer3','base.layer4','head'],
                    model1_name='High ASR model',
                     model2_name='Low ASR model',
                     device='cuda' if torch.cuda.is_available() else 'cpu')
# print(model1.base.vit.vit.encoder.layer[0])
# print(model1.base.vit.vit.encoder.layer[11])
for name, layer in model1.named_modules():
    print("name of ",name)

# for name, layer in model2.named_modules():
#     print("name of ",name)
model1.eval()
model2.eval()
# cka = CKA(model1=model1, model2=model2,
#                      model1_layers=['base.vit.vit.encoder.layer.0.layernorm_after',
#                                     'base.vit.vit.encoder.layer.1.layernorm_after',
#                                     'base.vit.vit.encoder.layer.2.layernorm_after',
#                                     'base.vit.vit.encoder.layer.3.layernorm_after',
#                                     'base.vit.vit.encoder.layer.4.layernorm_after',
#                                     'base.vit.vit.encoder.layer.5.layernorm_after',
#                                     'base.vit.vit.encoder.layer.6.layernorm_after',
#                                     'base.vit.vit.encoder.layer.7.layernorm_after',
#                                     'base.vit.vit.encoder.layer.8.layernorm_after',
#                                     'base.vit.vit.encoder.layer.9.layernorm_after',
#                                     'base.vit.vit.encoder.layer.10.layernorm_after',
#                                     'base.vit.vit.encoder.layer.11.layernorm_after',
#                                     'head'
#                                     ],
#                      model2_layers=[
#                                     'base.vit.vit.encoder.layer.0.layernorm_after',
#                                     'base.vit.vit.encoder.layer.1.layernorm_after',
#                                     'base.vit.vit.encoder.layer.2.layernorm_after',
#                                     'base.vit.vit.encoder.layer.3.layernorm_after',
#                                     'base.vit.vit.encoder.layer.4.layernorm_after',
#                                     'base.vit.vit.encoder.layer.5.layernorm_after',
#                                     'base.vit.vit.encoder.layer.6.layernorm_after',
#                                     'base.vit.vit.encoder.layer.7.layernorm_after',
#                                     'base.vit.vit.encoder.layer.8.layernorm_after',
#                                     'base.vit.vit.encoder.layer.9.layernorm_after',
#                                     'base.vit.vit.encoder.layer.10.layernorm_after',
#                                     'base.vit.vit.encoder.layer.11.layernorm_after',
#                                     'head'
#                                     ],
#                      model1_name='High ASR model',
#                      model2_name='Low ASR model',
#                      device='cuda' if torch.cuda.is_available() else 'cpu')

# 计算 CKA
print(dir(cka))
# cka.hsic_matrix = torch.nan_to_num(cka.hsic_matrix, nan=0.0)
cka.kernel = "rbf"
cka.compare(dataloader)
cka.plot_results(save_path="specified_layers_cka.png")
print(dir(cka))

# cka_matrix = cka.cka  # 这是一个二维数组，形状为 (len(model1_layers), len(model2_layers))
# print("CKA相似度矩阵（数组形式）:")
# print(cka_matrix)

test(model2,2)
test(model2,3)
test(model2,4)
test(model2,5)
test(model2,6)
test(model2,7)
test(model2,8)
test(model2,9)
test(model2,0)

test(model1,2)
test(model1,3)
test(model1,4)
test(model1,5)
test(model1,6)
test(model1,7)
test(model1,8)
test(model1,9)
test(model1,0)
# X = torch.load("./checkpoint_cifar10/z4_True.pt", map_location='cpu')
# Y = torch.load("./checkpoint_cifar10/z4_False.pt", map_location='cpu')
# cka = CKA()

# 计算 CKA 相似度
# print(type(X))
# cka_value = CKA(X, Y)
# print(cka_value)

# model2.eval()

# test_acc = 0
# test_num = 0
# poison_num=0
# t=0
# y_prob = []
# y_true = []

# test_data = read_client_data("Cifar10", 1, is_train=False)
# test_data = create_poisoned_dataset(test_data,3,is_train=False,blend=False)
# dataloader = DataLoader(test_data, 32, drop_last=False, shuffle=True)
# with torch.no_grad():
#     for x, y in dataloader:
#         if type(x) == type([]):
#             x[0] = x[0].to(device)
#         else:
#             x = x.to(device)
#         y = y.to(device)
#         output = model2(x)


#         test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
#         test_num += y.shape[0]

#         poison_num += (torch.sum(torch.argmax(output, dim=1) == 0)).item()
#         t += (torch.sum(y == 0)).item()

#         y_prob.append(output.detach().cpu().numpy())
#         nc = 10
#         lb = label_binarize(y.detach().cpu().numpy(), classes=np.arange(nc))


#         y_true.append(lb)

# # self.model.cpu()
# # self.save_model(self.model, 'model')
# y_prob = np.concatenate(y_prob, axis=0)
# y_true = np.concatenate(y_true, axis=0)


# auc = metrics.roc_auc_score(y_true, y_prob, average='micro')

# print(test_acc/test_num)

