from model import Alexnet
import torch,pickle
from matplotlib import pyplot as plt
from torch import nn,optim
import torchvision
from torchvision import transforms#进行训练数据的转换
from torch.utils.data import DataLoader
torch.manual_seed(4)

train_loss = []
val_loss = []
val_acc = []
best_acc = 0.0

def train(epoch,train_data_loader,device,net,loss_function,optm):
    net.train()
    epoch_loss=0
    iter_couter=len(train_data_loader)
    for index,(train,target) in enumerate(train_data_loader):
        train=train.to(device)
        target=target.to(device)
        y_predict=net(train)
        loss=loss_function(y_predict,target)
        optm.zero_grad()
        loss.backward()
        optm.step()
        epoch_loss+=loss.item()
        print(f"训练轮次:{epoch + 1}\t第几个批次:{index + 1}\t训练误差:{loss.item()}")
    train_loss.append(epoch_loss / iter_couter)

def val(epoch,train_data_loader,device,net,loss_function,optm):
    global best_acc
    net.eval()
    val_epoch_loss=0
    predict_correct_num = 0
    test_num = 0
    for index,(test,target) in enumerate(train_data_loader):
        test=test.to(device)
        target=target.to(device)
        test_ = len(test)
        test_num+=test_
        y_predict=net(test)
        y_predict_index = torch.argmax(y_predict, 1)
        loss=loss_function(y_predict,target)
        val_epoch_loss+=loss.item()
        predict_correct_num += sum([1 for index in range(len(target)) if target[index] == y_predict_index[index]])
        print(f"训练轮次:{epoch + 1}\t第几个批次:{index + 1}\t训练误差:{loss.item()}")

    acc = predict_correct_num / test_num
    print("acc", acc, test_num)
    val_acc.append(acc)
    val_loss.append(val_epoch_loss / len(train_data_loader))
    if val_acc[-1] > best_acc:
        best_acc = val_acc[-1]
        # 保存模型
        torch.save(net.state_dict(), f'best_model.pth')
        print("y_predict_index", y_predict_index)
        print("target", target)

if __name__ == '__main__':

    train_path="./train/"
    val_path="./val/"

    transforms = transforms.Compose([transforms.Resize((224, 224)),  # 将每个图片都缩放为相同的分辨率64x64，便于GPU的处理
                                     transforms.ToTensor(),  # 将数据集转化为张量
                                     # transforms.Normalize(mean=[0.485,0.456,0.406],
                                     #                      std=[0.229,0.224,0.225])#设置用于归一化的参数
                                     ])

    train_data=torchvision.datasets.ImageFolder(root=train_path,transform=transforms)
    val_data=torchvision.datasets.ImageFolder(root=val_path,transform=transforms)
    train_data_load=DataLoader(train_data,16,True,drop_last=True)
    val_data_load=DataLoader(val_data,12)

    idx_to_class = {value: key for key, value in train_data.class_to_idx.items()}
    with open('image_label.pkl', 'wb') as f:
        pickle.dump(idx_to_class, f)

    device="cpu"
    net=Alexnet(2)
    net.to(device)
    loss_functoin = nn.CrossEntropyLoss()
    optm = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    epochs = 100  # 轮次   一轮就是把训练样本训练一遍

    for epoch in range(epochs):
        train(epoch,train_data_load,device,net,loss_functoin,optm)
        val(epoch,val_data_load,device,net,loss_functoin,optm)

    plt.plot(range(len(train_loss)), train_loss)
    plt.show()