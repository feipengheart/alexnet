import torch,pickle
from PIL import Image
from model import Alexnet
from torchvision import transforms

test_image=["images/0.jpg","images/00.jpg","images/11.jpg","images/1.jpg"]
net=Alexnet(2)
net.eval()
net.load_state_dict(torch.load("best_model.pth"))
transforms = transforms.Compose([transforms.Resize((224, 224)),  # 将每个图片都缩放为相同的分辨率64x64，便于GPU的处理
                                     transforms.ToTensor(),  # 将数据集转化为张量
                                     # transforms.Normalize(mean=[0.485,0.456,0.406],
                                     #                      std=[0.229,0.224,0.225])#设置用于归一化的参数
                                     ])
device="cpu"
with open('image_label.pkl', 'rb') as f:
    index_to_class=pickle.load(f)
for i in test_image:
    img=Image.open(i)
    val_data=transforms(img)
    val_data=torch.unsqueeze(val_data,0)
    y_predict=net(val_data)
    y_predict_index=torch.argmax(y_predict,1)
    print(index_to_class[y_predict_index.item()])