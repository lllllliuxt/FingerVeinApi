from Load_Image import LoadImage
from PIL import Image
from torchvision import transforms
import numpy as np
from Model import AlexNet
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

# 学习率
LR = 0.01

class FingerVein:
    url = "E:\graduate\QBroi\ROIs\\006\R_Middle\\08.bmp"      
    def deal_with_img(self, img):
        transform = transforms.Compose([
            transforms.Resize((60, 175)),
            transforms.ToTensor()
        ])

        img_transform = transform(img)
        new_img = torch.unsqueeze(img_transform, 1)
        
        return new_img
    
    def get_img(self, url):
        # 根据url获取图片
        img = Image.open(url).convert('L')
        return img
    
    def test(self, model, data):
        model.eval()
        # 测试数据
        data = data.float()
        output = model(data)
        pred = output.argmax(dim=1, keepdim=True)
        pred = pred.cpu().numpy()
        return pred[0][0] + 1
    
    def is_valid(self, url, target):
        img = self.get_img(url)
        img = self.deal_with_img(img)
        model = AlexNet()
        model = torch.nn.DataParallel(model)

        optimizer = optim.SGD(model.parameters(), lr=LR)
        scheduler = StepLR(optimizer, step_size=20, gamma=0.1)

        checkpoint = torch.load('AlexNet.pkl', map_location='cpu')
        if checkpoint is not None:
            model.load_state_dict(checkpoint["net"])
            optimizer.load_state_dict(checkpoint["optimizer"])
        
        return self.test(model,img) == target