import torch.nn.functional as F 
import matplotlib.pyplot as plt 
from tqdm import tqdm
import torchvision
import numpy as np
import torch 
import cv2 
import os 

class Net(torch.nn.Module):
    def __init__(self,num,inputSize,Neuron):
        super(Net,self).__init__()
        self.iSize = inputSize
        self.fc1 = torch.nn.Linear(self.iSize*self.iSize,Neuron)
        self.fc2 = torch.nn.Linear(Neuron,num)

    def forward(self,x):
        x = x.view(-1,self.iSize*self.iSize)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return F.log_softmax(x,dim=1)

class CNN(torch.nn.Module):
    def __init__(self,num,inputSize,hidden1):
        super(CNN,self).__init__()
        self.iSize  = inputSize
        self.conv1  = torch.nn.Conv2d(1,4,3)
        self.bn1    = torch.nn.BatchNorm2d(4)
        self.pool   = torch.nn.MaxPool2d(2,2)
        self.conv2  = torch.nn.Conv2d(4,16,3)
        self.bn2    = torch.nn.BatchNorm2d(16)
        self.fc1    = torch.nn.Linear(16*38*38,hidden1)
        self.fc2    = torch.nn.Linear(hidden1,num)

    def forward(self,x):
        x = self.conv1(x)
        # print(x.size())
        x = torch.relu(x)
        x = self.bn1(x)
        x = self.pool(x)
        # print(x.size())
        x = self.conv2(x)
        # print(x.size())
        x = torch.relu(x)
        x = self.bn2(x)
        x = self.pool(x)    
        # print(x.size())
        x = x.view(-1,16*38*38)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return F.log_softmax(x,dim=1)

class Model29:
    def __init__(self,epoch,image_size,model,optimizer,lr,train_dir,test_dir,pt_name,loss_png,acc_png):
        denjyo_classes = [
            [
                [], #   名前格納用
                []  #   精度格納用
            ] for i in os.listdir(test_dir)
        ]
        
        self.model = model
        self.optimizer = optimizer
        self.image_size = image_size
        self.epoch = epoch
        self.loss_png = loss_png
        self.acc_png = acc_png
        self.pt_name = pt_name

        train_data = torchvision.datasets.ImageFolder(
            root=train_dir,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.Grayscale(),
                torchvision.transforms.Resize((image_size,image_size)),
                torchvision.transforms.ToTensor(),
            ])
        )

        self.train_data = torch.utils.data.DataLoader(
            train_data,
            batch_size=1,
            shuffle=True
        )

        loss = []
        for e in range(epoch):
            loss.append(self.train())
            for i,name in enumerate(os.listdir(test_dir)):
                test_path = test_dir+name+"/"
                denjyo_classes[i][0] = name
                denjyo_classes[i][1].append(self.test(test_path,name,i))

        self.saveLoss(loss)

        for x in denjyo_classes:
            self.saveAcc(x[1],x[0])
        self.saveModel()
        pass

    def train(self):
        for data in tqdm(self.train_data):
            x,target = data     
            self.optimizer.zero_grad()
            output = self.model(x)
            loss = F.nll_loss(output,target)
            loss.backward()
            self.optimizer.step()
        return loss

    def test(self,test_dir,name,label):
        self.model.eval()
        total = 0
        correct = 0
        with torch.no_grad():
            for f in tqdm(os.listdir(test_dir)):
                img = cv2.imread(test_dir+"/"+f)
                imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                img = cv2.resize(imgGray,(self.image_size,self.image_size))
                img = np.reshape(img,(1,self.image_size,self.image_size))
                img = np.transpose(img,(1,2,0))
                img = img.astype(np.uint8)
                mInput = torchvision.transforms.ToTensor()(img)
                mInput = mInput.view(-1,self.image_size*self.image_size)
                output = self.model(mInput)
                p = self.model.forward(mInput)
                if p.argmax() == label:
                    correct += 1
                total += 1
        percent = 100*correct/total
        print("{}{:>10f}".format(name,percent))
        return percent

    def saveLoss(self,loss):
        plt.figure()
        plt.plot(range(1,self.epoch+1),loss,label="trainLoss")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.xlim(1,self.epoch)
        plt.ylim(0,100)
        plt.legend()
        plt.savefig(self.loss_png)
        plt.close()

    def saveAcc(self,acc,name):
        plt.figure()
        plt.plot(range(1,self.epoch+1),acc,label=str(name))
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.xlim(1,self.epoch)
        plt.ylim(0,100)
        plt.legend()
        plt.savefig(str(name)+"_"+self.acc_png)
        plt.close()

    def saveModel(self):
        torch.save(self.model.state_dict(),self.pt_name)

if __name__ == "__main__":
    epoch = 40
    image_size = 160
    model = Net(num=29,inputSize=image_size,Neuron=320)
    lr = 0.000005
    optimizer = torch.optim.Adam(params=model.parameters(),lr=lr)
    train_dir = "train_data29/"
    test_dir = "test_data29/"
    pt_name = "nn.pt"
    loss_png = "loss.png"
    acc_png = "acc.png"

    model = Model29(epoch,image_size,model,optimizer,lr,train_dir,test_dir,pt_name,loss_png,acc_png)

