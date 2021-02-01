import os 
import sys
import cv2
import torch
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as transforms

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

# class KaoSet:
#     cap_channel         = 0
#     WINDOW_WIDTH        = 1920
#     WINDOW_HEIGHT       = 1080
#     FRAME_WIDTH         = 500
#     FRAME_HEIGHT        = 500
#     color               = (255,0,255)

#     def show(self):
#         cap = cv2.VideoCapture(self.cap_channel)
#         cap.set(cv2.CAP_PROP_FRAME_WIDTH,   self.WINDOW_WIDTH)
#         cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  self.WINDOW_HEIGHT)
#         x   = 100
#         y   = 100
#         while True:
#             success,img = cap.read()
#             cv2.rectangle(img,(x,y),(x+self.FRAME_WIDTH,y+self.FRAME_HEIGHT),self.color,thickness=10)
#             cv2.imshow("Image",img)
#             H,W,C = img.shape
#             x = int((W - self.FRAME_WIDTH)/2)
#             y = int((H - self.FRAME_HEIGHT)/2)
#             cv2.waitKey(100)

class Suiron:
    CAP_CHANNEL         =   0     #   0か1にしてください
    IS_CAP_INIT         =   0
    WINDOW_WIDTH        =   1920
    WINDOW_HEIGHT       =   1080
    FRAME_WIDTH         =   600
    FRAME_HEIGHT        =   600
    x                   =   100
    y                   =   100
    CASCADEPATH         =   "haarcascades/haarcascade_frontalface_default.xml"
    MOJI_OOKISA         =   1.0
    # ---------- 学習の時と同じパラメータでなければならない ---------- #
    inputSize           =   160
    model               =   Net(num=26,inputSize=inputSize,Neuron=320)
    PATH                =   "models/26classes.pt"
    BODY_TEMP           =   36.5
    BODY_TEMP_SAFE      =   (255,0,0)
    BODY_TEMP_OUT       =   (255,0,255)
    COLOR               =   BODY_TEMP_SAFE
    CNT                 =   0

    DELAY_MSEC          =   1
    CNT_MAX             =   50
    PROGRESS_BAR_LEN    =   100

    NAME = [
        "ando",
        "enomaru",
        "hamada",
        "higashi",
        "kataoka",
        "kawano",
        "kodama",
        "masuda",
        "matsuzaki",
        "miyatake",
        "mizuki",
        "nagao",
        "okamura",
        "ooshima",
        "ryuuga",
        "shinohara",
        "soushi",
        "suetomo",
        "takemoto",
        "tamejima",
        "teppei",
        "uemura",
        "wada",
        "watanabe",
        "yamaji",
        "yamashita"
    ]
    ListCNT = [
        [] for i in NAME
    ]
    notFirst = 0
    name = "-------"
    p = 0
    percent = 0

    def __init__(self):
        self.cap = cv2.VideoCapture(self.CAP_CHANNEL)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,   self.WINDOW_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  self.WINDOW_HEIGHT)
        self.cascade = cv2.CascadeClassifier(self.CASCADEPATH)
        self.model.load_state_dict(torch.load(self.PATH))
        self.model.eval()

        for i,_ in enumerate(self.ListCNT):
            self.ListCNT[i] = 0
        

    def __del__(self):
        self.cap.release()
        cv2.destroyAllWindows()
        

    def real_time_haar(self):
        success,img = self.cap.read()
        try:
            imgGray     = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        except cv2.error:
            if self.IS_CAP_INIT == 0:
                self.CAP_CHANNEL    ^=  1
                self.cap = cv2.VideoCapture(self.CAP_CHANNEL)
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,   self.WINDOW_WIDTH)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  self.WINDOW_HEIGHT)
            success,img = self.cap.read()
            imgGray     = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            self.IS_CAP_INIT        =   1

        
        imgResult   = img.copy()
        facerect    = self.cascade.detectMultiScale(imgGray,scaleFactor=1.1,minNeighbors=2,minSize=(200,200))

        H,W,C = img.shape
        self.x = int((W - self.FRAME_WIDTH)/2)
        self.y = int((H - self.FRAME_HEIGHT)/2)

        #   もし体温が37.0度以上の時は赤、未満は青
        if self.BODY_TEMP >= 37.0:
            self.COLOR  =   self.BODY_TEMP_OUT
        else:
            self.COLOR  =   self.BODY_TEMP_SAFE

        if len(facerect) > 0:
            for (x,y,w,h) in facerect:
                cv2.rectangle(imgResult,(x,y),(x+w,y+h),self.COLOR,thickness=2)
                imgTrim = img[y:y+h,x:x+w]
                str_y,percent,ld = self.maesyori_suiron(imgTrim,self.inputSize)

                cv2.rectangle(img,(self.x,self.y),(self.x+self.FRAME_WIDTH,self.y+self.FRAME_HEIGHT),self.COLOR,thickness=10)
                # cv2.putText(img, str_y+" "+str(percent)+"%", (40, 40), cv2.FONT_HERSHEY_SIMPLEX,self.MOJI_OOKISA,self.COLOR,thickness=2)
                if self.notFirst == 1:
                    cv2.putText(img, self.name+" "+str(self.percent)+"%", (40, 40), cv2.FONT_HERSHEY_SIMPLEX,self.MOJI_OOKISA,self.COLOR,thickness=2)
                    cv2.putText(img,"Body TEMP",(40,40*2),cv2.FONT_HERSHEY_SIMPLEX,self.MOJI_OOKISA,self.COLOR,thickness=2)
                    cv2.putText(img,str(self.BODY_TEMP),(40,40*3),cv2.FONT_HERSHEY_SIMPLEX,self.MOJI_OOKISA,self.COLOR,thickness=2)
                
                cv2.line(
                        img,
                        (self.x+self.FRAME_WIDTH+50,                        int((self.y+self.FRAME_HEIGHT)/2)+40*3),
                        (self.x+self.FRAME_WIDTH+50+self.PROGRESS_BAR_LEN,  int((self.y+self.FRAME_HEIGHT)/2)+40*3),
                        (204,204,204),
                        15
                )

                #   もし、ldが  "-------"ではないとき
                if ld != "-------":
                    # print("ok")
                    self.notFirst = 1
                    self.p /= self.CNT_MAX 
                    self.name = ld
                    self.percent = int(self.p)
                    if self.percent > 100:
                        self.percent = 100
                    pass 
                else:
                    self.p += percent

                    cv2.putText(img,"Please",(self.x+self.FRAME_WIDTH+40,int((self.y+self.FRAME_HEIGHT)/2)+40),cv2.FONT_HERSHEY_SIMPLEX,self.MOJI_OOKISA,self.COLOR,thickness=2)
                    cv2.putText(img,"wait.",(self.x+self.FRAME_WIDTH+40,int((self.y+self.FRAME_HEIGHT)/2)+40*2),cv2.FONT_HERSHEY_SIMPLEX,self.MOJI_OOKISA,self.COLOR,thickness=2)
                    cv2.line(
                        img,
                        (self.x+self.FRAME_WIDTH+50,                                                    int((self.y+self.FRAME_HEIGHT)/2)+40*3),
                        (self.x+self.FRAME_WIDTH+50+(int(self.PROGRESS_BAR_LEN/self.CNT_MAX))*self.CNT, int((self.y+self.FRAME_HEIGHT)/2)+40*3),
                        self.COLOR,
                        15
                    )

                cv2.imshow("Image",img)
                if self.DELAY_MSEC != 0:
                    cv2.waitKey(self.DELAY_MSEC)
                else:
                    cv2.waitKey(self.DELAY_MSEC)

                return ld
        else:
            #   もし顔が認識できていなかったらCNTをリセットする
            self.CNT    =   0
            str_y = "-------"
            cv2.rectangle(img,(self.x,self.y),(self.x+self.FRAME_WIDTH,self.y+self.FRAME_HEIGHT),self.COLOR,thickness=10)
            cv2.putText(img, "Set Face", (40*2, 40*2), cv2.FONT_HERSHEY_SIMPLEX,self.MOJI_OOKISA*2,self.COLOR,thickness=4)
            cv2.imshow("Image",img)
            cv2.waitKey(self.DELAY_MSEC)
            return str_y

            
    
    def maesyori_suiron(self,imgCV,imgSize):
        # チャンネル数を１
        imgGray = cv2.cvtColor(imgCV,cv2.COLOR_BGR2GRAY)
        
        #リサイズ
        img = cv2.resize(imgGray,(imgSize,imgSize))

        # リシェイプ
        img = np.reshape(img,(1,imgSize,imgSize))

        # transpose h,c,w
        img = np.transpose(img,(1,2,0))

        # ToTensor 正規化される
        img = img.astype(np.uint8)
        mInput = transforms.ToTensor()(img)  

        #推論
        #print(mInput.size())
        output = self.model(mInput[0])

        #予測値
        # p0 = self.model.forward(mInput)
        p1 = self.model.forward(mInput).exp()

        #予測値のパーセンテージ
        # m = torch.nn.Softmax(dim=1)
        # x0 = m(p0)
        # x0 = x0.to('cpu').detach().numpy().copy() 
        # x0 = x0[0]
        x1 = p1.to('cpu').detach().numpy().copy() 
        x1 = x1[0]
        # すべての中で最も大きい値
        p1 = p1.argmax().to('cpu').detach().numpy().copy()
        percent = 0

        str_y = str(self.NAME[p1])
        percent = x1[p1]*100
        self.ListCNT[p1] += 1
        s = "-------"

        self.CNT += 1

        if self.CNT == self.CNT_MAX:
            max_value = max(self.ListCNT)
            max_index = self.ListCNT.index(max_value)
            s = str(self.NAME[max_index])

            self.CNT = 0
            for i,_ in enumerate(self.ListCNT):
                self.ListCNT[i] = 0

        # 戻り値は予測値とパーセンテージ,確実な値
        return str_y,percent,s

    def imshow(self,path):
        img = cv2.imread(path)
        cv2.imshow("Thermo sensor",img)
        cv2.waitKey(self.DELAY_MSEC)
        # cv2.moveWindow("Thermo sensor",self.WINDOW_WIDTH,int(self.WINDOW_HEIGHT/2))


if __name__ == "__main__":
    k = Suiron()

    while True:
        k.real_time_haar()
