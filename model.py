import torch
import torchvision.transforms as transforms
import torchvision.models as models
import matplotlib.pyplot as plt
import torch.nn as nn
from PIL import Image
from pathlib import Path
from torchvision.models import VGG19_Weights


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)
torch.set_default_dtype(torch.float16)

training_data_list = [] 
pathlist = Path("./DataSuperRes/LR/train/").glob('**/*.jpg')
for path in pathlist:
    path_s = str(path)
    path_l = path_s.split("/")
    img_256 = Image.open("./DataSuperRes/HR/train/"+path_l[-2]+"/"+path_l[-1][2:-6]+"256.jpg").resize((224, 224),Image.BICUBIC)
    img_64 = Image.open(path_s).resize((224, 224),Image.BICUBIC)
    training_data_list.append((img_64,img_256))

testing_data_list = []
pathlist = Path("./DataSuperRes/LR/test/").glob('**/*.jpg')
for path in pathlist:
    path_s = str(path)
    path_l = path_s.split("/")
    img_256 = Image.open("./DataSuperRes/HR/test/"+path_l[-2]+"/"+path_l[-1][2:-6]+"256.jpg").resize((224, 224),Image.BICUBIC)
    img_64 = Image.open(path_s).resize((224, 224),Image.BICUBIC)
    testing_data_list.append((img_64,img_256))


class DataTransformer(torch.utils.data.Dataset):

    def __init__(self,base_dataset,transform):
        self.base = base_dataset
        self.transform = transform

    def __getitem__(self,index):
        item,label = self.base[index]
        return self.transform(item),self.transform(label)

    def __len__(self):
        return len(self.base)

train_dataset = DataTransformer(training_data_list,transforms.ToTensor())
test_dateset = DataTransformer(testing_data_list,transforms.ToTensor())

num_threads = 4
batch_size = 128

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True,num_workers=num_threads,generator=torch.Generator(device=device))
test_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=False,num_workers=num_threads,generator=torch.Generator(device=device))

#print(next(iter(train_loader))[0])

class RUnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.k7n64 = nn.Sequential(nn.Conv2d(3,64,7,stride=1,padding="same"),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU(inplace=True))
        self.pooling = nn.MaxPool2d(kernel_size=2,stride=2)
        self.down_block1 = [self.add_down_block(64,3,64),
                            self.add_down_block(64,3,64),
                            self.add_down_block(64,3,64),
                            self.add_down_block(64,3,128),
                            nn.Conv2d(64,128,1,1,padding="same")]
        self.down_block2 = [self.add_down_block(128,3,128),
                            self.add_down_block(128,3,128),
                            self.add_down_block(128,3,128),
                            self.add_down_block(128,3,256),
                            nn.Conv2d(128,256,1,1,padding="same")]
        self.down_block3 = [self.add_down_block(256,3,256),
                            self.add_down_block(256,3,256),
                            self.add_down_block(256,3,256),
                            self.add_down_block(256,3,512),
                            nn.Conv2d(256,512,1,1,padding="same")]
        self.down_block4 = [self.add_down_block(512,3,512),
                            self.add_down_block(512,3,512),
                            nn.Sequential(nn.BatchNorm2d(512),nn.ReLU(inplace=True))]

        self.k3n1024 = nn.Sequential(nn.Conv2d(512,1024,3,stride=1,padding="same"),
                                   nn.ReLU(inplace=True))
        self.k3n512 = nn.Sequential(nn.Conv2d(1024,512,3,stride=1,padding="same"),
                                   nn.ReLU(inplace=True))

        self.pixel_shuffle = nn.PixelShuffle(2)
        self.dropout = nn.Dropout(p=0.05)

        self.up_block1 = self.add_up_block(1024,3,512)
        self.up_block2 = self.add_up_block(640,3,384)
        self.up_block3 = self.add_up_block(352,3,256)
        self.up_block4 = self.add_up_block(192,3,96)

        self.k3n99 = nn.Sequential(nn.Conv2d(88,99,3,stride=1,padding="same"),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(99,99,3,stride=1,padding="same"),
                                   nn.ReLU(inplace=True))

        self.k1n3 = nn.Conv2d(99,3,1,stride=1,padding="same")


    def add_up_block(self,input_c,k,n):
        return nn.Sequential(nn.BatchNorm2d(input_c),
                             nn.Conv2d(input_c,n,k,stride=1,padding="same"),
                             nn.ReLU(inplace=True),
                             nn.Conv2d(n,n,k,stride=1,padding="same"),
                             nn.ReLU(inplace=True))

    def add_down_block(self,input_c,k,n):
        return nn.Sequential(nn.Conv2d(input_c,n,k,stride=1,padding="same"),
                             nn.BatchNorm2d(n),
                             nn.ReLU(inplace=True),
                             nn.Conv2d(n,n,k,stride=1,padding="same"),
                             nn.BatchNorm2d(n))

    def forward(self,x):
        x=self.k7n64(x)
        x_5=x
        x=self.pooling(x)
        for i in range(len(self.down_block1)-1):
            x_ = self.down_block1[i](x)
            if x.shape[1]!=x_.shape[1]:
                x = self.down_block1[-1](x)
            x= x + x_
        x=self.dropout(x)
        x_4=x
        x=self.pooling(x)

        for i in range(len(self.down_block2)-1):
            x_ = self.down_block2[i](x)
            if x.shape[1]!=x_.shape[1]:
                x = self.down_block2[-1](x)
            x= x + x_
        x=self.dropout(x)
        x_3=x
        x=self.pooling(x)

        for i in range(len(self.down_block3)-1):
            x_ = self.down_block3[i](x)
            if x.shape[1]!=x_.shape[1]:
                x = self.down_block3[-1](x)
            x= x + x_
        x=self.dropout(x)
        x_2=x
        x=self.pooling(x)

        for i in range(len(self.down_block4)):
            if i!=(len(self.down_block4)-1):
                x=x + self.down_block4[i](x)
            else:
                x= self.down_block4[i](x)
                
        x=self.dropout(x)
        x_1=x
        x=self.k3n1024(x)
        x=self.k3n512(x)

        x=torch.concat((x,x_1),dim=1)
        x=self.up_block1(x)
        x=self.pixel_shuffle(x)

        x=torch.concat((x,x_2),dim=1)
        x=self.up_block2(x)
        x=self.pixel_shuffle(x)

        x=torch.concat((x,x_3),dim=1)
        x=self.up_block3(x)
        x=self.pixel_shuffle(x)

        x=torch.concat((x,x_4),dim=1)
        x=self.up_block4(x)
        x=self.pixel_shuffle(x)

        x=torch.concat((x,x_5),dim=1)
        x=self.k3n99(x)
        x=self.k1n3(x)
        return x

my_RUnet = RUnet()
my_RUnet.to(device)

vgg19 = models.vgg19(weights=VGG19_Weights.DEFAULT)
vgg19.to(device)

loss_f = nn.MSELoss()
optimizer = torch.optim.Adam(my_RUnet.parameters(),weight_decay=1e-3)

def train(model,train_loader,loss_f,optimizer,device):

    model.train()

    for i,(img,label) in enumerate(train_loader):
      with torch.autocast(device_type=device.type, dtype=torch.float16):
        img,label = img.to(device),label.to(device)
        y_pred = model(img)
        loss = loss_f(vgg19.features(label),vgg19.features(y_pred))
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()


def test(model,loader,loss_f,optimizer,device):

    with torch.no_grad():

        model.eval()
        N=0
        tot_loss=0.0

        for i,(img,label) in enumerate(loader):
            with torch.autocast(device_type=device.type, dtype=torch.float16):
              img,label = img.to(device),label.to(device)
              y_pred = model(img)
              current_loss = loss_f(vgg19.features(label),vgg19.features(y_pred)).item()*img.shape[0]
              print(current_loss)
              tot_loss+=current_loss
              N+=img.shape[0]

        tot_loss/=N

        return tot_loss
    

epochs = 50
for i in range(epochs):
    print("Epoch {}".format(i))
    train(my_RUnet,train_loader,loss_f,optimizer,device)

    test_loss = test(my_RUnet,test_loader,loss_f,optimizer,device)
    print(" Testing : Loss : {:.4f}".format(test_loss))
