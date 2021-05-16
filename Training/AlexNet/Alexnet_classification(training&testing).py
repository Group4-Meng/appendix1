from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import cv2
from PIL import Image
import torch.nn as nn




my_transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

my_transform_2=transforms.Compose([
    
    transforms.CenterCrop(227),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


num_classes=2
learning_rate=0.00001
batch_size=4
num_epochs=10


device =torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('Cuda Availability:',torch.cuda.is_available())

class Wood_Plastic_Dataset(Dataset):
        
    def __init__ (self, csv_file,root_dir,transform=None):
    
        self.DataFrame = pd.read_csv(csv_file)
        
        self.root=root_dir
        
        self.transform= transform
    
    
    def __len__(self):
        
        return(len(self.DataFrame ))
    
    def __getitem__(self,index):
        
        image_path= os.path.join(self.root, self.DataFrame.iloc[index,0 ])
        
        img= Image.open(image_path)
        
        img=np.asarray(img)
        
        image=Image.fromarray(img)
        
        
        img_label=torch.tensor(self.DataFrame.iloc[index,1])
        
        
        
        
        # print('The image shape is ',image.shape)
        
        # print('The image array type is ,',type(image))
        
        if self.transform:
            image=self.transform(image)
        
        img_sample=(image,img_label)
            
            # print('The image new array type is ,',type(img_sample[0]))
            
        return img_sample  
    

     
            
dataset=Wood_Plastic_Dataset('Desktop/zena/wood&plasticNew.csv','Desktop/zena/Wood&Plastic',transform= my_transform)



sample= dataset[1]

print(sample)
print('my', sample[0].shape)

image=sample[0]
print(image[0].mean())


# plt.imshow(sample[0].reshape(227,227,3))



train_dataset,test_dataset=torch.utils.data.random_split(dataset=dataset, lengths=[800,200])

train_dataloader=torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size, shuffle = True )  

test_dataloader=torch.utils.data.DataLoader(dataset=test_dataset,batch_size=batch_size, shuffle = True )    

example=iter(train_dataloader)

print('The length is:',len(train_dataloader))

print(len(test_dataloader))


for i in example.next():
    
    print('Here',i.shape)
    
classes=('wood','plasticBags')



class Alexnet(nn.Module):
    
    def __init__(self):
        
        super().__init__()
        
        self.convolutional =  nn.Sequential(
        
        nn.Conv2d(in_channels=3,out_channels=96, kernel_size=11, stride=4 ),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3,stride=2),
        
        nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5,padding=2),
        nn.ReLU(inplace= True),
        nn.MaxPool2d(kernel_size=3,stride=2),
    
        nn.Conv2d(256,384, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        
        nn.Conv2d(384, 384, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        
        nn.Conv2d(384, 256, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        
        nn.MaxPool2d(kernel_size=3, stride=2),
        
        )
        
        self.averagepooling=nn.AdaptiveAvgPool2d((6,6))
        
        self.FullyConnected=nn.Sequential(
            
        nn.Dropout(),
        nn.Linear(256*6*6,4096),
        nn.ReLU(),
            
        nn.Dropout(),
        nn.Linear(4096,4096),
        nn.ReLU(),
        
        nn.Linear(4096,num_classes),
        )
        
    def forward(self, x ):
            
        x=self.convolutional(x)
            
        x=self.averagepooling(x)
            
        x=torch.flatten(x,1)
            
        x=self.FullyConnected(x)
            
        return x
        
model=Alexnet().to(device)

criterion = nn.CrossEntropyLoss()

optimizer= torch.optim.Adam(model.parameters(),lr=learning_rate)

num_total_steps=len(train_dataloader)



# ----------Training Loop(uncommenet to train)---------------
Loss_List=[]
epochs_list=[i for i in range(num_epochs)]
for epoch in range(num_epochs):
    
   
    
    for i,(images,labels)in enumerate(train_dataloader):
        
        images=images.to(device)
    
        labels=labels.to(device)
        
        output=model(images)
        
        if i==1:
            print('shape-output',output.shape)
            print('OUT',output)
        
        loss=criterion(output,labels)
       
        optimizer.zero_grad ()
        
        loss.backward()
         
        optimizer.step()
        
        
        if (i+1)%2==0:
            print(f'epoch:{epoch+1}/{num_epochs},step:{i+1}/{num_total_steps},loss:{loss.item()}')
            
    Loss_List.append(loss.item())
            
plt.xlabel('num_epochs')
plt.ylabel('loss')
plt.title('cross entropy')
plt.plot(epochs_list,Loss_List)
plt.show()
            
        
# -----------------Testing Loop----------------------
Correct=0
Total=0


for x,(images,labels) in enumerate(test_dataloader):
    j=0
    
    images=images.to(device)
    
    labels=labels.to(device)
    
    output=model(images)
    
    Probs,Predicted=torch.max(output,1)
    
    if x==0:
        print('predicted:',Predicted) 
        
        print('labels:',labels)
        
    

    Total+=labels.shape[0]
    
    for i in Predicted:
        if i==labels[j]:
            Correct+=1
        j+=1
            
           
accuracy=(Correct/Total)*100

print('accuracy:',accuracy)



# -------Testing using computer camera-------
cap=cv2.VideoCapture(0)
class Camera:
    
    def __init__(self,transform=None):
        
        self.transform=transform
        
        
    def change_frame(self,frame):
        
        if self.transform:
            frame=self.transform(frame)
            
        return frame
            
cam=Camera(transform=my_transform_2)

x=0

while True:   
    ret,frame=cap.read()
    
    cv2.imshow('Video',frame)   
    
    frame_np=np.asarray(frame)
    
    frame_PIL=Image.fromarray(frame_np)
    
    
    Updated_frame=cam.change_frame(frame_PIL) 
    
    stacked=torch.stack((Updated_frame,Updated_frame),0)
    if x==0:
        
        print(stacked.shape)
        
        x=1
        
    
    
    Final_frame=stacked.to(device)

    output_2=model(Final_frame)
    
    _,predictions=torch.max(output_2,1)
    
   
    
        
    print('the shape is :',predictions.shape)
    if predictions[1]== 0:
        print('Wood')
        
    elif  predictions[1]==1:
        print('Plastic')
    
    
    if cv2.waitKey(1)&0XFF==27:
        break
        
        
cv2.destroyAllWindows()



            
            
        
            
        
        
        

