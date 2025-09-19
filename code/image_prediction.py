import numpy as np
from PIL import Image

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import torch
import torch.nn as nn

import math

import segmentation_models_pytorch as smp



# 기본적으로 있는 모드
class_name=["건물",
            "주차장","비닐하우스",
            "나지","도로","가로수",
            "밭","산림","농경지","비대상지"]

evaluate=[-1,-1,-1,-1,-1,1,1,1,1,0]
model_path = "F:/kaist/4/sw_acp/climate_prediction/code/best_epoch-05.bin"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
backbone = 'efficientnet-b0'
num_classes = 10
size = 0

def load_img_bypath(path):
    image_path = path
    img = Image.open(image_path)
    w,h=img.size
    exponent = int(int(math.log(w*h,2))/2)
    sl = 2**max(5,exponent)
    img = img.resize((sl,sl))
    img = np.array(img)
    
    img = img.astype('float32')
    mx = np.max(img)
    if mx:
        img/=mx
        
    img = np.transpose(img,(2, 0, 1)) # shape 의 0,1,2번째를 순서를 바꿈
    #print(img.shape)
    img = torch.tensor([img])
    
    return img,sl

def build_model():
    model = smp.Unet(
        encoder_name=backbone,      # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
        in_channels=3, #5로 바꿀수있는지 해보기                 # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=num_classes,        # model output channels (number of classes in your dataset)
        activation=None,
    )
    model.to(device)
    return model

def load_model(path):
    model = build_model()
    model.load_state_dict(torch.load(path))
    model.eval()
    return model

def predict_bypath(PATH):
    global size
    img,size = load_img_bypath(path=PATH)

    preds = []
    for fold in range(1):
        model = load_model(model_path)
        with torch.no_grad():
            pred = model(img.to(device, dtype=torch.float))
            pred = (nn.Sigmoid()(pred)>0.5).double()
        preds.append(pred)

    img  = img.cpu().detach().numpy()
    preds = torch.mean(torch.stack(preds, dim=0), dim=0).cpu().detach().numpy()

    return preds


def Synthesis_img_mask_bypath(path): # 0<=idx<=9
    
    return_data = []
    mask = predict_bypath(path)[0]
    for i in range(10):
        unique,counts = np.unique(mask[i],return_counts=True)
        dt = dict(zip(unique,counts)) # dt 는 이미지 내 해당 픽셀이 어떤 클래스로 되어있는지를 나타낸 dict
        
        #print(dt)
        distribution=[0,0] 
        if 0 in dt:
            distribution[0]+= dt[0]
            
        if 1 in dt:
            distribution[1]+=dt[1]
            
        return_data.append(distribution)
            
        print(f"{class_name[i]} : {distribution}")
        
        
    bad_percent=0
    good_percent=0
    
    #print(return_data)
    
    for i in range(5):
        bad_percent+=return_data[i][1]
    for i in range(5,9):
        good_percent+=return_data[i][1]
        
        
    marking = bad_percent+good_percent
    if marking==0:return -1, -1
    bad_percent/=(bad_percent+good_percent)
    good_percent/=(bad_percent+good_percent)
    bad_percent*=100
    good_percent*=100
    
    print((marking)/(size*size),marking,size)
    if (marking)/(size*size) <= 0.15:
        return -1, -1
    
    if bad_percent>=70:
        return 4,bad_percent # 최고 위험
    elif bad_percent>=45:
        return 3,bad_percent # 중
    elif bad_percent>=10:
        return 2,bad_percent # 하
    else:
        return 1,bad_percent # x
        
        
    
        
        
    #return return_data
        