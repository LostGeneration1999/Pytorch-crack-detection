# -*- coding: utf-8 -*-
from utils.my_dataloader import ImageTransform, make_datapath_list, HymenopterDataset
import os
import glob
import shutil
from PIL import Image, ImageDraw
import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision
from torch.autograd import Variable #自動微分用
from torchvision import models,transforms

def build_model() :
    input_tensor = Input(shape=(img_rows, img_cols, 3))
    vgg16 = VGG16(include_top=False, weights='imagenet', input_tensor=input_tensor)

    _model = Sequential()

    _model.add(Flatten(input_shape=vgg16.output_shape[1:]))
    _model.add(Dense(256, activation='relu'))
    _model.add(Dropout(0.5))
    _model.add(Dense(nb_classes, activation='softmax'))

    model = Model(inputs=vgg16.input, outputs=_model(vgg16.output))
    
    for layer in model.layers[:15]:
        layer.trainable = False

    model.compile(loss='categorical_crossentropy',
                  optimizer=SGD(lr=1e-4, momentum=0.9), metrics=['accuracy'])
    return model

def binary(ftitle, ext):
    print('crack/'+ftitle+ext)
    img_color = cv2.imread('crack/'+ftitle+ext)
    #img = cv2.medianBlur(img,5)

    img = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
                                cv2.THRESH_BINARY,11,20)
    ret,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
    #th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
    #                            11,15)
    th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                                cv2.THRESH_BINARY,11,15)

    titles = ['Original Image', 'Global Thresholding (v = 127)',
              'Adaptive_Mean', 'Adaptive_Gaussian']
    images = [img, th1, th2, th3]
    h,w = img.shape

    mask = images[2]
    #mask = cv2.bitwise_not(img3)
    img3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    img3[np.where((img3==[0,0,0]).all(axis=2))] = [0,0,255]
    #cv2.imwrite('binary_images/'+ftitle+'_'+titles[2]+ext,img3)

    dst = np.zeros((h,w,3),dtype='uint8')
    for y in range(0, h):
        for x in range(0,w):
            if (img3[y][x] == 255).all(): dst[y][x] = img_color[y][x]
            else: dst[y][x]=[0,0,255]
    
    #cv2.imwrite('binary_images/'+ftitle+'_'+titles[2]+ext,dst)
    #cv2.imwrite('binary_images/'+ftitle+'_'+ext,img_color)
    return dst


def convert2(size, box):
    dw = 1./size[0]
    dh = 1./size[1]
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = abs(x*dw)
    x = ('%06.6f' % x)
    w = abs(w*dw)
    w = ('%06.6f' % w)
    y = abs(y*dh)
    y = ('%06.6f' % y)
    h = abs(h*dh)
    h = ('%06.6f' % h)
    return (x,y,w,h)

def convert(size, elems):
    
    dw = 1./size[0]
    dh = 1./size[1]
    #print('width: %d, height: %d' % (size[0], size[1]))
    
    #center_point
    x, y, w, h = float(elems[1]),float(elems[2]),float(elems[3]),float(elems[4])     
    x = abs(x/dw)
    x = float('%06.6f' % x)
    y = abs(y/dh)
    y = float('%06.6f' % y)
    w = abs(w/dw)
    w = float('%06.6f' % w) 
    h = abs(h/dh)
    h = float('%06.6f' % h)
    
    length = h if h >= w else w
    length = float(length)
    #print('length: %f' % length)


    left = x - length/2.0
    right = x + length/2.0

    top = y - length/2.0
    bottom = y + length/2.0
    
    return (left, right, top, bottom)

imsize = 50
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
loader = transforms.Compose([transforms.Scale(imsize), transforms.ToTensor()])
def image_loader(image_name):
    """load image, returns cuda tensor"""
    image = Image.open(image_name)
    image = loader(image).float()
    image = transforms.Normalize(mean, std)(image)
    image = Variable(image, requires_grad=True)
    image = image.unsqueeze(0)  #this is for VGG, may not be needed for ResNet
    return image  #assumes that you're using GPU  

#異常箇所プロット設定
#img=resultに格納された分割画像
#point=正規化された異常検知領域座標
def coloring(red_img, img, point, ftitle):
    back_im = img.copy()
    back_im.paste(red_img,(point[0], point[1])) 
    back_im.save('cut_convert/new/%s.jpg'%ftitle, quality=95)
    print('plot to image...')
    return back_im

classes = ['crack', 'black']
nb_classes = len(classes)
img_rows, img_cols = 100, 100
#img_rows, img_cols = 75, 75

net = models.vgg16(pretrained=True)
net.classifier[6] = nn.Linear(in_features=4096, out_features=2)
load_path = './weights_fine_tuning.pth'
load_weights = torch.load(load_path)
net.load_state_dict(load_weights)
net = net.eval()

os.chdir('cut_convert')
files = glob.glob('*.jpg')
new_dir_path = 'cut_crack'
if not (os.path.exists(new_dir_path)):
    os.makedirs(new_dir_path)

new_dir_path2 = 'new'
if not (os.path.exists(new_dir_path2)):
    os.makedirs(new_dir_path2)
os.chdir('../')
for f in files:
    ftitle, fext = os.path.splitext(f)
    txt_file = open('cut_convert/%s.txt'%(ftitle))
    
    img = Image.open('cut_convert/%s.jpg'%(ftitle))
    lines = txt_file.read().replace("\r\n","\n").split('\n')
    
    ct = 0
    for line in lines:
        if(len(line) >= 2):
            ct = ct + 1
            box = line.split(' ')
            
            print(ftitle)
            w, h= img.size           
            
            x, y = box[1], box[2]
            
            width, height = box[3], box[4]
            print('x: %f , y: %f , w: %f , h: %f' % (float(box[1]), float(box[2]), float(box[3]), float(box[4])))
            
            
            bbox = convert((w,h), box)
            left, right, top, bottom = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
            print('left: %f , right: %f , top: %f , bottom: %f' % (left, right, top, bottom))
            #cut_img = img[int(top):int(bot), int(left):int(right)]
            cut_img = img.crop((int(left),int(top),int(right),int(bottom))).resize((50, 50))
            #cv2.imwrite('%s/%s.jpg'%(new_dir_path, line), cut_img)
            filename = 'cut_convert/%s/%s_%s.jpg'%(new_dir_path, ftitle, ct)
            cut_img.save(filename)

            #filename = "%s/%s.jpg"%(new_dir_path, line)

            image = image_loader(filename)
            print(filename)
            out = net(image).cpu().data
            y = np.argmax(out.numpy())
            
            print("Name: ", classes[y], y)
            
            if y == 0:
                savepath = 'crack/%s_%s.jpg'%(ftitle, ct)
                cut_img.save(savepath)
                
                new_file = open('cut_convert/new/%s.txt'%(ftitle),'a')
                box = (float(left), float(right), float(bottom), float(top))
                point_list = [int(left), int(top), int(right), int(bottom)]
                print(point_list)
                print(img)

                red_img = binary('%s_%s'%(ftitle,ct), '.jpg')
                red_img = Image.fromarray(red_img)
                red_img.save('binary_image/%s.jpg'%ftitle, quality=95)
                
                coloring(img, red_img, point_list, ftitle)
                #img.save('cut_convert/new/%s.jpg'%ftitle)
                
                bbox = convert2((w,h), box)
                new_file.write(str(y) + " " + " "\
		               .join([str(a) for a in bbox]) + '\n')
                new_file.close()

                
            elif y == 1:
                savepath = 'black/%s_%s.jpg'%(ftitle, ct)
                cut_img.save(savepath)
         

            if(ct != 0):
                continue
    img.save('cut_convert/new/%s.jpg'%(ftitle))
#shutil.rmtree(new_dir_path)
