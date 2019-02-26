import os
import torch
from PIL import Image
from torchvision import models, transforms
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
import cv2
import json
import random

global net
global normalize
global preprocess
global features_blobs
global classes
global weight_softmax
labels_path='tools/labels.json'
idxs=[401,402,486,513,558,642,776,889]
names=['accordion','acoustic_guitar','cello','trumpet','flute','xylophone','saxophone','violin']

def returnCAM(feature_conv, weight_softmax, class_idx):
    size_upsample = (256, 256)
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam

def hook_feature(module, input, output):
    global features_blobs
    features_blobs=output.data.cpu().numpy()

def load_model():
    global net
    global normalize
    global preprocess
    global features_blobs
    global classes
    global weight_softmax
    net = models.densenet161(pretrained=True)
    finalconv_name = 'features'
    net.eval()
    net._modules.get(finalconv_name).register_forward_hook(hook_feature)
    params = list(net.parameters())
    weight_softmax = np.squeeze(params[-2].data.numpy())
    normalize = transforms.Normalize(
       mean=[0.485, 0.456, 0.406],
       std=[0.229, 0.224, 0.225]
    )
    preprocess = transforms.Compose([
       transforms.Resize((224,224)),
       transforms.ToTensor(),
       normalize
    ])
    classes = {int(key):value for (key, value)
              in json.load(open(labels_path,'r')).items()}
    if torch.cuda.is_available():
        net=net.cuda()

def get_CAM(imdir,imname):
    img_cv2 = cv2.imread(os.path.join(imdir,imname),cv2.IMREAD_COLOR)
    img_x = cv2.Sobel(img_cv2, cv2.CV_16S, 1, 0)
    absx = cv2.convertScaleAbs(img_x)
    img_reduce = np.sum(absx, axis=0)
    max_pos = np.argmax(img_reduce, axis=0)
    find_max = int(np.mean(max_pos))
    img_pair = [img_cv2[:, 0:find_max, :], img_cv2[:, find_max:img_cv2.shape[1], :]]
    prob_pair = []

    for i in range(2):
        image_PIL = Image.fromarray(cv2.cvtColor(img_pair[i], cv2.COLOR_BGR2RGB))
        img_tensor = preprocess(image_PIL)
        img_variable = Variable(img_tensor.unsqueeze(0))
        if torch.cuda.is_available():
            img_variable = img_variable.cuda()
        img = cv2.imread(os.path.join(imdir,imname))
        height, width, _ = img.shape
        logit = net(img_variable)
        h_x = F.softmax(logit, dim=1).data.squeeze()
        if torch.cuda.is_available():
            h_x=h_x.cpu()
        probs1 = h_x.numpy()
        probs=[]
        for i in range(0, 8):
            probs.append(probs1[idxs[i]])
        prob_pair.append(probs)
    return prob_pair

def Instru_from_image(imdir):
    imlist = os.listdir(imdir)
    load_model()
    #print(imlist)
    probs = np.zeros([2, 8])
    imlist_slice = random.sample(imlist, 3)
    for im in imlist_slice:
        probs1 = get_CAM(imdir, im)
        for i in range(2):
            probs[i] = probs[i]+np.array(probs1[i])
    max1=np.argmax(probs[0])
    max2=np.argmax(probs[1])
    #print(folders)
    #print(names[max1],names[max2])
    return [names[max1], names[max2]]


