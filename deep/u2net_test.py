import os
from skimage import io, transform
import torch
#import torchvision
from torch.autograd import Variable
#import torch.nn as nn
#from torch.utils.data import DataLoader
from torchvision import transforms

import numpy as np
from PIL import Image
#import glob
import cv2

# from .data_loader import RescaleT
# from .data_loader import ToTensor
# from .data_loader import ToTensorLab
# from .data_loader import SalObjDataset

from .model import U2NETP # small version u2net 4.7 MB

from django.conf import settings # Django image path

# normalize the predicted SOD probability map
def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d-mi)/(ma-mi)

    return dn

def save_output(image_name,pred,d_dir, image_dir,im, d_dir_s,d1,d2,d3,d4,d5,d6,d7):
    im_object = im
    predict = pred
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()

    im = Image.fromarray(predict_np*255).convert('RGB')
    #img_name = image_name.split(os.sep)[-1]
    image = io.imread(image_name)
    imo = im.resize((image.shape[1],image.shape[0]),resample=Image.BILINEAR)

    pb_np = np.array(imo)
    RESCALE = 255
    out_img = pb_np/RESCALE
    alpha = out_img.astype(float)
    THRESHOLD = 0.7
    shape = out_img.shape

 
    # refine the output
    alpha[alpha > THRESHOLD] = 1
    alpha[alpha <= THRESHOLD] = 0


    #####################################################
    # Original Image
    #####################################################
    image_dir = image_dir
    foreground = cv2.imread(image_dir)
    action= str(im_object.action)
    if action=='1':
        # Remove background
        foreground = foreground.astype(float)
        mask_out=cv2.subtract(alpha,foreground)
        mask_out=cv2.subtract(alpha,mask_out)
        mask_out[alpha == 0] = 255
    elif action=='2':
        # Blur background
        blurredImage = cv2.GaussianBlur(foreground, (7,7), 0)
        blurredImage = cv2.GaussianBlur(foreground, (7,7), 0)
        blurredImage = blurredImage.astype(float)
        foreground = foreground.astype(float)
        alpha = cv2.GaussianBlur(alpha, (7,7),0)
        foreground = cv2.multiply(alpha, foreground)
        background = cv2.multiply(1.0 - alpha, blurredImage)
        mask_out = cv2.add(foreground, background)
    else:
        # Black and white background
        background = cv2.cvtColor(foreground, cv2.COLOR_BGR2GRAY)
        background = cv2.cvtColor(background, cv2.COLOR_GRAY2RGB)
        foreground = foreground.astype(float)
        background = background.astype(float)
        alpha = cv2.GaussianBlur(alpha, (7,7),0)
        foreground = cv2.multiply(alpha, foreground)
        background = cv2.multiply(1.0 - alpha, background)
        mask_out = cv2.add(foreground, background)
    new_name = im_object.Image.name
    initial_path = im_object.Image.path
    cv2.imwrite(d_dir_s+ new_name, mask_out)
    del d1,d2,d3,d4,d5,d6,d7


def main(im):

    # --------- 1. get image path and name ---------
    model_name='u2netp'
    image_dir = im.Image.path
    prediction_dir = settings.MEDIA_ROOT+ '/images/'
    model_dir = os.path.join(os.getcwd()+'/deep/', model_name + '.pth') 

    img_name_list = image_dir
    p_s = settings.MEDIA_ROOT + '/'

    # --------- 3. model define ---------
    net = U2NETP(3,1)
    net.load_state_dict(torch.load(model_dir, map_location='cpu'))
    if torch.cuda.is_available():
        net.cuda()
    net.eval()

    # --------- 4. inference for each image ---------
  
    read_image = Image.open(image_dir).convert('RGB')
    read_image = np.array(read_image)
    tran = transforms.ToTensor()

    #torch_image = torch.from_numpy(read_image)
    torch_image = tran(read_image)
    torch_image = torch_image.unsqueeze(0)
    print(torch_image.shape)
    inputs_test = torch_image.type(torch.FloatTensor)

    if torch.cuda.is_available():
        inputs_test = Variable(inputs_test.cuda())
    else:
        inputs_test = Variable(inputs_test, volatile=True)

    with torch.no_grad():
            d1,d2,d3,d4,d5,d6,d7= net(inputs_test)

    # normalization
    pred = d1[:,0,:,:]
    pred = normPRED(pred)
    
    # save results to test_results folder
    if not os.path.exists(prediction_dir):
        os.makedirs(prediction_dir, exist_ok=True)
    save_output(img_name_list,pred,prediction_dir, image_dir, im, p_s, d1,d2,d3,d4,d5,d6,d7)


if __name__ == "__main__":
    main()
