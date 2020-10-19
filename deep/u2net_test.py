import os
from skimage import io, transform
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms#, utils
# import torch.optim as optim

import numpy as np
from PIL import Image
import glob

from .data_loader import RescaleT
from .data_loader import ToTensor
from .data_loader import ToTensorLab
from .data_loader import SalObjDataset

from .model import U2NET # full size version 173.6 MB
from .model import U2NETP # small version u2net 4.7 MB

from django.conf import settings #for Django image path

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
    img_name = image_name.split(os.sep)[-1]
    image = io.imread(image_name)
    imo = im.resize((image.shape[1],image.shape[0]),resample=Image.BILINEAR)

    pb_np = np.array(imo)
    RESCALE = 255
    out_img = pb_np/RESCALE
    THRESHOLD = 0.7
    shape = out_img.shape
    a_layer_init = np.ones(shape = (shape[0],shape[1],1))
    mul_layer = np.expand_dims(out_img[:,:,0],axis=2)
    a_layer = mul_layer*a_layer_init
    rgba_out = np.append(out_img,a_layer,axis=2)

 
    # refine the output
    out_img[out_img > THRESHOLD] = 1
    out_img[out_img <= THRESHOLD] = 0

    shape = out_img.shape
    a_layer_init = np.ones(shape = (shape[0],shape[1],1))
    mul_layer = np.expand_dims(out_img[:,:,0],axis=2)
    a_layer = mul_layer*a_layer_init
    rgba_out = np.append(out_img,a_layer,axis=2)

    #####################################################
    # Original Image
    #####################################################
    image_dir = image_dir#os.path.join(os.getcwd(), 'images')
    #names = [name[:-4] for name in os.listdir(image_dir)]
    #name = names[0]
    in_img = Image.open(image_dir)#+'/'+name+'.jpg')
    inp_img = np.array(in_img)
    inp_img =inp_img/ RESCALE
    a_layer = np.ones(shape = (shape[0],shape[1],1))
    rgba_inp = np.append(inp_img,a_layer,axis=2)
    rem_back = (rgba_inp*rgba_out)
    rem_back_scaled = Image.fromarray((rem_back*RESCALE).astype('uint8'), 'RGBA')
    background = Image.new("RGB", rem_back_scaled.size, (255, 255, 255))
    background.paste(rem_back_scaled, mask=rem_back_scaled.split()[3]) # 3 is the alpha channel
    #return rem_back_scaled
    new_name = im_object.Img.name
    initial_path = im_object.Img.path
    background.save(d_dir_s+ new_name)
    del d1,d2,d3,d4,d5,d6,d7
    #im_object.Img.name = 'images/removed_background.png'
    #neW_path = d_dir_s + im_object.Img.name
    #os.rename(initial_path, neW_path)
    #print("doooooooooooone")


    # aaa = img_name.split(".")
    # bbb = aaa[0:-1]
    # imidx = bbb[0]
    # for i in range(1,len(bbb)):
    #     imidx = imidx + "." + bbb[i]

    # imo.save(d_dir+imidx+'.png')

def main(im):

    # --------- 1. get image path and name ---------
    model_name='u2netp'# fixed as u2netp
    image_dir = im.Img.path

    #print("running main")
    #print(os.getcwd())
    #image_dir = os.path.join(os.getcwd(), 'images') # changed to 'images' directory which is populated while running the script
    prediction_dir = settings.MEDIA_ROOT+ '/images/' #E:/deep_app/media/images/' #settings.MEDIA_ROOT+ '/images' # changed to 'results' directory which is populated after the predictions
    #print(f"prediction directory : "  +prediction_dir)
    model_dir = os.path.join(os.getcwd()+'/deep/', model_name + '.pth') # path to u2netp pretrained weights

    img_name_list = [image_dir]#glob.glob('E:/deep_app/media/images/' + os.sep + '*')
    #print(img_name_list)
    p_s = settings.MEDIA_ROOT + '/'#'E:/deep_app/media/'

    # --------- 2. dataloader ---------
    #1. dataloader
    test_salobj_dataset = SalObjDataset(img_name_list = img_name_list,
                                        lbl_name_list = [],
                                        transform=transforms.Compose([RescaleT(320),
                                                                      ToTensorLab(flag=0)])
                                        )
    test_salobj_dataloader = DataLoader(test_salobj_dataset,
                                        batch_size=1,
                                        shuffle=False,
                                        num_workers=1)

    # --------- 3. model define ---------
    net = U2NETP(3,1)
    net.load_state_dict(torch.load(model_dir, map_location='cpu'))
    if torch.cuda.is_available():
        net.cuda()
    net.eval()

    # --------- 4. inference for each image ---------
    for i_test, data_test in enumerate(test_salobj_dataloader):

        #print("inferencing:",img_name_list[i_test].split(os.sep)[-1])
        image_name = img_name_list[i_test].split(os.sep)[-1]
        inputs_test = data_test['image']
        inputs_test = inputs_test.type(torch.FloatTensor)

        if torch.cuda.is_available():
            inputs_test = Variable(inputs_test.cuda())
        else:
            inputs_test = Variable(inputs_test)

        d1,d2,d3,d4,d5,d6,d7= net(inputs_test)

        # normalization
        pred = d1[:,0,:,:]
        pred = normPRED(pred)
        
        # save results to test_results folder
        if not os.path.exists(prediction_dir):
            os.makedirs(prediction_dir, exist_ok=True)
        save_output(img_name_list[i_test],pred,prediction_dir, image_dir, im, p_s, d1,d2,d3,d4,d5,d6,d7)

        




if __name__ == "__main__":
    main()
