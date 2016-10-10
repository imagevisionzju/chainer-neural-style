import numpy as np
import os
import argparse
from PIL import Image
import time
from chainer import cuda, Variable, optimizers, serializers
from net import *
#from lbfgs import LBFGS


def open_and_resize_image(path, target_width, model):
    image = Image.open(path).convert('RGB')
    width, height = image.size
    if width > target_width or height > target_width:
        ratio = float(width) / height
        #keep ratio
        if ratio < 1.0:
            targetsize = (int(round(ratio * target_width)), int(round(target_width)))
        else:
            targetsize = (int(round(target_width)), int(round(target_width / ratio)))
        image = image.resize(targetsize, Image.BILINEAR)

    return np.expand_dims(model.preprocess(np.asarray(image, dtype=np.float32), input_type='RGB'), 0)
def gram_matrix(y):
    b, ch, h, w = y.data.shape
    features = F.reshape(y, (b, ch, w*h))
    gram = F.batch_matmul(features, features, transb=True)/np.float32(ch*w*h)
    return gram

def total_variation(x):
    xp = cuda.get_array_module(x.data)
    b, ch, h, w = x.data.shape
    wh = Variable(xp.array([[[[0,-1,0],[0,0,0],[0,1,0]],[[0,-1,0],[0,0,0],[0,1,0]],[[0,-1,0],[0,0,0],[0,1,0]]]], dtype=xp.float32),volatile=x.volatile)
    ww = Variable(xp.array([[[[0,0,0],[-1,0,1],[0,0,0]],[[0,0,0],[-1,0,1],[0,0,0]],[[0,0,0],[-1,0,1],[0,0,0]]]], dtype=xp.float32),volatile=x.volatile)
    return F.sum(F.convolution_2d(x, W=wh,pad = 1) ** 2) + F.sum(F.convolution_2d(x, W=ww,pad = 1) ** 2)


parser = argparse.ArgumentParser(description='neural style transfer')
parser.add_argument('--gpu', '-g', default=0, type=int,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--style_image', '-s', type=str,default='style/teeth.jpg',
                    help='style image path')
parser.add_argument('--content_image', '-c', type=str,default='content/dog.jpg',
                    help='content image path')
parser.add_argument('--output_image', '-o', type=str,default='output/out.jpg',
                    help='output image name')
parser.add_argument('--maxWidthHeightContent', '-mwc', type=int, default=512,
                    help='max width and height of content_image(default value is 512)')
parser.add_argument('--maxWidthHeightStyle', '-mws', type=int, default=512,
                    help='max width and height of style image  (default value is 512)')
parser.add_argument('--lambda_tv', default=0, type=float,help='weight of total variation regularization.')
parser.add_argument('--lambda_feat', default=5, type=float)
parser.add_argument('--lambda_style', default=100, type=float)
parser.add_argument('--iternum', '-iter', default=500, type=int)
parser.add_argument('--lr', '-l', default=10.0, type=float)
parser.add_argument('--checkPoint', '-ck', default=10, type=int)
args = parser.parse_args()

lambda_tv = args.lambda_tv
lambda_f = args.lambda_feat
lambda_s = args.lambda_style

#load vgg
vgg = VGG()
serializers.load_npz('vgg16.model', vgg)
if args.gpu >= 0:
    cuda.get_device(args.gpu).use()
    vgg.to_gpu()
xp = np if args.gpu < 0 else cuda.cupy

#loading content image
print 'loading content...'
content_b = open_and_resize_image(args.content_image,args.maxWidthHeightContent,vgg)#get_image_to_xpArray(args.content_image,args.maxWidthHeight,args.batchsize)
feature_c = vgg(Variable(cuda.to_gpu(content_b,args.gpu), volatile=True))
print content_b.shape

#loading style image
print 'loading style...'
style_b = open_and_resize_image(args.style_image,args.maxWidthHeightStyle,vgg)#get_image_to_xpArray(args.style_image,args.maxWidthHeight,args.batchsize)
feature_s = vgg(Variable(cuda.to_gpu(style_b,args.gpu), volatile=True))
gram_s = [gram_matrix(y) for y in feature_s]
print style_b.shape

#init with content image (or random noise),same shape with content_b
optLink=chainer.Link(x=content_b.shape)
if args.gpu >= 0:
    optLink.to_gpu(args.gpu)
optLink.x.data[:] = xp.asarray(content_b)
#O = LBFGS(args.lr)
O = optimizers.Adam(alpha=args.lr)
O.setup(optLink)

print 'begen ...'
start = time.time()
for it in range(args.iternum):
    #vgg.zerograds() 
    optLink.cleargrads()

    #go through vgg network
    feature = vgg(optLink.x)

    #content loss
    L_feat  = lambda_f * F.mean_squared_error(feature[2],Variable(feature_c[2].data))

    #style loss
    L_style  = 1.0 * lambda_s * F.mean_squared_error(gram_matrix(feature[0]), Variable(gram_s[0].data))
    L_style += 1.0 * lambda_s * F.mean_squared_error(gram_matrix(feature[1]), Variable(gram_s[1].data))
    L_style += 1.0 * lambda_s * F.mean_squared_error(gram_matrix(feature[2]), Variable(gram_s[2].data))
    L_style += 1.0 * lambda_s * F.mean_squared_error(gram_matrix(feature[3]), Variable(gram_s[3].data))
    L =  L_style + L_feat# + L_tv

    #calc grad of input image
    L.backward()

    #update input image
    O.update()

    #show updated input image
    if it > 0 and it % args.checkPoint == 0:
        timeCost = time.time() - start
        if args.gpu >= 0:
            result = cuda.to_cpu(optLink.x.data)
        else:
            result = optLink.x.data
        resultImg = (result[0].transpose(1, 2, 0)+ vgg.mean).clip(0, 255).astype(np.uint8)
        #cv2.imshow('middleresult',resultImg)
        #cv2.waitKey(10)
        Image.fromarray(resultImg[:,:,::-1]).save('middle/out_{}.jpg'.format(it))
        print 'epoch {}/{}... L...{},L_feat...{},L_style...{}, cost {} sec'.format(it,args.iternum,L.data,L_feat.data,L_style.data,timeCost)
        start = time.time()

#save result
if args.gpu >= 0:
    result = cuda.to_cpu(optLink.x.data)
else:
    result = optLink.x.data
resultImg = (result[0].transpose(1, 2, 0)+ vgg.mean).clip(0, 255).astype(np.uint8)
#cv2.imwrite(args.output_image,resultImg)
Image.fromarray(resultImg[:,:,::-1]).save(args.output_image)
print 'end ...'
