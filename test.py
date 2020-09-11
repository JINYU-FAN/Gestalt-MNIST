import torch
from matplotlib import pyplot as plt
import os
import cv2
dtype = torch.uint8

def edge(imgs, w):
    imgs = (imgs>0).int()*255.0 
    g=cv2.getStructuringElement(cv2.MORPH_RECT,(w,w))
    edges = torch.zeros(10000, 28, 28)
    for i in range(10000):
        img = imgs[i]
        img_erode=cv2.erode(img.numpy(),g)
        img_dilate=cv2.dilate(img.numpy(),g)
        edge=img_dilate-img.numpy()
        edges[i] = torch.Tensor(edge)
    return edges

if __name__ == '__main__':
    def save(data, path):
        dir_path = './MNIST/'+path
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
            torch.save(data, dir_path+'/test.pt')
    original_data = torch.load('./MNIST/original/test.pt')
    new_data = (edge(original_data[0], 7), original_data[1])
    save(new_data, 'edge_7')
