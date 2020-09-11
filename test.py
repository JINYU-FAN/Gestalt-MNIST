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

def proximity(img, w):
    mask = torch.zeros(28, 28)
    for i in range(28):
        for j in range(28):
            if i%2==0 and j%2==0:
                mask[i][j] = 255
    edges = edge(img, w)
    return (edges==0).int()*mask
                

if __name__ == '__main__':
    def save(data, path):
        dir_path = './FashionMNIST/'+path
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
            torch.save(data, dir_path+'/test.pt')
    original_data = torch.load('./FashionMNIST/original/test.pt')
    new_data = (proximity(original_data[0], 5), original_data[1])
    #plt.imshow(new_data[0][1])
    #plt.savefig('te.png')
    save(new_data, 'proximity5')
