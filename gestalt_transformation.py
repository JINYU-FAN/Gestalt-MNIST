import torch
from matplotlib import pyplot as plt
import os
import cv2
dtype = torch.uint8

def reverse(img):
    return (255 - img).to(dtype)

def half(img):
    size = img.shape[-1]
    mask = torch.zeros(size, size)
    for i in range(size):
        for j in range(size):
            if i>=size//2:
                mask[i][j] = 1.0
    return (img * mask + (255-img) * (1-mask)).to(dtype)

def quarter(img):
    size = img.shape[-1]
    mask = torch.zeros(size, size)
    for i in range(size):
        for j in range(size):
            if (i>=size//2 and j>=size//2) or (i<size//2 and j<size//2):
                mask[i][j] = 1
    return (img * mask + (255-img) * (1-mask)).to(dtype)

def continuity(img, n):
    size = img.shape[-1]
    mask = torch.zeros(size, size)
    for i in range(size):
        for j in range(size):
            if i % n == 0:
                mask[i][j] = 255
    return (img + mask).clamp(0, 255).to(dtype)

def closure(img, n):
    size = img.shape[-1]
    mask = torch.ones(size, size)
    for i in range(size):
        for j in range(size):
            if i % n == 0:
                mask[i][j] = 0
    return (img * mask).to(dtype)

def illusory(img, n):
    size = img.shape[-1]
    mask = torch.zeros(size, size)
    for i in range(size):
        for j in range(size):
            if i%n == 0:
                mask[i][j] = 255
    return (mask - img).clamp(0, 255).to(dtype)  

def illusory_complex(img, m, n):
    size = img.shape[-1]
    mask1 = torch.zeros(size, size)
    mask2 = torch.zeros(size, size)
    img_mask = (img > 0).int()
    for i in range(size):
        for j in range(size):
            if i%m == 0:
                mask1[i][j] = 255
            if j%n == 0:
                mask2[i][j] = 255
    return (img_mask * mask1 + (1-img_mask) * mask2).clamp(0,255).to(dtype)


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
    return edges.to(dtype)


def illusory_edge(img, m, n, w):
    edges = edge(img, w)
    size = img.shape[-1]
    mask1 = torch.zeros(size, size)
    mask2 = torch.zeros(size, size)
    img_mask = (img > 0).int()
    for i in range(size):
        for j in range(size):
            if i%m == 0:
                mask1[i][j] = 255
            if j%n == 1:
                mask2[i][j] = 255
    return ((edges==0).int()*(img_mask * mask1 + (1-img_mask) * mask2).clamp(0,255)).to(dtype)


def proximity(img, w):
    mask = torch.zeros(28, 28)
    for i in range(28):
        for j in range(28):
            if i%2==0 and j%2==0:
                mask[i][j] = 255
    edges = edge(img, w)
    return ((edges==0).int()*mask).to(dtype)

if __name__ == '__main__':
    dataset = 'FashionMNIST'
    def save(data, path):
        dir_path = f'./{dataset}/'+path
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
            torch.save(data, dir_path+'/test.pt')
    original_data = torch.load(f'./{dataset}/original/test.pt')

    new_data = (illusory_complex(original_data[0], 2,2), original_data[1])
    save(new_data, 'illusory_complex_2_2_r')

    


