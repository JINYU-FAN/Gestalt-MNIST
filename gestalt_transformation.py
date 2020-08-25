import torch
from matplotlib import pyplot as plt
import os
dtype = torch.uint8

def reverse(img):
    return (255 - img).to(dtype)

def half(img):
    size = img.shape[-1]
    mask = torch.zeros(size, size).to(DEVICE)
    for i in range(size):
        for j in range(size):
            if i>=size//2:
                mask[i][j] = 1.0
    return (img * mask + (255-img) * (1-mask)).to(dtype)

def quarter(img):
    size = img.shape[-1]
    mask = torch.zeros(size, size).to(DEVICE)
    for i in range(size):
        for j in range(size):
            if (i>=size//2 and j>=size//2) or (i<size//2 and j<size//2):
                mask[i][j] = 1
    return (img * mask + (255-img) * (1-mask)).to(dtype)

def continuity(img, n):
    size = img.shape[-1]
    mask = torch.zeros(size, size).to(DEVICE)
    for i in range(size):
        for j in range(size):
            if i % n == 0:
                mask[i][j] = 255
    return (img + mask).clamp(0, 255).to(dtype)

def closure(img, n):
    size = img.shape[-1]
    mask = torch.ones(size, size).to(DEVICE)
    for i in range(size):
        for j in range(size):
            if i % n == 0:
                mask[i][j] = 0
    return (img * mask).to(dtype)

def illusory(img, n):
    size = img.shape[-1]
    mask = torch.zeros(size, size).to(DEVICE)
    for i in range(size):
        for j in range(size):
            if i%n == 0:
                mask[i][j] = 255
    return (mask - img).clamp(0, 255).to(dtype)  

def illusory_complex(img, m, n):
    size = img.shape[-1]
    mask1 = torch.zeros(size, size).to(DEVICE)
    mask2 = torch.zeros(size, size).to(DEVICE)
    img_mask = (img > 0).int()
    for i in range(size):
        for j in range(size):
            if i%m == 0:
                mask1[i][j] = 255
            if j%n == 1:
                mask2[i][j] = 255
    return (img_mask * mask1 + (1-img_mask) * mask2).clamp(0,255).to(dtype)










if __name__ == '__main__':
    original_data = torch.load('./MNIST/original/test.pt')
    print(original_data[0][0])
    new_data = (half(original_data[0]), original_data[1])
    print(new_data[0][0])
    plt.imshow(new_data[0][0])
    plt.show()
    torch.save(new_data, './MNIST/half/test.pt')


