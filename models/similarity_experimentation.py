import torch

import numpy as np
import matplotlib.pyplot as plt

# a = torch.randn((4,4))
# b = torch.randn((4,4))
# c = torch.randn((4,4))
# d = torch.randn((4,4))

# torch.save(a, 'a.pt')
# torch.save(b, 'b.pt')
# torch.save(c, 'c.pt')
# torch.save(d, 'd.pt')
# def cn(m1, m2):
#     nn1 = torch.sum(m1+m1, dim = 1)/2
#     nn2 = torch.sum(m2+m2, dim = 1)/2
#     print('nn: ', nn.shape, nn)

#     na = torch.unsqueeze(nn1, dim = 0)
#     nb = torch.unsqueeze(nn2, dim = 1)

#     nc = na+nb

# a = torch.load('a.pt')
# b = torch.load('b.pt')
# c = torch.load('c.pt')
# d = torch.load('d.pt')

# # print(a)
# # print(b)

# # print(a + b)
# # print(torch.sigmoid(a) + torch.sigmoid(b))

# # print((a-torch.mean(a)) + (b-torch.mean(b)))

# cos_sim = torch.nn.CosineSimilarity()
# a = torch.tensor([[1,2,3,4],
#      [5,6,7,8],
#      [9,10,11,12]]).float()

# b = torch.tensor([[9,10,,4],
#      [5,6,7,8],
#      [9,10,11,12]]).float()

# anorm = np.linalg.norm(a, axis=1)
# print('Norm: ', anorm)
# print('Outer: ', np.outer(anorm, anorm))
# print(cos_sim(a, a))

# nn = torch.sum(a+a, dim = 1)/2
# nn1 = torch.sum(b+b, dim = 1)/2
# print('nn: ', nn.shape, nn)

# na = torch.unsqueeze(nn, dim = 0)
# nb = torch.unsqueeze(nn1, dim = 1)

# nc = na+nb
# print('nc: ', nc.shape, nc)



a = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]).float()
b = torch.tensor([[11, 23, 12, 13, 16, 14, 15, 18, 21, 23]]).float()

print('----: ', a@b.transpose(-2, -1) * (10**-0.5))
print('----: ', a@a.transpose(-2, -1) * (10**-0.5))
print('----: ', b@b.transpose(-2, -1) * (10**-0.5))

print('self add a: ', a+a, ' ', torch.sum(a+a))
print('self add b: ', b+b, ' ', torch.sum(b+b))
print('cross add: ', a+b, ' ', torch.sum(a+b))

a1 = a / 2
b1 = b / 2

print('self add: ', a1+a1, ' ', torch.sum(a1+a1)/10)
print('cross add: ', a1+b1, ' ', torch.sum(a1+b1)/10)

a2 = a*((torch.max(a))**(-a))
b2 = b*((torch.max(b))**(-a))

print('self add a: ', a2+a2, ' ', torch.sum(a2+a2))
print('self add b: ', b2+b2, ' ', torch.sum(b2+b2))
print('cross add: ', a2+b2, ' ', torch.sum(a2+b2))

a2 = (a)*((torch.max(a)+1)**(-a))
b2 = (b)*((torch.max(b)+1)**(-a))

print('--- just arrays: ', a2)
print('--- just arrays: ', a2)

print('self add a: ', a2+a2, ' ', torch.sum(a2+a2)/(torch.max(a2) + torch.max(a2)))
print('self add b: ', b2+b2, ' ', torch.sum(b2+b2)/(torch.max(b2) + torch.max(b2)))
print('cross add: ', a2+b2, ' ', torch.sum(a2+b2)/(torch.max(a2) + torch.max(b2)))

an = 1-a
bn = 1-b

print('self add a: ', an+an, ' ', torch.sum(an+an))
print('self add b: ', bn+bn, ' ', torch.sum(bn+bn))
print('cross add: ', an+bn, ' ', torch.sum(an+bn))

