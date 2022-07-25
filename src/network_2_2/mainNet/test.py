import torch

arr = torch.randint(1,10,size=(3,5,5))

print(arr)

arr = arr.view(3,-1)

print(arr)

arr = arr.view(3,5,5)

print(arr)


print(123)