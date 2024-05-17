import torch

matrix1 = torch.randn(3, 4, 2, 5)
matrix2 = torch.randn(3, 4, 2, 5)

result = torch.add(matrix1, matrix2)

print(result)
