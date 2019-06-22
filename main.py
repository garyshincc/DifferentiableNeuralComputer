import torch
import torch.nn as nn
import numpy as np
from pprint import pprint


input_size = 1
hidden_size = 4
output_size = 1
batch_size = 1
read_heads = 1
mem_size = 8
mem_dim = 1

from pprint import pprint
import torch.optim as optim

memory = Memory(batch_size, mem_size, mem_dim)
link_matrix = TemporalLinkMatrix(batch_size, mem_size, mem_dim)
mem_model = MemoryNetwork(input_size, hidden_size, output_size, mem_size, mem_dim, read_heads)
criterion = nn.MSELoss()
optimizer = optim.Adam(mem_model.parameters(), lr=0.05)


# x = torch.Tensor([[[0.1], [0.9], [0.2], [0.8], [1], [2], [3], [4]]])
# y = torch.Tensor([[[1], [2], [3], [4]]])
num_epochs = 20000
xsig = torch.Tensor([[[1], [2], [3], [4]]])
for epoch in range(num_epochs):
	x = torch.zeros(1, 8, 1)
	y = torch.rand(1, 4, 1)
	x[:,0:4] = y
	x[:,4:] = xsig
	
	optimizer.zero_grad()
	memory = Memory(batch_size, mem_size, mem_dim)
	link_matrix = TemporalLinkMatrix(batch_size, mem_size, mem_dim)
	controls = None
	pred = list()
	for i in range(8):
		if i < 4:
			memory, link_matrix, controls, read_vector = mem_model(x[:,i], controls, memory, link_matrix)
		else:
			memory, link_matrix, controls, read_vector = mem_model(x[:,i], controls, memory, link_matrix)
			pred.append(read_vector)
	pred = torch.stack(pred, dim=1).view(y.shape)
	loss = criterion(pred, y)
	loss.backward(retain_graph=True)
	optimizer.step()
#	 break
	if epoch % 500 == 0:
		print(f"loss: {loss.item()}")
		print(f"y: {y}")
		print(f"pred: {pred}")
		print(f"memory: {memory}")
		pprint(controls)
# print(x)
# print(y)