import torch
import torch.nn as nn

class Memory:
	def __init__(self, batch_size, mem_size, mem_dim):
		self.batch_size = batch_size
		self.mem_size = mem_size
		self.mem_dim = mem_dim
		self.memory = torch.zeros(batch_size, mem_size, mem_dim)

	def write(self, write_attn, erase_vector, write_vector):
		erase_mem = self.memory * (1 - torch.bmm(erase_vector.transpose(1,2), write_attn)).transpose(1,2)
		write_mem = torch.bmm(write_vector.transpose(1,2), write_attn).transpose(1,2)
		self.memory = erase_mem + write_mem
		
	def read(self, read_attn):
		read_vector = torch.bmm(read_attn, self.memory)
		return read_vector
	
	def __str__(self):
		return str(self.memory)

class BatchCosineSimilarity(nn.Module):
	def __init__(self):
		super(BatchCosineSimilarity, self).__init__()

	def forward(self, batch_mem, batch_x):
		# read key: b x r x d
		# write key: b x 1 x d
		# memory:	b x n x d
		# bmm ( memory , read_key.tranpose(1,2))
		# b x n x r/1
		dot = torch.bmm(batch_mem, batch_x.transpose(1,2)).sum(1)
		mul_norm = torch.norm(batch_mem, 2, dim=-1).unsqueeze(-1) * torch.norm(batch_x, 2, dim=-1).unsqueeze(1)
		sim = dot.unsqueeze(1) / (mul_norm + 1e-4)
		sim = torch.softmax(sim, dim=1)
		return sim


if __name__ == "__main__":
	batch_size = 2
	read_heads = 2
	mem_size = 4
	mem_dim = 3
	memory = Memory(batch_size, mem_size, mem_dim)
	print(memory)
	write_key = torch.rand(batch_size, 1, mem_size)
	print(f"write_key: {write_key}")
	read_key = torch.rand(batch_size, read_heads, mem_size)
	print(f"read_key: {read_key.shape}")
	read_vector = memory.read(read_key)
	read_vector.shape