import torch
import torch.nn as nn


class TemporalLinkMatrix:
	def __init__(self, batch_size, mem_size, mem_dim):
		self.batch_size = batch_size
		self.mem_size = mem_size
		self.mem_dim = mem_dim
		
		self.L = torch.rand(batch_size, mem_size, mem_size)
		self.p = torch.zeros(batch_size, 1, mem_size)
		self.dij = np.diag_indices(mem_size, 2)
		self.ij = np.indices((mem_size, mem_size))

	def update_p(self, ww):
		self.p = (1 - ww[:,:,self.ij[0]].sum(-1)) * self.p.clone() + ww
		
	def update_L(self, ww):
		self.L = (1 - torch.eye(self.mem_size)) * self.L.clone()
		self.L[:,self.ij[0], self.ij[1]] = (1 - ww.squeeze(1)[:,self.ij[0]] - ww.squeeze(1)[:,self.ij[1]]) * self.L[:,self.ij[0],self.ij[0]].clone() + ww.squeeze(1)[:,self.ij[0]] * self.p.squeeze(1)[:,self.ij[1]].clone()



if __name__ == "__main__":
	batch_size = 2
	read_heads = 2
	mem_size = 4
	mem_dim = 3

	link_matrix = TemporalLinkMatrix(batch_size, mem_size, mem_dim)
	write_weighting = torch.rand(batch_size, 1, mem_size)

	print(link_matrix.ij[1])
	write_weighting = torch.Tensor([[[0.5226, 0.5320, 0.7195, 0.1745]],[[0.8679, 0.2953, 0.4556, 0.0334]]])
	print(link_matrix.p.shape)
	print(write_weighting.shape)
	link_matrix.update_L(write_weighting)
	# print(link_matrix.p.shape)
	# link_matrix.update_p(write_weighting)
	# print(link_matrix.p)

	# write_weighting = torch.Tensor([[[0.5226, 0.5320, 0.7195, 0.1745]],[[0.8679, 0.2953, 0.4556, 0.0334]]])
	# link_matrix.update_p(write_weighting)
	# link_matrix.update_L(write_weighting)
	# print(link_matrix.p)

	# write_weighting = torch.Tensor([[[0.5226, 0.5320, 0.7195, 0.1745]],[[0.8679, 0.2953, 0.4556, 0.0334]]])
	# link_matrix.update_p(write_weighting)
	# link_matrix.update_L(write_weighting)
	# print(link_matrix.p)