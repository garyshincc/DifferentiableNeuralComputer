import torch
import torch.nn as nn

from memory import memory, BatchCosineSimilarity
from temporal_link_matrix import TemporalLinkMatrix
from controller import Controller
from dnc import DifferentiableNeuralComputer


class MemoryNetwork(nn.Module):
	def __init__(self, input_size, hidden_size, output_size, mem_size, mem_dim, read_heads):
		super(MemoryNetwork, self).__init__()
		self.dnc = DifferentiableNeuralComputer(input_size, hidden_size, output_size, mem_size, mem_dim, read_heads)

	def forward(self, x, prev_state, memory, link_matrix):
		memory, link_matrix, controls, read_vector = self.dnc(x, prev_state, memory, link_matrix)
		return memory, link_matrix, controls, read_vector
