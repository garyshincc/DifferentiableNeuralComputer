class Oneplus(nn.Module):
	def __init__(self):
		super(Oneplus, self).__init__()
		self.softplus = nn.Softplus()

	def forward(self, x):
		return 1 + self.softplus(x)

class Controller(nn.Module):
	def __init__(self, input_size, hidden_size, output_size, mem_size, mem_dim, read_heads):
		super(Controller,self).__init__()
		self.input_size = input_size
		self.output_size = output_size
		self.mem_size = mem_size
		self.mem_dim = mem_dim
		self.read_heads = read_heads
		
		self.oneplus = Oneplus()
		self.sigmoid = nn.Sigmoid()
		'''
		read keys: read_heads x mem_dim
		read strengths: (oneplus) read_heads x 1
		write key: 1 x mem_dim
		write strength: (oneplus) 1 x 1
		
		erase vector: (sigmoid) 1 x mem_dim
		write vector: 1 x mem_dim
		free gates: (sigmoid) read_heads x 1
		allocation gate: (sigmoid) 1 x 1
		write gate: (sigmoid) 1 x 1
		read modes: (softmax) read_heads x 3
		'''
		self.fc1 = nn.Linear(input_size, hidden_size)
		self.fc2 = nn.Linear(input_size, output_size)
		
		self.fc_read_keys = nn.Linear(hidden_size, self.read_heads * self.mem_dim)
		self.fc_read_strengths = nn.Linear(hidden_size, self.read_heads)
		self.fc_write_keys = nn.Linear(hidden_size, self.mem_dim)
		self.fc_write_strength = nn.Linear(hidden_size, 1)
		self.fc_erase_vector = nn.Linear(hidden_size, self.mem_dim)
		self.fc_write_vector = nn.Linear(hidden_size, self.mem_dim)
		self.fc_free_gates = nn.Linear(hidden_size, self.read_heads)
		self.fc_allocation_gate = nn.Linear(hidden_size, 1)
		self.fc_write_gate = nn.Linear(hidden_size, 1)
		self.fc_read_modes = nn.Linear(hidden_size, self.read_heads * 3)

		
	def forward(self, x):
		controls = dict()
		output = torch.tanh(self.fc2(x))
		x = torch.tanh(self.fc1(x))
		
		controls["read_keys"] = self.fc_read_keys(x).view(-1, self.read_heads, self.mem_dim)
		controls["read_strengths"] = self.oneplus(self.fc_read_strengths(x)).view(-1, self.read_heads, 1)
		controls["write_key"] = self.fc_write_keys(x).view(-1, 1, self.mem_dim)
		controls["write_strength"] = self.oneplus(self.fc_write_strength(x)).view(-1, 1, 1)
		controls["erase_vector"] = self.sigmoid(self.fc_erase_vector(x)).view(-1, 1, self.mem_dim)
		controls["write_vector"] = self.fc_write_vector(x).view(-1, 1, self.mem_dim)
		controls["free_gates"] = self.sigmoid(self.fc_free_gates(x)).view(-1, self.read_heads, 1)
		controls["allocation_gate"] = self.sigmoid(self.fc_allocation_gate(x)).view(-1, 1, 1)
		controls["write_gate"] = self.sigmoid(self.fc_write_gate(x)).view(-1, 1, 1)
		controls["read_modes"] = torch.softmax(self.fc_read_modes(x).view(-1, self.read_heads, 3), -1)
		controls["output_vector"] = output.view(-1, 1, output_size)
		return controls


if __name__ == "__main__":
	input_size = 4
	hidden_size = 8
	output_size = 3
	batch_size = 1
	read_heads = 2
	mem_size = 4
	mem_dim = 4

	x = torch.rand(batch_size, input_size)
	controller = Controller(input_size, hidden_size, output_size, mem_size, mem_dim, read_heads)
	controls = controller(x)
	for key, value in controls.items():
	  print(f"{key}: {value.shape}")
	'''
	read keys: read_heads x mem_dim
	read strengths: (oneplus) read_heads x 1
	write key: 1 x mem_dim
	write strength: (oneplus) 1 x 1
	erase vector: (sigmoid) 1 x mem_dim
	write vector: 1 x mem_dim
	free gates: (sigmoid) read_heads x 1
	allocation gate: (sigmoid) 1 x 1
	write gate: (sigmoid) 1 x 1
	read modes: (softmax) read_heads x 3
	'''