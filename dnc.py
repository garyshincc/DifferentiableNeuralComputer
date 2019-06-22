import torch
import torch.nn as nn

from memory import memory, BatchCosineSimilarity
from temporal_link_matrix import TemporalLinkMatrix
from controller import Controller


class DifferentiableNeuralComputer(nn.Module):
  def __init__(self, input_size, hidden_size, output_size, mem_size, mem_dim, read_heads):
    super(DifferentiableNeuralComputer, self).__init__()
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.output_size = output_size
    self.mem_size = mem_size
    self.mem_dim = mem_dim
    self.read_heads = read_heads
    
    self.similarity = BatchCosineSimilarity()
    
    self.controller = Controller(self.input_size, self.hidden_size, self.output_size, self.mem_size, self.mem_dim, self.read_heads)
  
  def forward(self, x, prev_state, memory, link_matrix):
    controls = self.controller(x)
    batch_size = x.shape[0]
    if prev_state == None:
      prev_state = dict()
      for key,value in controls.items():
        prev_state[key] = torch.zeros(value.shape)
      prev_state['read_weighting'] = torch.ones(batch_size, self.read_heads, self.mem_size) / (self.mem_size)
      prev_state['usage_vector'] = torch.zeros(batch_size, 1, self.mem_size)
      prev_state['write_weighting'] = torch.zeros(batch_size, 1, self.mem_size) + 1e-4

    prev_read_weighting = prev_state['read_weighting']
    prev_usage_vector = prev_state['usage_vector']
    prev_write_weighting = prev_state['write_weighting']
    var_phi = torch.prod(1 - controls['free_gates'] * prev_read_weighting, dim=1).unsqueeze(1)
    controls['var_phi'] = var_phi
    usage_vector = var_phi * (prev_usage_vector + prev_write_weighting - prev_usage_vector * prev_write_weighting)
    phi = var_phi.sort(dim=-1)[1]
    controls['phi'] = phi
    allocation_weighting = torch.zeros(batch_size, 1, self.mem_size)
    
    # allocation weighting is wrong
    for batch_num in range(batch_size):
      allocation_weighting[batch_num,:,phi[batch_num]] = (1 - usage_vector[batch_num,:, phi[batch_num]]) * torch.cumprod(usage_vector[batch_num,:, phi[batch_num]], dim=-1)
    controls['allocation_weighting'] = allocation_weighting
    write_key_strength = controls["write_key"] * controls["write_strength"]
    content_write_weighting = self.similarity(memory.memory, write_key_strength).transpose(1,2)
    controls['content_write_weighting'] = content_write_weighting
#     print(f"content_write_weighting: {content_write_weighting.shape}")
#     print(f"controls['write_gate']: {controls['write_gate'].shape}")
#     print(f"controls['allocation_gate']: {controls['allocation_gate'].shape}")
#     print(f"allocation_weighting: {allocation_weighting.shape}")
    write_weighting = controls["write_gate"] * (controls["allocation_gate"] * allocation_weighting + (1 - controls['allocation_gate']) * content_write_weighting)

    memory.write(write_weighting, controls['erase_vector'], controls['write_vector'])

    link_matrix.update_L(write_weighting)
    link_matrix.update_p(write_weighting)
    
    forward_read_weighting = torch.softmax(torch.bmm(prev_read_weighting, link_matrix.L), -1)
    backward_read_weighting = torch.softmax(torch.bmm(prev_read_weighting, link_matrix.L.transpose(1,2)), -1)
    
    read_key_strenghts = controls['read_keys'] * controls['read_strengths']

    content_read_weighting = self.similarity(memory.memory, read_key_strenghts).transpose(1,2)

    total_read_weighting = torch.cat([backward_read_weighting.unsqueeze(2), content_read_weighting.unsqueeze(2), forward_read_weighting.unsqueeze(2)], dim=2)

    read_weighting = (controls['read_modes'].unsqueeze(-1) * total_read_weighting).sum(2)
#     print(f"controls['read_modes']: {controls['read_modes']}")
#     print(f"total_read_weighting: {total_read_weighting}")
#     print(f"read_weighting: {read_weighting}")

    read_vector = memory.read(read_weighting)
    controls["read_weighting"] = read_weighting
    controls["usage_vector"] = usage_vector
    controls["write_weighting"] = write_weighting
    
    
    total_state = torch.cat([controls['output_vector'].view(batch_size,1,-1), read_vector.view(batch_size,1,-1)], dim=-1)
    controls["total_state"] = total_state
    return memory, link_matrix, controls, read_vector.view(batch_size, 1, self.read_heads * self.mem_dim)
    
    
if __name__ == "__main__":
    input_size = 4
    hidden_size = 8
    output_size = 4
    batch_size = 1
    read_heads = 1
    mem_size = 11
    mem_dim = 6

    x = torch.zeros(batch_size, input_size)

    memory = Memory(batch_size, mem_size, mem_dim)
    link_matrix = TemporalLinkMatrix(batch_size, mem_size, mem_dim)

    dnc = DifferentiableNeuralComputer(input_size, hidden_size, output_size, mem_size, mem_dim, read_heads)
    # print(memory)
    memory, link_matrix, controls, read_vector = dnc(x, None, memory, link_matrix)
    print(read_vector.shape)
    # print(read_vector)
    memory, link_matrix, controls, read_vector = dnc(x, controls, memory, link_matrix)
    # print(read_vector)