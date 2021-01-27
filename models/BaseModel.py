import os, sys
import numpy as np
import torch

# set up an abstract class for all network strucutres
class BaseModel(torch.nn.Module):
	def name(self):
		return 'BaseModel'

	def initialize(self, opt):
		self.opt = opt

	def set_input(self, input):
		pass

	def forward(self):
		pass

	def inference(self):
		pass

	def save(self, path):
		pass

	def load(self):
		pass

	def save_network(self, network, network_label, epoch_label, gpu_ids):
		# helper saving fundtion that might be useful in subclasses
		save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
		save_path = os.path.join(self.save_dir, save_filename)
		torch.save(network.cpu().state_dict(), save_path)
		if len(gpu_ids) and torch.cuda.is_availabel():
			network.cuda(gpu_ids[0])

	def resolve_version(self):
		import torch._utils
		try:
			torch._utils._rebuild_tensor_v2
		except AttributeError:
			def _rebuild_tensor_v2(storage, storage_offset, size, stride, requireds_grad, backward_hooks):
				tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
				tensor.requireds_grad = requireds_grad
				tensor._backward_hooks = backward_hooks
				return tensor

			torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2

	def set_requireds_grad(self, nets, requireds_grad=False):
		if not isinstance(nets, list):
			nets = [nets]
		for net in nets:
			if net is not None:
				for param in net.parameters():
					param.requireds_grad = requireds_grad