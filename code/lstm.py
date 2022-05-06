import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack


class Encoder(nn.Module):
	def __init__(self, vocab_size, hidden_size, num_layers=4, **kwargs):
		super(Encoder, self).__init__()
		self.hidden_size = hidden_size
		self.num_layers = num_layers

		self.embedding = nn.Embedding(vocab_size, hidden_size)
		""" TO DO: Implement your LSTM """
		self.rnn = YOURLSTM(hidden_size, hidden_size, num_layers, batch_first=True)
	
	def forward(self, x, state):
		""" TO DO: feed the unpacked input x to Encoder """
	
		return output, state
	

class Decoder(nn.Module):
	def __init__(self, vocab_size, hidden_size, num_layers=4, **kwargs):
		super(Decoder, self).__init__()
		self.hidden_size = hidden_size
		
		self.embedding = nn.Embedding(vocab_size, hidden_size)
		""" TO DO: Implement your LSTM """
		self.rnn = YOURLSTM(hidden_size, hidden_size, num_layers, batch_first=True)

		self.classifier = nn.Sequential(
			nn.Linear(hidden_size, vocab_size),
			nn.LogSoftmax(dim=-1)
		)
	
	def forward(self, x, state):
		""" TO DO: feed the input x to Decoder """
	
		return output, state


class AttnDecoder(nn.Module):
	""" TO DO: Implement your Decoder with Attention """
	pass

