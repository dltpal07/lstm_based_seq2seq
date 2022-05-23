import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

class CustomLSTM(nn.Module):
	def __init__(self, input_size, hidden_size, num_layers, max_len = None):
		super(CustomLSTM, self).__init__()
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.num_layers = num_layers

		self.lstm_layer1 = nn.LSTMCell(input_size, hidden_size)
		self.lstm_layer2 = nn.LSTMCell(input_size, hidden_size)
		self.lstm_layer3 = nn.LSTMCell(input_size, hidden_size)
		self.lstm_layer4 = nn.LSTMCell(input_size, hidden_size)

		self.max_len = max_len

	def forward(self, x, state):
		output = []
		hx, cx = state

		# pack sequence
		if isinstance(x, torch.nn.utils.rnn.PackedSequence):
			input, batch_sizes, sorted_indices, unsorted_indices = x
			cur_idx = 0
			for i, batch_size in enumerate(batch_sizes):
				hx[0, :batch_size], cx[0, :batch_size] = self.lstm_layer1(input[cur_idx:cur_idx + batch_size].clone(), (hx[0, :batch_size].clone(), cx[0, :batch_size].clone()))
				hx[1, :batch_size], cx[1, :batch_size] = self.lstm_layer2(hx[0, :batch_size].clone(), (hx[1, :batch_size].clone(), cx[1, :batch_size].clone()))
				hx[2, :batch_size], cx[2, :batch_size] = self.lstm_layer3(hx[1, :batch_size].clone(), (hx[2, :batch_size].clone(), cx[2, :batch_size].clone()))
				hx[3, :batch_size], cx[3, :batch_size] = self.lstm_layer4(hx[2, :batch_size].clone(), (hx[3, :batch_size].clone(), cx[3, :batch_size].clone()))
				cur_idx += batch_size
				output.append(hx[3].clone())
			output = torch.stack(output, dim=1)
			hx = self.permute_hidden(hx, unsorted_indices)
			cx = self.permute_hidden(cx, unsorted_indices)
		else:
			x = x.squeeze(1)
			hx[0], cx[0] = self.lstm_layer1(x.clone(), (hx[0].clone(), cx[0].clone()))
			hx[1], cx[1] = self.lstm_layer2(hx[0].clone(), (hx[1].clone(), cx[1].clone()))
			hx[2], cx[2] = self.lstm_layer3(hx[1].clone(), (hx[2].clone(), cx[2].clone()))
			hx[3], cx[3] = self.lstm_layer4(hx[2].clone(), (hx[3].clone(), cx[3].clone()))
			output.append(hx[3].clone())
			output = torch.stack(output, dim=1)
		return output, (hx, cx)

	def permute_hidden(self, hx, permutation):
		if permutation is None:
			return hx
		return self.apply_permutation(hx, permutation)

	def apply_permutation(self, tensor, permutation, dim=1):
		return tensor.index_select(dim, permutation)

class Encoder(nn.Module):
	def __init__(self, vocab_size, hidden_size, num_layers=4, **kwargs):
		super(Encoder, self).__init__()
		self.hidden_size = hidden_size
		self.num_layers = num_layers

		self.embedding = nn.Embedding(vocab_size, hidden_size)
		self.rnn = CustomLSTM(hidden_size, hidden_size, num_layers)
	
	def forward(self, x, state):
		inputs_length = torch.sum(torch.where(x>0, True, False), dim=1)
		x = self.embedding(x)
		packed = pack(x, inputs_length.tolist(), batch_first=True, enforce_sorted=False)
		output, state = self.rnn(packed, state)
		output = pack(output, inputs_length.tolist(), batch_first=True, enforce_sorted=False)
		output, outputs_length = unpack(output, batch_first=True, total_length=x.shape[1])
		return output, state
	

class Decoder(nn.Module):
	def __init__(self, vocab_size, hidden_size, num_layers=4, max_len = 20, **kwargs):
		super(Decoder, self).__init__()
		self.hidden_size = hidden_size
		
		self.embedding = nn.Embedding(vocab_size, hidden_size)
		self.rnn = CustomLSTM(hidden_size, hidden_size, num_layers, max_len)

		self.classifier = nn.Sequential(
			nn.Linear(hidden_size, vocab_size),
			nn.LogSoftmax(dim=-1)
		)
	
	def forward(self, x, state):
		x = self.embedding(x)
		output, state = self.rnn(x, state)
		output = self.classifier(output)
		return output, state


class AttnDecoder(nn.Module):
	def __init__(self, vocab_size, hidden_size, num_layers=4, max_len=20, *kwargs):
		super(AttnDecoder, self).__init__()
		self.hidden_size = hidden_size
		self.embedding = nn.Embedding(vocab_size, hidden_size)
		self.rnn = CustomLSTM(hidden_size, hidden_size, num_layers)
		self.softmax = nn.Softmax(dim=-1)
		self.classifier = nn.Sequential(
			nn.Linear(hidden_size * 2, vocab_size),
			nn.LogSoftmax(dim=-1)
		)

	def forward(self, x, state, src_output, cur_i):
		x = self.embedding(x)
		output, state = self.rnn(x, state)
		output_T = output.reshape(-1, self.hidden_size, 1)
		attention = torch.bmm(src_output, output_T)
		attention = attention.squeeze(-1)
		attention = self.softmax(attention)
		attention = attention.unsqueeze(-1)
		attention = torch.bmm(attention, output)
		attention = torch.sum(attention, dim=1, keepdim=True)
		feature_vector = torch.cat((attention, src_output[:,cur_i].unsqueeze(1)), dim=-1)
		feature_vector = self.classifier(feature_vector)
		return feature_vector, state



