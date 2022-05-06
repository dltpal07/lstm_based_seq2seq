import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import time, datetime
import argparse
import numpy as np
from pathlib import Path

import utils, dataloader, lstm


parser = argparse.ArgumentParser(description='NMT - Seq2Seq with Attention')
""" recommend to use default settings """
# environmental settings
parser.add_argument('--gpu-id', type=int, default=0)
parser.add_argument('--seed-num', type=int, default=0)
parser.add_argument('--save', action='store_true', default=0)
parser.add_argument('--res-dir', default='../result', type=str)
parser.add_argument('--res-tag', default='seq2seq', type=str)
# architecture
parser.add_argument('--num_layers', type=int, default=4)
parser.add_argument('--max-len', type=int, default=20)
parser.add_argument('--hidden-size', type=int, default=512)
parser.add_argument('--max-norm', type=float, default=5.0)
# hyper-parameters
parser.add_argument('--n_epochs', type=int, default=100)
parser.add_argument('--batch-size', type=int, default=128)
parser.add_argument('--lr', type=float, default=0.001)
# option
parser.add_argument('--autoregressive', action='store_true', default=False)
parser.add_argument('--teacher-forcing', action='store_true', default=False)
parser.add_argument('--attn', action='store_true', default=False)
# etc
parser.add_argument('--k', type=int, default=4, help='hyper-paramter for BLEU score')

args = parser.parse_args()


if not args.autoregressive:
	print(" *** Non-Autoregressive ***")

utils.set_random_seed(seed_num=args.seed_num)

use_cuda = utils.check_gpu_id(args.gpu_id)
device = torch.device('cuda:{}'.format(args.gpu_id) if use_cuda else 'cpu')

t_start = time.time()

vocab_src = utils.read_pkl('../data/de-en/nmt_simple.src.vocab.pkl')
vocab_tgt = utils.read_pkl('../data/de-en/nmt_simple.tgt.vocab.pkl')

# recommend to split trainset
tr_dataset = dataloader.NMTSimpleDataset(max_len=args.max_len,
										 src_filepath='../data/de-en/nmt_simple.src.train.txt',
										 tgt_filepath='../data/de-en/nmt_simple.tgt.train.txt',
										 vocab=(vocab_src, vocab_tgt))
val_dataset = dataloader.NMTSimpleDataset(max_len=args.max_len,
										  src_filepath='../data/de-en/nmt_simple.src.val.txt',
										  tgt_filepath='../data/de-en/nmt_simple.tgt.val.txt',
										  vocab=(tr_dataset.vocab_src, tr_dataset.vocab_tgt))
vocab_src = tr_dataset.vocab_src
vocab_tgt = tr_dataset.vocab_tgt
i2w_src = {v:k for k, v in vocab_src.items()}
i2w_tgt = {v:k for k, v in vocab_tgt.items()}

tr_dataloader = DataLoader(tr_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=2)
val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True, num_workers=2)

encoder = lstm.Encoder(len(vocab_src), args.hidden_size, num_layers=args.num_layers)
if not args.attn:
	decoder = lstm.Decoder(len(vocab_tgt), args.hidden_size, num_layers=args.num_layers)
else:
	decoder = lstm.AttnDecoder(len(vocab_tgt), args.hidden_size, num_layers=args.num_layers, max_len=args.max_len)

utils.init_weights(encoder, init_type='uniform')
utils.init_weights(decoder, init_type='uniform')
encoder = encoder.to(device)
decoder = decoder.to(device)

""" TO DO: (masking) convert this line for masking [PAD] token """
criterion = nn.NLLLoss()

optimizer_enc = optim.Adam(encoder.parameters(), lr=args.lr)
optimizer_dec = optim.Adam(decoder.parameters(), lr=args.lr)


def train(dataloader, epoch):
	encoder.train()
	decoder.train()
	tr_loss = 0.
	correct = 0

	cnt = 0
	total_score = 0.
	prev_time = time.time()
	for idx, (src, tgt) in enumerate(dataloader):
		src, tgt = src.to(device), tgt.to(device)

		optimizer_enc.zero_grad()
		optimizer_dec.zero_grad()

		"""
			TO DO: feed the input to Encoder

			Encoder
				input: 
					(h0, c0)  <- init state for encoder
					src
				output:
					enc_outputs
					(h, c)
		"""			

		"""
			TO DO: feed the context from Encoder to Decoder

			Decoder
				input: 
					(h, c)  	<- context from encoder
					dec_input
					enc_outputs <- only attention
				output:
					dec_outputs

			* teacher forcing, non-autoregressive might be implemented here
		"""			

		outputs = torch.stack(dec_outputs, dim=1).squeeze()

		outputs = outputs.reshape(args.batch_size * args.max_len, -1)
		tgt = tgt.reshape(-1)

		loss = criterion(outputs, tgt)
		tr_loss += loss.item()
		loss.backward()

		""" TO DO: (clipping) convert this line for clipping the 'gradient < args.max_norm' """
		torch.nn.utils.clip_grad_norm_(encoder.parameters())
		torch.nn.utils.clip_grad_norm_(decoder.parameters())

		optimizer_enc.step()
		optimizer_dec.step()

		# accuracy
		pred = outputs.argmax(dim=1, keepdim=True)
		pred_acc = pred[tgt != 0]
		tgt_acc = tgt[tgt != 0]
		correct += pred_acc.eq(tgt_acc.view_as(pred_acc)).sum().item()

		cnt += tgt_acc.shape[0]

		# BLEU score
		score = 0.
		with torch.no_grad():
			pred = pred.reshape(args.batch_size, args.max_len, -1).detach().cpu().tolist()
			tgt = tgt.reshape(args.batch_size, args.max_len).detach().cpu().tolist()
			for p, t in zip(pred, tgt):
				eos_idx = t.index(vocab_tgt['[PAD]']) if vocab_tgt['[PAD]'] in t else len(t)
				p_seq = [i2w_tgt[i[0]] for i in p][:eos_idx]
				t_seq = [i2w_tgt[i] for i in t][:eos_idx]
				k = args.k if len(t_seq) > args.k else len(t_seq)
				s = utils.bleu_score(p_seq, t_seq, k=k)
				score += s
				total_score += s

		score /= args.batch_size

		# verbose
		batches_done = (epoch - 1) * len(dataloader) + idx
		batches_left = args.n_epochs * len(dataloader) - batches_done
		time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
		prev_time = time.time()
		print("\r[epoch {:3d}/{:3d}] [batch {:4d}/{:4d}] loss: {:.6f} (eta: {})".format(
			epoch, args.n_epochs, idx+1, len(dataloader), loss, time_left), end=' ')

	tr_loss /= cnt
	tr_acc = correct / cnt
	tr_score = total_score / len(dataloader.dataset)
	
	return tr_loss, tr_acc, tr_score


def validate(dataloader, save=False):
	encoder.eval()
	decoder.eval()
	val_loss = 0.
	correct = 0

	cnt = 0
	total_score = 0.
	prev_time = time.time()
	with torch.no_grad():
		for src, tgt in dataloader:
			src, tgt = src.to(device), tgt.to(device)

			"""
				TO DO: feed the input to Encoder

				Encoder
					input: 
						(h0, c0)  <- init state for encoder
						src
					output:
						enc_outputs
						(h, c)
			"""			

			"""
				TO DO: feed the context from Encoder to Decoder

				Decoder
					input: 
						(h, c)  	<- context from encoder
						dec_input
						enc_outputs <- only attention
					output:
						dec_outputs

				* teacher forcing, non-autoregressive might be implemented here
			"""			

			outputs = torch.stack(dec_outputs, dim=1).squeeze()

			outputs = outputs.reshape(args.batch_size * args.max_len, -1)
			tgt = tgt.reshape(-1)

			loss = criterion(outputs, tgt)
			val_loss += loss.item()

			pred = outputs.argmax(dim=1, keepdim=True)
			pred_acc = pred[tgt != 0]
			tgt_acc = tgt[tgt != 0]
			correct += pred_acc.eq(tgt_acc.view_as(pred_acc)).sum().item()

			cnt += tgt_acc.shape[0]

			# BLEU score
			score = 0.
			with torch.no_grad():
				pred = pred.reshape(args.batch_size, args.max_len, -1).detach().cpu().tolist()
				tgt = tgt.reshape(args.batch_size, args.max_len).detach().cpu().tolist()
				for p, t in zip(pred, tgt):
					eos_idx = t.index(vocab_tgt['[PAD]']) if vocab_tgt['[PAD]'] in t else len(t)
					p_seq = [i2w_tgt[i[0]] for i in p][:eos_idx]
					t_seq = [i2w_tgt[i] for i in t][:eos_idx]
					k = args.k if len(t_seq) > args.k else len(t_seq)
					s = utils.bleu_score(p_seq, t_seq, k=k)
					score += s
					total_score += s

			score /= args.batch_size

	val_loss /= cnt
	val_acc = correct / cnt
	val_score = total_score / len(dataloader.dataset)
	
	return val_loss, val_acc, val_score


for epoch in range(1, args.n_epochs + 1):
	tr_loss, tr_acc, tr_score = train(tr_dataloader, epoch)
	val_loss, val_acc, val_score = validate(val_dataloader)
	# {format: (loss, acc, BLEU)}
	print("tr: ({:.4f}, {:5.2f}, {:5.2f}) | ".format(tr_loss, tr_acc * 100, tr_score * 100) + \
		  "val: ({:.4f}, {:5.2f}, {:5.2f})".format(val_loss, val_acc * 100, val_score * 100), end='')

print("\n[ Elapsed Time: {:.4f} ]".format(time.time() - t_start))

