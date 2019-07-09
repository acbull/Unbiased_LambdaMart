# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import numpy as np
import json
import random
import os

class Raw_data:
	def __init__(self, data_path = None, file_prefix = None, rank_cut=100000):
		if data_path == None:
			self.embed_size = -1
			self.rank_list_size = -1
			self.features = []
			self.dids = []
			self.initial_list = []
			self.qids = []
			self.gold_list = []
			self.gold_weights = []
			return

		settings = json.load(open(data_path + 'settings.json'))
		self.embed_size = int(settings['embed_size'])
		self.rank_list_size = rank_cut
		print(str(self.embed_size) + '---' + str(self.rank_list_size))
		self.features = []
		self.dids = []
		feature_fin = open(data_path + file_prefix + '/' + file_prefix + '.feature')
		for line in feature_fin:
			arr = line.strip().split(' ')
			self.dids.append(arr[0])
			self.features.append([0.0 for _ in range(self.embed_size)])
			for x in arr[1:]:
				arr2 = x.split(':')
				self.features[-1][int(arr2[0])] = float(arr2[1])
		feature_fin.close()
		print(str(self.embed_size) + '---' + str(self.rank_list_size))
		self.initial_list = []
		self.qids = []
		init_list_fin = open(data_path + file_prefix + '/' + file_prefix + '.init_list')
		for line in init_list_fin:
			arr = line.strip().split(' ')
			self.qids.append(arr[0])
			self.initial_list.append([int(x) for x in arr[1:][:self.rank_list_size]])
		init_list_fin.close()
		print(str(self.embed_size) + '---' + str(self.rank_list_size))
		self.gold_list = []
		gold_list_fin = open(data_path + file_prefix + '/' + file_prefix + '.gold_list')
		for line in gold_list_fin:
			self.gold_list.append([int(x) for x in line.strip().split(' ')[1:][:self.rank_list_size]])
		gold_list_fin.close()
		print(str(self.embed_size) + '---' + str(self.rank_list_size))
		self.gold_weights = []
		gold_weight_fin = open(data_path + file_prefix + '/' + file_prefix + '.weights')
		for line in gold_weight_fin:
			self.gold_weights.append([float(x) for x in line.strip().split(' ')[1:][:self.rank_list_size]])
		gold_weight_fin.close()
		print(str(self.embed_size) + '---' + str(self.rank_list_size))
		#self.initial_scores = []
		#if os.path.isfile(data_path + file_prefix + '/' + file_prefix + '.intial_scores'):
		#with open(data_path + file_prefix + '/' + file_prefix + '.initial_scores') as fin:
		#	for line in fin:
		#		self.initial_scores.append([float(x) for x in line.strip().split(' ')[1:]])
		print(str(self.embed_size) + '---' + str(self.rank_list_size))
	def pad(self, rank_list_size, pad_tails = True):
		self.rank_list_size = rank_list_size
		self.features.append([0 for _ in range(self.embed_size)])  # vector for pad
		for i in range(len(self.initial_list)):
			if len(self.initial_list[i]) < self.rank_list_size:
				if pad_tails: # pad tails
					self.initial_list[i] += [-1] * (self.rank_list_size - len(self.initial_list[i]))
				else:	# pad heads
					self.initial_list[i] = [-1] * (self.rank_list_size - len(self.initial_list[i])) + self.initial_list[i]
				self.gold_list[i] += [-1] * (self.rank_list_size - len(self.gold_list[i]))
				self.gold_weights[i] += [0.0] * (self.rank_list_size - len(self.gold_weights[i]))
				#self.initial_scores[i] += [0.0] * (self.rank_list_size - len(self.initial_scores[i]))


def read_data(data_path, file_prefix, rank_cut = 100000):
	data = Raw_data(data_path, file_prefix, rank_cut)
	return data

def generate_ranklist(data, rerank_lists):
	if len(rerank_lists) != len(data.initial_list):
		raise ValueError("Rerank ranklists number must be equal to the initial list,"
						 " %d != %d." % (len(rerank_lists)), len(data.initial_list))
	qid_list_map = {}
	for i in range(len(data.qids)):
		if len(rerank_lists[i]) != len(data.initial_list[i]):
			raise ValueError("Rerank ranklists length must be equal to the gold list,"
							 " %d != %d." % (len(rerank_lists[i]), len(data.initial_list[i])))
		#remove duplicate and organize rerank list
		index_list = []
		index_set = set()
		for j in rerank_lists[i]:
			#idx = len(rerank_lists[i]) - 1 - j if reverse_input else j
			idx = j
			if idx not in index_set:
				index_set.add(idx)
				index_list.append(idx)
		for idx in range(len(rerank_lists[i])):
			if idx not in index_set:
				index_list.append(idx)
		#get new ranking list
		qid = data.qids[i]
		did_list = []
		new_list = [data.initial_list[i][idx] for idx in index_list]
		for ni in new_list:
			if ni >= 0:
				did_list.append(data.dids[ni])
		qid_list_map[qid] = did_list
	return qid_list_map

def generate_ranklist_by_scores(data, rerank_scores):
	if len(rerank_scores) != len(data.initial_list):
		raise ValueError("Rerank ranklists number must be equal to the initial list,"
						 " %d != %d." % (len(rerank_lists)), len(data.initial_list))
	qid_list_map = {}
	for i in range(len(data.qids)):
		scores = rerank_scores[i]
		rerank_list = sorted(range(len(scores)), key=lambda k: scores[k], reverse=True)
		if len(rerank_list) != len(data.initial_list[i]):
			raise ValueError("Rerank ranklists length must be equal to the gold list,"
							 " %d != %d." % (len(rerank_lists[i]), len(data.initial_list[i])))
		#remove duplicate and organize rerank list
		index_list = []
		index_set = set()
		for j in rerank_list:
			#idx = len(rerank_lists[i]) - 1 - j if reverse_input else j
			idx = j
			if idx not in index_set:
				index_set.add(idx)
				index_list.append(idx)
		for idx in range(len(rerank_list)):
			if idx not in index_set:
				index_list.append(idx)
		#get new ranking list
		qid = data.qids[i]
		did_list = []
		for idx in index_list:
			ni = data.initial_list[i][idx]
			ns = scores[idx]
			if ni >= 0:
				did_list.append((data.dids[ni], ns))
		qid_list_map[qid] = did_list
	return qid_list_map

def output_ranklist(data, rerank_scores, output_path, file_name = 'test'):
	qid_list_map = generate_ranklist_by_scores(data, rerank_scores)
	fout = open(output_path + file_name + '.ranklist','w')
	for qid in data.qids:
		for i in range(len(qid_list_map[qid])):
			fout.write(qid + ' Q0 ' + qid_list_map[qid][i][0] + ' ' + str(i+1)
							+ ' ' + str(qid_list_map[qid][i][1]) + ' RankLSTM\n')
	fout.close()
