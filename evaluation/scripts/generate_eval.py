import click_models as cm
import data_utils
import json
import os
import sys
import numpy as np
data_dir = '../'
fi = sys.argv[1]
name = fi.split('/')[-1]
name = name[:name.find('.')]
print(fi, name)
model_desc = json.load(open(fi))
click_model = cm.loadModelFromJson(model_desc)
target = './tmp/'

def parse_data(data_dir, ti='train', tp = 'train', rank_cut=10, target='./'):
    train_session = ''
    train_size = ''
    train_svm = ''
    train_rank = ''
    print("Reading data in %s" % data_dir)
    train_set = data_utils.read_data(data_dir, ti, rank_cut)
    l = len(train_set.initial_list)
    fout1 = open(target + 'letor.%s' % tp, 'w')
    fout2 = open(target + 'letor.%s.query' % tp, 'w')
    fout3 = open(target + 'letor.%s.svm' % tp, 'w')
    fout4 = open(target + 'letor.%s.rank' % tp, 'w')
    rg = 1
    qid = 0
    for r in range(rg):
        ser = 0
        for i in range(l):
            if i % 1000 == 0:
                print(i, l, qid)
            train_size += str(len(train_set.initial_list[i])) + '\n'
            gold_label_list = [0 if train_set.initial_list[i][x] < 0 else 
                               train_set.gold_weights[i][x] for x in xrange(len(train_set.initial_list[i]))]
            click_list, _, _ = click_model.sampleClicksForOneList(list(gold_label_list))
            while sum(click_list) == 0:
                click_list, _, _ = click_model.sampleClicksForOneList(list(gold_label_list))
            for s in range(len(click_list)):
                feat_str = ''
                hit = 0
                for cnt, f in enumerate(train_set.features[ser + s]):
                    if f != 0:
                        hit += 1
                        feat_str += ' ' + str(cnt+1) + ':' + str(f)
                if hit == 0:
                    print(feat_str)
                    return
                train_session += str(click_list[s]) + feat_str + '\n'
                train_svm += str(click_list[s]) + ' qid:' + str(qid) + feat_str + '\n'
                train_rank += str(s) + '\n'
            qid += 1
            ser += len(click_list)
    
        fout1.write(train_session)
        fout2.write(train_size)
        fout3.write(train_svm)
        fout4.write(train_rank)
        train_size = ''
        train_session = ''
        train_svm = ''
        train_rank = ''
    fout1.close()
    fout2.close()
    fout3.close()
    fout4.close()
    return train_set


train_set = parse_data(data_dir=data_dir + 'generate_dataset/', ti='train', tp=name+'_train', rank_cut=100000, target=target)
test_set = parse_data(data_dir=data_dir + 'generate_dataset/', ti='test', tp=name+'_test', rank_cut=100000, target=target)




