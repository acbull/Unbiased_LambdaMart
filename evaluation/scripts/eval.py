import data_utils
import sys
import numpy as np
import os

if __name__ == "__main__":
    data_dir = '../generate_dataset/'
    test_set = data_utils.read_data(data_dir, 'test', 100000)
    res = open(sys.argv[1]).readlines()
    res = [float(fi[:-1]) for fi in res]
    ser = 0
    fout = open('./trec_out.txt', 'w')
    for i, j in zip(test_set.initial_list, test_set.qids):
        l = np.array(res[ser: ser + len(i)])
        r = np.argsort(-l)
        for s, si in zip(r, range(len(r))):
            fout.write(('%s Q0 %s %d %f SVM\n') %
                       (j, test_set.dids[i[s]], si+1, l[s]))
        ser += len(r)
    fout.close()

    os.system('./trec_eval -c -m ndcg_cut.1,3,5,10 ' + data_dir +
              'test/test.qrels ./trec_out.txt > res.out')
    print(''.join(open('./res.out').readlines()))
    os.system('./trec_eval -c ' + data_dir +
              'test/test.qrels ./trec_out.txt > res.out')
    print(''.join(open('./res.out').readlines()))
