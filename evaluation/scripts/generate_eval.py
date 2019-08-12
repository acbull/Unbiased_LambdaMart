import click_models
import data_utils
import json
import os
import sys


if __name__ == "__main__":
    data_dir = '../'

    click_model_file = sys.argv[1]
    click_model_fname = os.path.basename(click_model_file)
    click_model_fname = os.path.splitext(click_model_fname)[0]
    print(click_model_file, click_model_fname)

    click_model = click_models.loadModelFromJson(click_model_file)
    target = '../test_data/'

    train_set = data_utils.parse_data(click_model=click_model,
                           data_dir=data_dir+'generate_dataset/',
                           task='eval', ti='train',
                           tp=name+'_train',
                           rank_cut=100000,
                           target=target)

    test_set = data_utils.parse_data(click_model=click_model,
                          data_dir=data_dir+'generate_dataset/',
                          task='eval', ti='test',
                          tp=name+'_test',
                          rank_cut=100000,
                          target=target)
