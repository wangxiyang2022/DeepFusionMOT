# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com

# combine KITTI tracking results from two categories for submission
import copy
import os

from utils.file_operation.file import mkdir_if_inexistence


def combine_files(file_list, save_path):
    # collect all files
    data_all = list()
    for file_tmp in file_list:
        data, num_lines = load_txt_file(file_tmp)
        data_all += data

    # sort based on frame number
    data_all.sort(key=lambda x: int(x.split(' ')[0]))

    save_txt_file(data_all, save_path)


def save_txt_file(data_list, save_path, debug=True):
    '''
    save a list of string to a file
    '''
    save_path = safe_path(save_path)
    first_line = True
    with open(save_path, 'w') as file:
        for item in data_list:
            if first_line:
                file.write('%s' % item)
                first_line = False
            else:
                file.write('\n%s' % item)
    file.close()


def load_txt_file(file_path, debug=True):
    '''
    load data or string from text file
    '''
    file_path = safe_path(file_path)
    with open(file_path, 'r') as file: data = file.read().splitlines()
    num_lines = len(data)
    file.close()
    return data, num_lines


def safe_path(input_path, warning=True, debug=True):
    '''
    convert path to a valid OS format, e.g., empty string '' to '.', remove redundant '/' at the end from 'aa/' to 'aa'

    parameters:
    	input_path:		a string

    outputs:
    	safe_data:		a valid path in OS format
    '''
    safe_data = copy.copy(input_path)
    safe_data = os.path.normpath(safe_data)
    return safe_data


def combine_category_result(cfg):
    root_dir = 'results/KITTI'
    seq_list = ['%04d' % tmp for tmp in range(0, len(cfg.tracking_seqs))]
    cat_lists = cfg.cat_list

    # save path
    save_dir = cfg.eval_save_path
    mkdir_if_inexistence(save_dir)

    # merge
    for seq_tmp in seq_list:
        file_list_tmp = list()
        for cat_list in cat_lists:
            file_tmp = os.path.join(root_dir, cat_list, 'data', seq_tmp + '.txt')
            file_list_tmp.append(file_tmp)

        save_path_tmp = os.path.join(save_dir, seq_tmp + '.txt')
        combine_files(file_list_tmp, save_path_tmp)
