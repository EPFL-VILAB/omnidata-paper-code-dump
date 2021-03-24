import os
import csv
import pickle
import json
import torch
import shutil


def pickle_save(obj, save_path):
    with open(save_path, 'wb') as f:
        pickle.dump(obj, f)

def pickle_load(load_path):
    with open(load_path, 'rb') as f:
        return pickle.load(f)

def write_tuples_list_to_csv(tuples, save_path):
    with open(save_path, 'w') as f:
        writer = csv.writer(f)
        for tup in tuples:
            writer.writerow(tup)

def read_tuples_list_from_csv(save_path):
    tuples = []
    with open(save_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            tuples.append(tuple(row))
    return tuples

def save_checkpoint(state, is_best, save_dir='./', model='model', ext='.pth.tar'):
    check_point_path = os.path.join(save_dir, model + '_checkpoint' + ext)
    torch.save(state, check_point_path)
    if is_best:
        shutil.copyfile(check_point_path, os.path.join(save_dir, model + '_best' + ext))

def load_checkpoint(save_dir='./', model='model', ext='.pth.tar'):
    return torch.load(os.path.join(save_dir, model + '_checkpoint' + ext))
