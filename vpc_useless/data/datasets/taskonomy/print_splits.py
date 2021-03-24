import argparse
import csv
import os
import subprocess

parser = argparse.ArgumentParser(description='Print buildings in split, one line at a time.')
parser.add_argument('split', type=str, nargs='?', choices=['debug', 'tiny', 'medium', 'full', 'fullplus'],
                    help='which split to print')
parser.add_argument('--split-file', type=str, default='',
                    help='name of split file (if different from the standard naming). Overrides "split_dir" and "split".')

args = parser.parse_args()


if args.split == 'debug':
    print('allensville')
    exit(0)

if args.split_file:
    split_csv = args.split_file
else:
    if not args.split:
        raise ValueError("Must specify split if not using --split-file")
        parser.print_help()
    split_csv = os.path.join('train_val_test_' + args.split + '.csv')

if not os.path.isfile(split_csv):
    split_csv = os.path.join(os.path.dirname(__file__), 'splits', 'train_val_test_' + args.split + '.csv')
    if not os.path.isfile(split_csv):
        split_csv = os.path.join('splits_taskonomy', 'train_val_test_' + args.split + '.csv')
    if not os.path.isfile(split_csv):
        subprocess.call('wget -q https://github.com/StanfordVL/taskonomy/raw/master/data/assets/splits_taskonomy.zip && unzip -qo splits_taskonomy.zip -x "__MACOSX/*" && rm splits_taskonomy.zip', shell=True)
        split_csv = os.path.join('splits_taskonomy', 'train_val_test_' + args.split + '.csv')

with open(split_csv, 'r') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    models = [row[0] for row in readCSV][1:]
    for model in models:
        print(model)

