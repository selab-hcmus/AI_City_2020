import argparse
import os

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ann_dir', type=str, default=None,
                        help='path to annotation files directory.')
    parser.add_argument('--result_train_path', type=str, default=None,
                        help='path to result files (txt file).')
    parser.add_argument('--result_val_path', type=str, default=None,
                        help='path to result files (txt file).')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='train ratio in dataset (train_val_ratio).')
    args = parser.parse_args()

    ann_dir = args.ann_dir
    result_val_path = args.result_val_path
    result_train_path = args.result_train_path
    train_ratio = args.train_ratio
    assert train_ratio < 1, f"Error: train_ratio must be < 1!"

    lsd = os.listdir(ann_dir)
    lsd_train = lsd[:int(len(lsd)*train_ratio)]
    lsd_val = lsd[int(len(lsd)*train_ratio):]

    with open(result_train_path, 'w') as f:
        for ele in lsd_train:
            f.write(ele+'\n')

    with open(result_val_path, 'w') as f:
        for ele in lsd_val:
            f.write(ele+'\n')

    print('result file save in {} & {}'.format(result_train_path, result_val_path))
