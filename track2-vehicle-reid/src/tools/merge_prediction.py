import os
import glob
import os.path as osp
import sys
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..')))
import argparse
import pandas as pd
import ipdb
def parse_args():
    """Parse input arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i_cis', '--input_cispd', type = str,
        help='Path to the input predictions'
    )
    
    parser.add_argument(
        '-i_real', '--input_realpd', type = str,
        help='Path to the input predictions'
    )

    parser.add_argument(
        '-o', '--out_dir', type = str,
        help='Path to output directory'
    )
    parser.add_argument(
        '-t', '--template', type = str,
        help='Path to template csv file'
    )
    args = parser.parse_args()
    return args
def main():
    args  = parse_args()
    all_files = glob.glob(osp.join(args.input_cispd,"*.csv"))
    all_files+= glob.glob(osp.join(args.input_realpd,"*.csv"))
    pred_dict={}
    for file in all_files:
        with open(file,"r") as fi:
            lines = fi.readlines()
            lines = [i.strip() for i in lines]
            for line in lines:
                info = line.split(',')
                pred_dict[info[0]]= info[1]
            fi.close()

    out_data = pd.read_csv(args.template)
    template_name = osp.basename(args.template).replace(
        "_Template",""
    )
    for i, msr_id in enumerate(out_data['measurement_id']):
        if (msr_id in pred_dict):
            out_data.loc[i,'prediction'] = pred_dict[msr_id]
    
    out_dir = osp.dirname(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    out_data.to_csv(osp.join(out_dir, template_name))

if __name__ == '__main__':
    sys.exit(main())