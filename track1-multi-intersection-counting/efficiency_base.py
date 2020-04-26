#!/usr/bin/python3
"""
Evaluate submissions for the Multi-Camera Vehicle Counting by Class track of the AI City Challenge.
"""
import os
import sys
import json
import time
import subprocess
import numpy as np


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def run_test(ofile="aic2020-base.json"):
    if os.path.exists(ofile):
        print("Reusing efficiency results from previous execution.")
        print()
        with open(ofile, 'r') as fh:
            data = json.load(fh)
        return data

    print(f"""
This script will execute a number of Python benchmark tests, which may take 30 minutes 
or longer. Execution details will be printed on the screen. Once all tests are complete, 
the results will be analyzed and your system's base factor will be displayed. The results 
from the benchmark tests will be saved in a file called {ofile} in the current directory.

Testing will start in 5 seconds.

""")
    time.sleep(5)

    run = ['pyperformance', 'run', '-r',
           '-b', 'chaos,crypto_pyaes,deltablue,fannkuch,float,telco,pidigits,scimark,pyflate,unpack_sequence,raytrace,mdp,nbody,regex_dna',
           '-o', ofile]
    print(" ".join(run))
    process = subprocess.Popen(run,
                               stdout=subprocess.PIPE,
                               universal_newlines=True)

    while True:
        output = process.stdout.readline()
        print(output.strip())
        return_code = process.poll()
        if return_code is not None:
            for output in process.stdout.readlines():
                print(output.strip())
            break

    if return_code != 0 or not os.path.exists(ofile):
        eprint("An error was encountered and the performance evaluation did not complete successfully.")
        eprint("Try running the following command outside of this script and correct any issues you may encounter.")
        eprint(" ".join(run))
        return None

    with open(ofile, 'r') as fh:
        data = json.load(fh)
    return data


def aggregate_results(data):
    d = {}
    for t in data['benchmarks']:
        k = t['metadata']['name']
        v = np.mean([r['values'] for r in t['runs'] if 'values' in r])
        d[k] = v
    return d


def compare_results(d):
    b = {'chaos': 0.12057809454466527,
         'crypto_pyaes': 0.12106639864311243,
         'deltablue': 0.007840610432079604,
         'fannkuch': 0.4709514254704118,
         'float': 0.12307816236279905,
         'mdp': 2.8187601251954524,
         'nbody': 0.13694479484111072,
         'pidigits': 0.16544497444216782,
         'pyflate': 0.6741043091674025,
         'raytrace': 0.49927737620115903,
         'regex_dna': 0.15620897532595943,
         'scimark_fft': 0.3424340309536395,
         'scimark_lu': 0.19092780489784975,
         'scimark_monte_carlo': 0.11332551787684982,
         'scimark_sor': 0.1970164019187602,
         'scimark_sparse_mat_mult': 0.004494478550380639,
         'telco': 0.006094860816422927,
         'unpack_sequence': 4.915164166045353e-08}
    return np.mean([b[k]/d[k] for k in b.keys()])


if __name__ == '__main__':
    print("AIC2020 Efficiency Base Factor: %f" % compare_results(aggregate_results(run_test())))
