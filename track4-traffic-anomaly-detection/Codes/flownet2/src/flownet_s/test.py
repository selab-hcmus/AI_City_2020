import argparse
import os

from .flownet_s import FlowNetS
from ..net import Mode

FLAGS = None


def main():
    # Create a new network
    net = FlowNetS(mode=Mode.TEST)

    # Train on the data
    net.test(
        checkpoint='./checkpoints/FlowNetS/flownet-S.ckpt-0',
        input_a_path=FLAGS.input_a,
        input_b_path=FLAGS.input_b,
        out_path=FLAGS.out,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_a',
        type=str,
        required=True,
        help='Path to first image'
    )
    parser.add_argument(
        '--input_b',
        type=str,
        required=True,
        help='Path to second image'
    )
    parser.add_argument(
        '--out',
        type=str,
        required=True,
        help='Path to output flow result'
    )
    FLAGS = parser.parse_args()

    # Verify arguments are valid
    if not os.path.exists(FLAGS.input_a):
        raise ValueError('image_a path must exist')
    if not os.path.exists(FLAGS.input_b):
        raise ValueError('image_b path must exist')
    if not os.path.isdir(FLAGS.out):
        raise ValueError('out directory must exist')
    main()
