import argparse
import subprocess
import time
from pathlib import Path

def parse_cmd_args():
    parser = argparse.ArgumentParser(description='extract frames from video')
    parser.add_argument('-p', '--path', type=str, 
                        default=r"F:\Datasets\AIC20_track4\test-data",
                        help='input folder path')
    parser.add_argument('-f', '--file', type=str,
                        default="trace_input.txt",
                        help="name of file containing videos names to extract from")
    parser.add_argument('-n', '--name', type=str,
                        help='name of video to extract')
    parser.add_argument('--ss', type=str,
                        help='start time to extract')
    parser.add_argument('-t', '--type', type=str,
                        default="jpg",
                        help='file type of output image')
    parser.add_argument('-o', '--output', type=str, 
                        default=r"frames",
                        help='output folder path')

    args = parser.parse_args()
    return args

def second_to_time(sec):
    return time.strftime('%H:%M:%S', time.gmtime(sec))

if __name__ == "__main__":
    args = parse_cmd_args()
    if args.file is None:
        name = args.name.split('.')[0]
        (Path(args.output) / name).mkdir(exist_ok=True)
        subprocess.run(["ffmpeg", "-ss", args.ss, "-i", str(Path(args.path) / args.name), str(Path(args.output) / name / (r"%05d." + args.type))])
    else:
        with open(args.file, 'r') as f:
            inputs = [line for line in f.readlines() if line[0] is not '#']
        for line in inputs:
            line = line.strip().split('\t')
            video_name = line[0]
            start_time = second_to_time(max(int(line[6]) - 10, 0))
            end_time = second_to_time(int(line[5]) + 2)
            input_path = Path(args.path) / (video_name + ".mp4")
            output_path = Path(args.output) / video_name / (r"%05d." + args.type)  
            
            try:
                (Path(args.output) / video_name).mkdir()
                subprocess.run(["ffmpeg", "-ss", start_time, "-i", str(input_path), "-to", end_time, "-copyts", str(output_path)])
            except FileExistsError:
                print("Skipping video {} because it already existed.".format(video_name))