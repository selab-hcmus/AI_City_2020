import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw
import tensorflow as tf

def get_threshold(metric, box):
    if metric == "l2":
        return (box[2] - box[0]) * (box[3] - box[1]) * 400
    elif metric == "psnr":
        return 26.5
    else:
        raise ValueError("Invalid metric!")

def decide(metric, diff, threshold):
    if metric == "psnr":
        return diff < threshold
    else:
        return diff > threshold

def get_image(frames_path, frame):
    try:
        return Image.open(frames_path / ("{:05d}.jpg".format(frame))).convert("RGBA")
    except FileNotFoundError:
        pass
    try:
        return Image.open(frames_path / ("{:05d}.bmp".format(frame))).convert("RGBA")
    except FileNotFoundError:
        raise

def get_frame(frames_path, frame, box):
    im = get_image(frames_path, frame)
    return np.array(im, dtype=np.int32)[box[1]:box[3], box[0]:box[2]]

def draw_test(frames_path, video_id, start_frame, box):
    im = get_image(frames_path, start_frame)
    draw = ImageDraw.Draw(im)
    draw.rectangle(box, outline="red")
    im.save(output_path / ("{:05d}_{:05d}.png".format(video_id, start_frame)))

def diff_sum(im1, im2):
    return np.sum(np.abs(im1 - im2))

def diff_l2(im1, im2):
    return np.sum((im1 - im2) ** 2)

def calculate_diff(metric, im1, im2):
    if metric == "sum":
        return diff_sum(im1, im2)
    elif metric == "l2":
        return diff_l2(im1, im2)
    elif metric == "psnr":
        return tf.image.psnr(im1, im2, max_val=255)
    else:
        raise ValueError("Invalid metric!")

frames_all_path = Path(r"frames")
input_path = Path.cwd() / "trace_input.txt"
output_path = Path.cwd() / "trace_output"
output_file_path = output_path / "output.txt"
patience = 30 * 3
metric = "psnr"

if __name__ == "__main__":
    with open(input_path, 'r') as f:
        videos_to_run = [line.strip() for line in f.readlines() if line[0] != '#']

    results = []

    for line in videos_to_run:
        line = line.split('\t')
        video_id, x1, y1, x2, y2, start_sec, gt_sec = [int(x) for x in line]
        box = [x1, y1, x2, y2]
        gen_start = max(gt_sec - 10, 0)
        offset = gen_start * 30
        start_frame = start_sec * 30
        frames_path = frames_all_path / str(video_id)
        threshold = get_threshold(metric, box)
        draw_test(frames_path, video_id, start_frame - offset, box)
        #continue
        print("Debug: video {}, max diff = {}, gt = {}".format(video_id, threshold, gt_sec * 30))

        base = get_frame(frames_path, start_frame - offset, box)
        misses = 0
        for i in range(start_frame - 1, -1, -1):
            frame = get_frame(frames_path, i - offset, box)
            diff = calculate_diff(metric, base, frame)
            if decide(metric, diff, threshold):
                misses += 1
                if misses > patience:
                    print("start = {}, got = {}/{:.2f}, gt = {}".format(start_frame, i + misses, (i + misses) / 30, gt_sec * 30))
                    results.append([video_id, i + misses])
                    break
            else:
                misses = 0
            if i % 30 == 0:
                print("Debug: currently at {}, diff = {}".format(i, diff))
                pass
        else:
            results.append([video_id, 0])

    with open(output_file_path, 'w') as f:
        for result in results:
            print(*result, sep=',')
    