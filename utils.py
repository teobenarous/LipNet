import os
import re
import cv2
import torch
import torchvision.transforms as transforms
import ffmpeg
from constants import ENCODER, CHECKPOINT_DIR


def load_video(path):
    cap = cv2.VideoCapture(path)
    frames = list()
    transform_to_tensor = transforms.ToTensor()
    transform_to_gray = transforms.Grayscale(num_output_channels=1)

    for _ in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
        ret, frame = cap.read()
        frame = transform_to_gray(transform_to_tensor(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))[:, 190:236, 80:220]
        frames.append(frame)
    cap.release()
    frames = torch.stack(frames, dim=1)
    return (frames - torch.mean(frames)) / torch.std(frames)


def load_alignments(path):
    tokens = list()
    with open(path, 'r') as file_obj:
        for line in file_obj:
            line = line.split()
            if line[2] != "sil":
                tokens.append(' ')
                tokens.extend(line[2])
    return ENCODER.batch_encode(tokens[1:])


def load_data(path):
    filename = path.split('/')[-1].split('.')[0]
    video_path = os.path.join("data", "s1", f"{filename}.mpg")
    alignment_path = os.path.join("data", "alignments", "s1", f"{filename}.align")
    frames = load_video(video_path)
    alignments = load_alignments(alignment_path)
    return frames, alignments


def get_largest_epoch():
    # match the basename pattern
    file_pattern = re.compile(r"checkpoint_epoch_(\d+)\.pth")
    largest_epoch = -1
    if os.path.exists(CHECKPOINT_DIR) and os.path.isdir(CHECKPOINT_DIR):
        for file in os.listdir(CHECKPOINT_DIR):
            match = file_pattern.match(file)
            if match:
                epoch_num = int(match.group(1))
                largest_epoch = max(largest_epoch, epoch_num)
    return largest_epoch


def gif_to_mp4():
    (
        ffmpeg
        .input("animation.gif")
        .output("animation.mp4",
                **{"movflags": 'faststart'},
                pix_fmt="yuv420p",
                vf="scale=trunc(iw/2)*2:trunc(ih/2)*2",
                y=None,
                loglevel="error")
        .global_args("-hide_banner")
        .run()
    )
