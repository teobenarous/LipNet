import os
import argparse
import torch
import imageio
import ffmpeg
import streamlit as st
from utils import load_data, gif_to_mp4, get_largest_epoch
from lip_net import LipNN, load_checkpoint
from constants import ENCODER
from decoders import get_beam_decoder, decode_tokens


def get_video_list():
    return [os.path.basename(x) for x in os.listdir(os.path.join("data", "s1")) if x.endswith(".mpg")]


def process_video(path):
    frames, alignments = load_data(path)
    model.eval()
    with torch.no_grad():
        logits = model(frames.unsqueeze(dim=0).to(device)).log_softmax(dim=2)
        decoded = beam_search_decoder(
            logits,
            torch.full(size=(frames.shape[0],), fill_value=frames.shape[2], dtype=torch.int32).to(device)
        )
        sentence = decode_tokens(decoded, device.type)[0]
    return frames, decoded[0][0].tokens, sentence


def play_video(path):
    (
        ffmpeg
        .input(path)
        .output("./raw_video.mp4", vcodec='libx264')
        .run()
    )
    video = open("./raw_video.mp4", "rb")
    video_bytes = video.read()
    st.video(video_bytes)
    os.remove("./raw_video.mp4")


def infer(path):
    frames, tokens, decoded_sentence = process_video(path)

    st.info("This what the model processes when making a prediction")
    imageio.mimsave("./animation.gif", (frames.squeeze().numpy() * 255), fps=10)
    gif_to_mp4()
    os.remove("./animation.gif")

    video = open("./animation.mp4", "rb")
    video_bytes = video.read()
    st.video(video_bytes)
    os.remove("./animation.mp4")

    st.info("This is the decoded sequence of tokens via beam search given a probability distribution at each time step")
    st.markdown(tokens.numpy())

    st.info("This it the decoded sequence of characters")
    st.text(decoded_sentence)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load a specific epoch for the model.")
    parser.add_argument(
        '-e', "--epoch", type=int, nargs='?', default=get_largest_epoch(), help="The epoch number to load."
    )
    epoch = parser.parse_args().epoch
    if not os.path.exists(f"./checkpoints/checkpoint_epoch_{epoch}.pth"):
        raise AssertionError(f"No weights found for epoch {epoch}.")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = LipNN(len(ENCODER.index_to_token)).to(device)
    beam_search_decoder = get_beam_decoder(device.type)
    load_checkpoint(model, epoch_checkpoint=epoch, device=device)
    st.set_page_config(layout="wide")
    st.title("LipNet")
    options = get_video_list()
    selected_video = st.selectbox("Choose a video", options)
    video_path = os.path.join("data", "s1", selected_video)
    col1, col2 = st.columns(2)
    if options:
        with col1:
            play_video(video_path)
        with col2:
            infer(video_path)
