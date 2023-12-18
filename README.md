# LipNet
This model builds upon the great work originally introduced by Yannis M. Assael, Brendan Shillingford, Shimon Whiteson and Nando de Freitas, as detailed in their research paper (https://arxiv.org/abs/1611.01599). 
LipNet is a model designed for lip-reading, integrating Convolutional Neural Networks (CNNs) with a Recurrent Neural Network (RNN) to effectively interpret lip movements. 
The model utilizes 3D convolution layers to extract spatial features from video frames, emphasizing the shape and movement of the lips. 
Following this, a bidirectional LSTM (Long Short-Term Memory) layer is employed to analyze the temporal aspects, tracking the progression and rhythm of lip movements over time. 
This combination of CNNs for capturing static lip positions and LSTM for understanding the sequence of movements enables LipNet to accurately decode spoken words from visual cues alone.

The model was trained on a subset of the [GRID corpus dataset](https://spandh.dcs.shef.ac.uk//gridcorpus/).

The best performing weights achieved the following accuracies:

| Epoch | CER  | WER  | BLEU |
|-------|------|------|------|
| 200   | 0.04 | 0.07 | 0.89 |

<img src="/accuracies.png" width="550">

### Setup
#### Prerequisites
* Python 3.9+
* A `ffmeg` executable in `$PATH`

#### Install the dependencies
`pip install -r requirements.txt`

### Usage
To visualize how the model is performing using Streamlit, run the following command:

`streamlit run app.py`

By default, this will load the weights saved at the 200th epoch.

https://github.com/teobenarous/LipNet/assets/96968228/a1098b13-9951-4e31-9abc-f26a5dfd6863

### Configuration
The script `app.py` allows for the selection of model weights based on epoch numbers through a command-line argument.
There are four epoch options available: 50, 100, 150, and 200.
The script is designed to accept an epoch number in two formats: either with `--epoch` or its short form `-e`. 

To provide this argument, include an additional `--` to the `streamlit` command to separate Streamlit arguments from the script arguments.

For example here's how to run the application with the weights saved at the 50th epoch:
* Using the full argument name:

`streamlit run app.py -- --epoch=50`

* Using the short argument name:

`streamlit run app.py -- -e=50`
