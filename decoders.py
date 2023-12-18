import torch
import torch.nn as nn
from torchaudio.models.decoder import ctc_decoder
from constants import ENCODER


def get_beam_decoder(device_type):
    if device_type == "cuda":
        # cuda_ctc_decoder is not included in both the CPU and MPS versions of torchaudio if installed via pip
        from torchaudio.models.decoder import cuda_ctc_decoder

    if device_type == "cuda":
        return cuda_ctc_decoder(
            tokens=ENCODER.index_to_token,
            nbest=1,
            beam_size=200
        )
    return ctc_decoder(
        tokens=ENCODER.index_to_token,
        lexicon=None,
        nbest=1,
        beam_size=200,
        # it's common practise to represent the blank token as '-' and the silence token as '|' however, with respect
        # to the scope of the project, it is not desirable since it would require an otherwise unnecessary
        # instruction to represent the reference and hypothesis appropriately
        blank_token='',
        sil_token=' '
    )


# the tokens in each CTCHypothesis are differently represented (list vs. tensor) depending on which decoded is used
def decode_tokens(hypotheses, device_type):
    if device_type == "cuda":
        return [''.join(ENCODER.batch_decode(torch.tensor(z[0].tokens))) for z in hypotheses]
    # current version of the CPU version of the decoder adds a leading and trailing `blank` token,
    # i.e. a whitespace in this case
    return [''.join(ENCODER.batch_decode(z[0].tokens)).strip() for z in hypotheses]


class GreedyCTCDecoder(nn.Module):
    def __init__(self, labels):
        super().__init__()
        self.labels = labels

    @staticmethod
    def forward(emission):
        indices = torch.argmax(emission, dim=-1)  # (num_seq,)
        indices = torch.unique_consecutive(indices, dim=-1)
        return [''.join(ENCODER.batch_decode(z)) for z in indices]
