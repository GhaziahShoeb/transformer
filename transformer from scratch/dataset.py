import torch
import torch.nn as nn
from torch.utils.data import Dataset
from typing import Any
class BilingualDataset(Dataset):
    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len, num_workers: int = 0) -> None:
        super().__init__()
    
        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.seq_len = seq_len
        self.num_workers = num_workers

        self.sos_token_src = torch.tensor([tokenizer_src.token_to_id("[SOS]")], dtype=torch.int64)
        self.eos_token_src = torch.tensor([tokenizer_src.token_to_id("[EOS]")], dtype=torch.int64)
        self.pad_token_src = torch.tensor([tokenizer_src.token_to_id("[PAD]")], dtype=torch.int64)

        self.sos_token_tgt = torch.tensor([tokenizer_tgt.token_to_id("[SOS]")], dtype=torch.int64)
        self.eos_token_tgt = torch.tensor([tokenizer_tgt.token_to_id("[EOS]")], dtype=torch.int64)
        self.pad_token_tgt = torch.tensor([tokenizer_tgt.token_to_id("[PAD]")], dtype=torch.int64)

        # Backward compatibility aliases
        self.sos_token = self.sos_token_src
        self.eos_token = self.eos_token_src
        self.pad_token = self.pad_token_src

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index: Any) -> Any:
        src_target_pair = self.ds[index]
        src_text = src_target_pair['translation'][self.src_lang]
        tgt_text = src_target_pair['translation'][self.tgt_lang]

        # Convert text to tokens
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1

        # Prevent training crash if sequence exceeds max sequence length
        if enc_num_padding_tokens < 0:
            enc_input_tokens = enc_input_tokens[:self.seq_len - 2]
            enc_num_padding_tokens = 0
        if dec_num_padding_tokens < 0:
            dec_input_tokens = dec_input_tokens[:self.seq_len - 1]
            dec_num_padding_tokens = 0

        enc_padding = torch.full((enc_num_padding_tokens,), self.pad_token_src.item(), dtype=torch.int64)
        dec_padding = torch.full((dec_num_padding_tokens,), self.pad_token_tgt.item(), dtype=torch.int64)

        # Add SOS and EOS to the source text
        encoder_input = torch.cat(
            [
                self.sos_token_src,
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                self.eos_token_src,
                enc_padding
            ]
        )
        # Add SOS to the decoder input
        decoder_input = torch.cat(
            [
                self.sos_token_tgt,
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                dec_padding
            ]
        )
        # Add EOS to the label (expected output from decoder)
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.eos_token_tgt,
                dec_padding
            ]
        )
        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return {
            "encoder_input": encoder_input,  # (seq_len)
            "decoder_input": decoder_input,  # (seq_len)
            "encoder_mask": (encoder_input != self.pad_token_src).unsqueeze(0).unsqueeze(0).int(),  # (1, 1, seq_len)
            "decoder_mask": (decoder_input != self.pad_token_tgt).unsqueeze(0).unsqueeze(0).int() & causal_mask(decoder_input.size(0)),  # (1, seq_len, seq_len)
            "label": label,
            "src_text": src_text,
            "tgt_text": tgt_text
        }
def causal_mask(size):
    mask=torch.triu(torch.ones(1,size,size),diagonal=1).type(torch.int)
    return mask==0