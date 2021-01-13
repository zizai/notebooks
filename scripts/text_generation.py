""" Conditional text generation with the auto-regressive models of the library (GPT/GPT-2/Transformer-XL/XLNet)
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import logging

import torch
import numpy as np

from neuroblast.agents.arwen import ArwenModel
from chaosbreaker.envs.text.wikipedia import WikipediaLibrary

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop


def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--padding_text", type=str, default="")
    parser.add_argument("--length", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    args = parser.parse_args()

    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()

    set_seed(args)

    data_dir = './data/wiki_test'
    model_path = None
    env = WikipediaLibrary(data_dir)
    wenjie = Wenjie(args.device, env, model_path=model_path)

    # add some past history
    with open('./scripts/wwi.txt') as f:
        lines = f.readlines()
        logits, past_codes = wenjie.encode(lines)

    past = ()
    for layer_codes in past_codes:
        layer_code = layer_codes[:, 0:1]
        print(layer_code.shape)
        past = past + (layer_code, )

    out = wenjie.sample_sequence(args.prompt, past, args.length)
    out = out[0].tolist()
    text = wenjie.tokenizer.decode(out, clean_up_tokenization_spaces=True)
    print(text)

    '''
    print(args)
    while True:
        raw_text = args.prompt if args.prompt else input("Model prompt >>> ")
        if args.model_type in ["transfo-xl", "xlnet"]:
            # Models with memory likes to have a long prompt for short inputs.
            raw_text = (args.padding_text if args.padding_text else PADDING_TEXT) + raw_text
        context_tokens = tokenizer.encode(raw_text)
        out = sample_sequence(
            model=model,
            context=context_tokens,
            length=args.length,
            past=past,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            device=args.device,
            is_xlnet=bool(args.model_type == "xlnet"),
        )
        out = out[0, len(context_tokens):].tolist()
        text = tokenizer.decode(out, clean_up_tokenization_spaces=True)
        print(text)
        if args.prompt:
            break
    return text
    '''


if __name__ == '__main__':
    main()
