import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    BertTokenizer,
    BertTokenizerFast,
    BertForMaskedLM,
    RobertaTokenizer,
    RobertaTokenizerFast,
    RobertaForMaskedLM,
)
from utils import hf_masked_encode, hf_reconstruction_prob_tok, fill_batch


def gen_neighborhood(args):

    # initialize seed
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    # load model and tokenizer

    # slow tokenizer is for non-unk decoding
    if args.is_roberta:
        r_model = RobertaForMaskedLM.from_pretrained(args.model)
        tokenizer = RobertaTokenizerFast.from_pretrained(
            args.tokenizer, max_len=512
        )
        old_style_tokenizer = RobertaTokenizer.from_pretrained(
            args.tokenizer, max_len=512
        )
        mask_length = max(
            len(tokenizer.vocab), r_model.lm_head.decoder.out_features
        )
        start_ignore = min(
            len(tokenizer.vocab), r_model.lm_head.decoder.out_features
        )
    else:
        tokenizer = BertTokenizerFast.from_pretrained(
            args.tokenizer,
            clean_text=True,
            tokenize_chinese_chars=True,
            strip_accents=False,
            do_lower_case=False,
        )
        old_style_tokenizer = BertTokenizer.from_pretrained(
            args.tokenizer,
            do_lower_case=False,
            strip_accents=False,
            tokenize_chinese_chars=True,
        )
        r_model = BertForMaskedLM.from_pretrained(args.model)

        assert (
            len(tokenizer.vocab)
            == r_model.cls.predictions.decoder.out_features
        )
        mask_length = len(tokenizer.vocab)
        start_ignore = mask_length

    r_model.eval()
    if torch.cuda.is_available():
        r_model.cuda()

    # remove unused vocab and special ids from sampling
    softmax_mask = np.full(mask_length, False)
    softmax_mask[tokenizer.all_special_ids] = True
    for k, v in tokenizer.vocab.items():
        if "[unused" in k:
            softmax_mask[v] = True
    for i in range(start_ignore, mask_length):
        # this is what happens if your vocab is smaller than it claims to be
        # we'll never use the rest of the ids anyways so we should mask them
        softmax_mask[i] = True
        if not args.is_roberta:
            import pdb

            pdb.set_trace()

    # load the inputs and labels
    lines = [
        tuple(s.strip().split("\t")) for s in open(args.in_file).readlines()
    ]
    num_lines = len(lines)
    # lines[i] is a list of [s], where s is each sentence in the ith column
    # of the file
    lines = [[[s] for s in s_list] for s_list in list(zip(*lines))]

    # load label file if it exists
    if args.label_file:
        labels = [s.strip() for s in open(args.label_file).readlines()]
        output_labels = True
    else:
        labels = [0] * num_lines
        output_labels = False

    # shard the input and labels
    if args.num_shards > 0:
        shard_start = (int(num_lines / args.num_shards) + 1) * args.shard
        shard_end = (int(num_lines / args.num_shards) + 1) * (args.shard + 1)
        lines = [s_list[shard_start:shard_end] for s_list in lines]
        labels = labels[shard_start:shard_end]

    # open output files
    if args.num_shards != 1:
        s_rec_file = open(args.output_prefix + "_" + str(args.shard), "w")
        unk_rec_file = open(
            args.output_prefix, "_unks_" + str(args.shard), "w"
        )
        if output_labels:
            l_rec_file = open(
                args.output_prefix + "_" + str(args.shard) + ".label", "w"
            )
    else:
        s_rec_file = open(args.output_prefix, "w")
        unk_rec_file = open(args.output_prefix + "_unks", "w")
        if output_labels:
            l_rec_file = open(args.output_prefix + ".label", "w")

    # sentences and labels to process
    sents = []
    l = []

    # number sentences generated
    num_gen = []

    # sentence index to noise from
    gen_index = []

    # number of tries generating a new sentence
    num_tries = []

    # next sentence index to draw from
    next_sent = 0

    # indices and words corresponding to each instance of [UNK] / <unk> for the
    # sentences in sents
    unks = []

    sents, l, next_sent, num_gen, num_tries, gen_index, unks = fill_batch(
        args,
        tokenizer,
        sents,
        l,
        lines,
        labels,
        next_sent,
        num_gen,
        num_tries,
        gen_index,
        unks,
        old_style_tokenizer,
    )

    total_unks_in_base_corpus = 0

    # main augmentation loop
    while sents != []:

        # remove any sentences that are done generating and dump to file
        for i in range(len(num_gen))[::-1]:
            if num_gen[i] == args.num_samples or num_tries[i] > args.max_tries:

                # get sent info
                gen_sents = sents.pop(i)
                num_gen.pop(i)
                gen_index.pop(i)
                label = l.pop(i)
                unk = unks.pop(i)

                total_unks_in_base_corpus += len(unk[0])

                # write generated sentences
                for sg in gen_sents[1:]:
                    # the [1:-1] here refers to some weirdness that repr() does
                    # on strings -- namely, adding quotes at the start and end
                    de_unked = [
                        de_unk(repr(val)[1:-1], unk[i], tokenizer)
                        for i, val in enumerate(sg)
                    ]
                    orig = [repr(val)[1:-1] for val in sg]

                    s_rec_file.write("\t".join(de_unked) + "\n")
                    unk_rec_file.write("\t".join(orig) + "\n")
                    if output_labels:
                        l_rec_file.write(label + "\n")

        # fill batch
        sents, l, next_sent, num_gen, num_tries, gen_index, unks = fill_batch(
            args,
            tokenizer,
            sents,
            l,
            lines,
            labels,
            next_sent,
            num_gen,
            num_tries,
            gen_index,
            unks,
            old_style_tokenizer,
        )

        # break if done dumping
        if len(sents) == 0:
            print(f"Total unks in base corpus: {total_unks_in_base_corpus}")
            break

        # build batch
        toks = []
        masks = []

        for i in range(len(gen_index)):
            s = sents[i][gen_index[i]]
            tok, mask = hf_masked_encode(
                tokenizer,
                *s,
                noise_prob=args.noise_prob,
                random_token_prob=args.random_token_prob,
                leave_unmasked_prob=args.leave_unmasked_prob,
            )
            toks.append(tok)
            masks.append(mask)

        # pad up to max len input
        max_len = max([len(tok) for tok in toks])
        pad_tok = tokenizer.pad_token_id

        toks = [
            F.pad(tok, (0, max_len - len(tok)), "constant", pad_tok)
            for tok in toks
        ]
        masks = [
            F.pad(mask, (0, max_len - len(mask)), "constant", pad_tok)
            for mask in masks
        ]
        toks = torch.stack(toks)
        masks = torch.stack(masks)

        # load to GPU if available
        if torch.cuda.is_available():
            toks = toks.cuda()
            masks = masks.cuda()

        # predict reconstruction
        rec, rec_masks = hf_reconstruction_prob_tok(
            toks,
            masks,
            tokenizer,
            r_model,
            softmax_mask,
            reconstruct=True,
            topk=args.topk,
        )

        # decode reconstructions and append to lists
        for i in range(len(rec)):
            rec_work = rec[i].cpu().tolist()
            s_rec = [
                s.strip()
                for s in tokenizer.decode(
                    [val for val in rec_work if val != tokenizer.pad_token_id][
                        1:-1
                    ]
                ).split(tokenizer.sep_token)
            ]
            s_rec = tuple(s_rec)

            # check if identical reconstruction or empty
            if s_rec not in sents[i] and "" not in s_rec:
                sents[i].append(s_rec)
                num_gen[i] += 1
                num_tries[i] = 0
                gen_index[i] = 0

            # otherwise try next sentence
            else:
                num_tries[i] += 1
                gen_index[i] += 1
                if gen_index[i] == len(sents[i]):
                    gen_index[i] = 0

        # clean up tensors
        del toks
        del masks


def de_unk(sentence, unk_data, tokenizer):
    current_unks = []
    ids = tokenizer.encode(sentence, add_special_tokens=False)
    unk_token_id = tokenizer.convert_tokens_to_ids(tokenizer.unk_token)
    for unk_idx, unk_val in unk_data.items():
        if ids[unk_idx] == unk_token_id:
            current_unks.append((unk_idx, unk_val))
    current_unks.sort(key=lambda tup: tup[0])
    for _, unk_val in current_unks:
        sentence = sentence.replace(tokenizer.unk_token, unk_val, 1)
    return sentence


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s",
        "--shard",
        type=int,
        default=0,
        help="Shard of input to process. Output filename "
        "will have _${shard} appended.",
    )

    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Total number of shards to shard input file with.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed to use for reconstruction and noising.",
    )

    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="bert-base-uncased",
        help="Name of HuggingFace BERT model to use for reconstruction,"
        " or filepath to local model directory.",
    )

    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="Name of HuggingFace tokenizer to use for vocabulary"
        " or filepath to local tokenizer. If None, uses the same"
        " as model.",
    )

    parser.add_argument(
        "-i",
        "--in-file",
        type=str,
        help="Path of input text file for augmentation."
        " Inputs should be separated by newlines with tabs indicating"
        " BERT <SEP> tokens.",
    )

    parser.add_argument(
        "-l",
        "--label-file",
        type=str,
        default=None,
        help="Path of input label file for augmentation if using "
        " label preservation.",
    )

    parser.add_argument(
        "-o",
        "--output-prefix",
        type=str,
        help="Prefix path for output files, including augmentations and"
        " preserved labels.",
    )

    parser.add_argument(
        "-p",
        "--noise-prob",
        type=float,
        default=0.15,
        help="Probability for selecting a token for noising."
        " Selected tokens are then masked, randomly replaced,"
        " or left the same.",
    )

    parser.add_argument(
        "-r",
        "--random-token-prob",
        type=float,
        default=0.1,
        help="Probability of a selected token being replaced"
        " randomly from the vocabulary.",
    )

    parser.add_argument(
        "-u",
        "--leave-unmasked-prob",
        type=float,
        default=0.1,
        help="Probability of a selected token being left"
        " unmasked and unchanged.",
    )

    parser.add_argument(
        "-b",
        "--batch",
        type=int,
        default=8,
        help="Batch size of inputs to reconstruction model.",
    )

    parser.add_argument(
        "-n",
        "--num-samples",
        type=int,
        default=4,
        help="Number of augmented samples to generate for each"
        " input example.",
    )

    parser.add_argument(
        "-t",
        "--max-tries",
        type=int,
        default=10,
        help="Number of tries to generate a unique sample"
        " before giving up.",
    )

    parser.add_argument(
        "--min-len",
        type=int,
        default=4,
        help="Minimum length input for augmentation.",
    )

    parser.add_argument(
        "--max-len",
        type=int,
        default=512,
        help="Maximum length input for augmentation.",
    )

    parser.add_argument(
        "--topk",
        "-k",
        type=int,
        default=-1,
        help="Top k to use for sampling reconstructed tokens from"
        " the BERT model. -1 indicates unrestricted sampling.",
    )

    parser.add_argument("--is-roberta", action="store_true")

    args = parser.parse_args()

    if args.shard >= args.num_shards:
        raise Exception(
            "Shard number {} is too large for the number"
            " of shards {}".format(args.shard, args.num_shards)
        )

    if not args.tokenizer:
        args.tokenizer = args.model

    gen_neighborhood(args)
