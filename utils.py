from io import UnsupportedOperation
import numpy as np
import torch


def hf_masked_encode(
    tokenizer,
    sentence: str,
    *addl_sentences,
    noise_prob=0.0,
    random_token_prob=0.0,
    leave_unmasked_prob=0.0,
):

    if random_token_prob > 0.0:
        weights = np.ones(len(tokenizer.vocab))
        weights[tokenizer.all_special_ids] = 0
        for k, v in tokenizer.vocab.items():
            if "[unused" in k:
                weights[v] = 0
        weights = weights / weights.sum()

    tokens = np.asarray(
        tokenizer.encode(sentence, *addl_sentences, add_special_tokens=True)
    )

    if noise_prob == 0.0:
        return tokens

    sz = len(tokens)
    mask = np.full(sz, False)
    num_mask = int(noise_prob * sz + np.random.rand())

    mask_choice_p = np.ones(sz)
    for i in range(sz):
        if tokens[i] in [
            tokenizer.sep_token_id,
            tokenizer.cls_token_id,
            tokenizer.pad_token_id,
        ]:
            mask_choice_p[i] = 0
    mask_choice_p = mask_choice_p / mask_choice_p.sum()

    mask[np.random.choice(sz, num_mask, replace=False, p=mask_choice_p)] = True

    # decide unmasking and random replacement
    rand_or_unmask_prob = random_token_prob + leave_unmasked_prob
    if rand_or_unmask_prob > 0.0:
        rand_or_unmask = mask & (np.random.rand(sz) < rand_or_unmask_prob)
        if random_token_prob == 0.0:
            unmask = rand_or_unmask
            rand_mask = None
        elif leave_unmasked_prob == 0.0:
            unmask = None
            rand_mask = rand_or_unmask
        else:
            unmask_prob = leave_unmasked_prob / rand_or_unmask_prob
            decision = np.random.rand(sz) < unmask_prob
            unmask = rand_or_unmask & decision
            rand_mask = rand_or_unmask & (~decision)
    else:
        unmask = rand_mask = None

    if unmask is not None:
        mask = mask ^ unmask

    tokens[mask] = tokenizer.mask_token_id
    if rand_mask is not None:
        num_rand = rand_mask.sum()
        if num_rand > 0:
            tokens[rand_mask] = np.random.choice(
                len(tokenizer.vocab), num_rand, p=weights,
            )

    mask_targets = np.full(len(mask), tokenizer.pad_token_id)
    mask_targets[mask] = tokens[mask == 1]

    return torch.tensor(tokens).long(), torch.tensor(mask_targets).long()


def hf_reconstruction_prob_tok(
    masked_tokens,
    target_tokens,
    tokenizer,
    model,
    softmax_mask,
    reconstruct=False,
    topk=1,
):
    single = False

    # expand batch size 1
    if masked_tokens.dim() == 1:
        single = True
        masked_tokens = masked_tokens.unsqueeze(0)
        target_tokens = target_tokens.unsqueeze(0)

    masked_fill = torch.ones_like(masked_tokens)

    masked_index = (target_tokens != tokenizer.pad_token_id).nonzero(
        as_tuple=True
    )
    masked_orig_index = target_tokens[masked_index]

    # edge case of no masked tokens
    if len(masked_orig_index) == 0:
        if reconstruct:
            return masked_tokens, masked_fill
        else:
            return 1.0

    masked_orig_enum = [list(range(len(masked_orig_index))), masked_orig_index]

    outputs = model(
        masked_tokens.long().to(device=next(model.parameters()).device),
        labels=target_tokens,
    )

    features = outputs[1]

    logits = features[masked_index].detach().clone()
    for l in logits:
        l[softmax_mask] = float("-inf")
    probs = logits.softmax(dim=-1)

    if reconstruct:

        # sample from topk
        if topk != -1:
            values, indices = probs.topk(k=topk, dim=-1)
            kprobs = values.softmax(dim=-1)
            if len(masked_index) > 1:
                samples = torch.cat(
                    [
                        idx[torch.multinomial(kprob, 1)]
                        for kprob, idx in zip(kprobs, indices)
                    ]
                )
            else:
                samples = indices[torch.multinomial(kprobs, 1)]

        # unrestricted sampling
        else:
            if len(masked_index) > 1:
                samples = torch.cat(
                    [torch.multinomial(prob, 1) for prob in probs]
                )
            else:
                samples = torch.multinomial(probs, 1)

        # set samples
        masked_tokens[masked_index] = samples
        masked_fill[masked_index] = samples

        if single:
            return masked_tokens[0], masked_fill[0]
        else:
            return masked_tokens, masked_fill

    return torch.sum(torch.log(probs[masked_orig_enum])).item()


def fill_batch(
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
):

    # load sentences into batch until full
    while len(sents) < args.batch:

        # search for the next valid sentence
        while True:
            if next_sent >= len(lines[0]):
                break

            next_sents = [s_list[next_sent][0] for s_list in lines]
            next_len = len(tokenizer.encode(*next_sents))

            # skip input if too short or long
            if next_len > args.min_len and next_len < args.max_len:
                break
            next_sent += 1

        # add it to our lists
        if next_sent < len(lines[0]):
            # list of the values at next_sent index for each column in file
            next_sent_lists = [s_list[next_sent] for s_list in lines]
            # list of tuples, where each tuple contains the information
            # for the different columns in the line, and subsequent tuples
            # after the 0th are augmented.  at this point, nothing should
            # be augmented, so there should only be one tuple, which is a
            # singleton if there's only one column
            sentence_data = list(zip(*next_sent_lists))
            if len(sentence_data) > 1 or len(sentence_data[0]) > 1:
                import pdb

                pdb.set_trace()
            sents.append(sentence_data)
            l.append(labels[next_sent])

            unks.append(
                [
                    tuple(
                        get_unk_toks_indices(
                            field, tokenizer, old_style_tokenizer
                        )
                        for field in sentence_data[0]
                    )
                ]
            )

            num_gen.append(0)
            num_tries.append(0)
            gen_index.append(0)
            next_sent += 1
        else:
            break

    return sents, l, next_sent, num_gen, num_tries, gen_index, unks


def get_unk_toks_indices(sentence, tokenizer, old_style_tokenizer):
    tokenizer_class = repr(old_style_tokenizer.__class__)
    if "models.bert" in tokenizer_class:
        recovery_method = recover_bert_unk_alignment
    elif "models.roberta" in tokenizer_class:
        recovery_method = recover_roberta_unk_alignment
    else:
        raise UnsupportedOperation(
            f"Unknown tokenizer class {tokenizer_class}"
        )

    # when we get to printing, the only guarantee we have is that the outputted
    # sentence has the same number of wordpieces as the inputted sentence, so
    # we need to use the wordpiece-based indices
    chunks = tokenizer.tokenize(sentence)
    unk_chunk_indices = [
        i for i, chunk in enumerate(chunks) if chunk == tokenizer.unk_token
    ]

    unknown_strings = recovery_method(old_style_tokenizer, sentence)

    # make sure no funky tokenization going on here
    if len(unk_chunk_indices) != len(unknown_strings):
        import pdb

        pdb.set_trace()

    mapping = {
        idx: string for idx, string in zip(unk_chunk_indices, unknown_strings)
    }
    return mapping


def recover_bert_unk_alignment(bert_tokenizer, text):
    # based on https://github.com/huggingface/transformers/issues/9714
    assert (
        not hasattr(bert_tokenizer, "basic_tokenizer")
        or not bert_tokenizer.basic_tokenizer.never_split
    )
    whitespace_tokens = (
        bert_tokenizer.basic_tokenizer.tokenize(
            text, never_split=bert_tokenizer.all_special_tokens
        )
        if bert_tokenizer.do_basic_tokenize
        else text.strip().split()
    )

    ids = [
        bert_tokenizer(piece, add_special_tokens=False)["input_ids"]
        for piece in whitespace_tokens
    ]
    # whitespace-based indices of unk tokens
    unk_indices = [
        i
        for i, encoded in enumerate(ids)
        if bert_tokenizer.unk_token_id in encoded
    ]
    # strings corresponding to them
    unknown_strings = [
        piece for i, piece in enumerate(whitespace_tokens) if i in unk_indices
    ]

    return unknown_strings


def recover_roberta_unk_alignment(roberta_tokenizer, text):
    tokenized = roberta_tokenizer.tokenize(text)

    if roberta_tokenizer.unk_token in tokenized:
        import pdb

        pdb.set_trace()
    return []

