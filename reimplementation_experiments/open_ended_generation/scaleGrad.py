import torch.nn.functional as F
import torch


def getNovelMask(target, vocab_size):
    b, l = target.size()
    zeros = torch.zeros(b, l, vocab_size).to(target.device)
    ones = torch.ones(b, l, vocab_size).to(target.device)
    target_index = target.unsqueeze(1).expand(b, l, l).transpose(-2, -1).triu().transpose(-2, -1)
    matrix = zeros.scatter_add_(2, target_index, ones)
    matrix[:, :, 0] = 0
    summ_true = torch.tensor(range(1, l + 1)).unsqueeze(0).float().to(target.device)
    summ_now = torch.sum(matrix, dim=-1)
    diff = summ_true - summ_now
    matrix[:, :, 0] = diff
    matrix = torch.cat((torch.zeros(b, 1, vocab_size).to(target.device), matrix[:, :-1, :]), 1)
    novel_mask = matrix < 1.

    return novel_mask


def sg_loss(batch, gamma, output):
    longer_sample = batch['input_ids'][0]
    inp = longer_sample
    model_output = output
    target = batch['labels'][0]
    logits = model_output[1]
    probs = F.softmax(logits, dim=-1)
    novel_mask = getNovelMask(target.unsqueeze(0), logits.size(-1))
    rep_mask = ~novel_mask
    new_probs = probs * novel_mask * gamma + probs * rep_mask + 1e-8
    new_probs = F.normalize(new_probs, p=1, dim=-1)
    lprobs = torch.log(new_probs)
    lprobs_flatten = torch.unsqueeze(lprobs[0, :, :], 0)
    assert lprobs_flatten.size(0) == 1, 'Nonflat sequence ERROR'
    loss = F.nll_loss(lprobs_flatten[0], target, reduction='sum')
    ntokens = inp.numel()

    return loss / ntokens