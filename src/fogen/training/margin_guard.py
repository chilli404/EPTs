import torch
import torch.nn.functional as F


def forced_choice_margin(model, encode, items, device):
    encoded = []
    for item in items:
        prefix = encode(item["prefix"])
        correct = encode(item["correct"])
        distractor = encode(item["distractor"])
        if prefix and correct and distractor:
            encoded.append((prefix, correct, distractor))
    if not encoded:
        raise ValueError("No valid margin-guard items after tokenization")
    sequences = []
    for prefix, correct, distractor in encoded:
        sequences.extend((prefix + correct, prefix + distractor))
    max_length = max(len(sequence) for sequence in sequences)
    ids = torch.zeros(len(sequences), max_length, dtype=torch.long, device=device)
    for index, sequence in enumerate(sequences):
        ids[index, :len(sequence)] = torch.tensor(sequence, device=device)
    logprobs = F.log_softmax(model(ids).float(), dim=-1)
    margins = []
    for index, (prefix, correct, distractor) in enumerate(encoded):
        correct_terms = [logprobs[2 * index, len(prefix) + offset - 1, token]
                         for offset, token in enumerate(correct)]
        distractor_terms = [logprobs[2 * index + 1, len(prefix) + offset - 1, token]
                            for offset, token in enumerate(distractor)]
        correct_mean = torch.stack(correct_terms).mean()
        distractor_mean = torch.stack(distractor_terms).mean()
        margins.append(correct_mean - distractor_mean)
    return torch.stack(margins).mean()


def project_gradient_(parameters, margin_gradients, max_dot=0.0):
    pairs = [
        (parameter, margin_gradient)
        for parameter, margin_gradient in zip(parameters, margin_gradients)
        if parameter.grad is not None and margin_gradient is not None
    ]
    dot_before = sum(
        torch.sum(parameter.grad.float() * margin_gradient.float()).item()
        for parameter, margin_gradient in pairs
    )
    margin_norm_sq = sum(
        torch.sum(margin_gradient.float() ** 2).item()
        for _, margin_gradient in pairs
    )
    projected = dot_before > max_dot and margin_norm_sq > 0
    coefficient = (dot_before - max_dot) / margin_norm_sq if projected else 0.0
    if projected:
        with torch.no_grad():
            for parameter, margin_gradient in pairs:
                parameter.grad.sub_(coefficient * margin_gradient.to(parameter.grad.dtype))
    dot_after = sum(
        torch.sum(parameter.grad.float() * margin_gradient.float()).item()
        for parameter, margin_gradient in pairs
    )
    return {
        "projected": projected,
        "dot_before": dot_before,
        "dot_after": dot_after,
        "margin_norm_sq": margin_norm_sq,
        "coefficient": coefficient,
    }
