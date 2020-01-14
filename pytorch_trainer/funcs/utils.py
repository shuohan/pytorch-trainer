# -*- coding: utf-8 -*-

def count_trainable_paras(model):
    """Counts the trainable parameters of a model.

    Args:
        model (torch.nn.Module): The model to count.

    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
