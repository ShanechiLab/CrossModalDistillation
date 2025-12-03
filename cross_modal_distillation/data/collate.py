import inspect
import sys
from collections import namedtuple
from functools import partial

import torch
import torch.nn.functional as F

BatchItem = namedtuple(
    "BatchItem",
    ["inputs", "subject_sessions", "position_ids", "segment_filenames"],
)


def collate_with_metadata_fn(batch):
    inputs, position_ids, subject_sessions = [], [], []
    input_dims, input_seq_lens = [], []
    segment_filenames = []

    for datapoint in batch:
        inputs.append(datapoint["input"])
        subject_sessions.append(datapoint["subject_session"])

        num_steps = datapoint["input"].shape[0]
        position_ids.append(torch.arange(0, num_steps, 1))

        input_dims.append(datapoint["input"].shape[-1])
        input_seq_lens.append(datapoint["input"].shape[0])

        segment_filenames.append(datapoint["segment_filename"])

    # means that all trials have the same number of dimensions, so we can form a tensor
    # otherwise, keep them in a list and tokenizer will handle the rest
    if (torch.tensor(input_dims) == input_dims[0]).all() and (
        torch.tensor(input_seq_lens) == input_seq_lens[0]
    ).all():
        inputs = torch.stack(inputs, dim=0)
        if len(position_ids) > 0:
            position_ids = torch.stack(position_ids, dim=0)

    if len(position_ids) == 0:
        position_ids = None

    batch = BatchItem(
        inputs=inputs,
        position_ids=position_ids,
        subject_sessions=subject_sessions,
        segment_filenames=segment_filenames,
    )
    return batch


def get_collate_fn(collate_fn_name, **partial_kwargs):

    current_module = sys.modules[__name__]
    funcs = {
        name: obj
        for name, obj in inspect.getmembers(current_module, inspect.isfunction)
    }
    collate_fn = funcs.get(collate_fn_name, None)
    if collate_fn:
        return partial(collate_fn, **partial_kwargs)
    else:
        return None
