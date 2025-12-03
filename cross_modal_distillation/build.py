from hydra.errors import InstantiationException
from hydra.utils import instantiate
from hydra import initialize, compose
from torch.utils.data import DataLoader
from cross_modal_distillation.data.collate import get_collate_fn


def build_module(*args, **kwargs):
    """
    config must be passed as a keyword argument

    config hierarchy must be:
        _target_
        config (DictConf with fields)
    """
    config = kwargs.pop("config")
    try:
        return instantiate(config, *args, **kwargs)
    except InstantiationException as e:
        if 'TypeError("__init__()' in repr(
            e
        ):  # cases where class do not expect config argument

            # here, we need to do this since config MUST be the first argument
            c = config.get("config")
            other_kwargs = {k: v for k, v in config.items() if k != "config"}
            return instantiate(c, *args, **other_kwargs, **kwargs)
        raise e


def build_config(config_name="ms_lfp_10s", config_dir="data/configs"):
    with initialize(version_base=None, config_path="data/configs"):
        config = compose(config_name=config_name)
    return config


def build_dataloader(dataset, collate_fn_name=None, **kwargs):
    collate_fn = get_collate_fn(collate_fn_name=collate_fn_name)
    loader = DataLoader(dataset, collate_fn=collate_fn, **kwargs)
    return loader
