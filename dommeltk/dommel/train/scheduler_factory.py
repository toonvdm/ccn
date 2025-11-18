import torch.optim  # noqa: F401


def scheduler_factory(optimizer, type, **kwargs):
    """
    Factory method to construct an scheduler
    :param optimizer: The optimizer which will be scheduled
    :param type: Name of the scheduler as defined in torch.optim.lr_scheduler
    :return: an optimizer object
    """
    scheduler = eval(f"torch.optim.lr_scheduler.{type}(optimizer, **kwargs)")
    return scheduler
