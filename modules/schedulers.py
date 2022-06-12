from transformers import get_scheduler


def create_scheduler(scheduler_cfg, optimizer):
    sched_name = scheduler_cfg.pop('sched')
    return get_scheduler(name=sched_name, optimizer=optimizer, **scheduler_cfg)