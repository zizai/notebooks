import logging

import torch
from torch.utils.tensorboard import SummaryWriter


def setup_logger(level=logging.DEBUG):
    formatter = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(module)s - %(message)s')

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    _logger = logging.getLogger(__name__)
    _logger.setLevel(level)
    _logger.addHandler(handler)
    return _logger


if __name__ == '__main__':
    writer = SummaryWriter("/tmp/runs/test_logger")
    print(writer.get_logdir())

    video = torch.rand((4, 20, 3, 100, 100), dtype=torch.float)  # N,T,C,H,W
    video = torch.clamp(video, 0., 1.)

    x = range(100)
    for i in x:
        writer.add_scalar('y=2x', i * 2, i)

    writer.add_video('videos', video, fps=20)
    writer.close()
