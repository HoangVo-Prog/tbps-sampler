# tbps

## Setup

```bash
!git clone https://github.com/HoangVo-Prog/tbps-sampler.git
%cd tbps-sampler
!pip install uv
!uv pip install -r requirements.txt

# download benchmarks
!uv pip install gdown
!gdown 1HQrJ9tlIoU6JmgNxdPZiYfbElee9HNrO
```

> If you are on Kaggle and want to use the benchmarks, please consider add and upvote [this](https://www.kaggle.com/datasets/hoanggv/tbps-benchmark)


## Useful new CLI

* `--lr-total-epochs INT`: Sets the epoch horizon used by the learning-rate scheduler independently from the actual training duration. Useful for short runs that should follow the same LR trajectory as a longer training schedule.

* `--eval-after-epoch INT`: Delays validation/evaluation until the specified epoch. Useful for skipping unnecessary early evaluations while preserving the original evaluation frequency afterward.

## Make file


