from torch.utils.data.sampler import Sampler
from collections import defaultdict
import copy
import random
import numpy as np

class RandomIdentitySampler(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.
    Args:
    - data_source (list): list of (img_path, pid, camid).
    - num_instances (int): number of instances per identity in a batch.
    - batch_size (int): number of examples in a batch.
    """

    def __init__(self, data_source, batch_size, num_instances):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic = defaultdict(list) #dict with list value
        #{783: [0, 5, 116, 876, 1554, 2041],...,}
        for index, (pid, _, _, _) in enumerate(self.data_source):
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())

        # estimate number of examples in an epoch
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)

        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
            random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []

        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []

        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)

        return iter(final_idxs)

    def __len__(self):
        return self.length


class RandomIdentityImageSampler(Sampler):
    """PK sampler that uses distinct images within each PID group."""

    def __init__(self, data_source, batch_size, num_instances):
        if batch_size % num_instances != 0:
            raise ValueError("batch_size must be divisible by num_instances")

        self.num_instances = num_instances
        self.num_pids_per_batch = batch_size // num_instances
        self.index_dic = defaultdict(lambda: defaultdict(list))
        for index, (pid, image_id, _, _) in enumerate(data_source):
            self.index_dic[pid][image_id].append(index)

        self.pids = list(self.index_dic)
        for pid, image_indices in self.index_dic.items():
            if len(image_indices) < num_instances:
                raise ValueError(
                    f"PID {pid} has {len(image_indices)} distinct images, "
                    f"fewer than num_instances={num_instances}"
                )

        self.groups_per_pid = {
            pid: sum(len(indices) for indices in image_indices.values()) // num_instances
            for pid, image_indices in self.index_dic.items()
        }
        self.length = sum(self.groups_per_pid.values()) * num_instances

    def __iter__(self):
        groups_by_pid = defaultdict(list)
        for pid, image_indices in self.index_dic.items():
            image_ids = list(image_indices)
            for _ in range(self.groups_per_pid[pid]):
                selected_images = random.sample(image_ids, self.num_instances)
                groups_by_pid[pid].append([
                    random.choice(image_indices[image_id])
                    for image_id in selected_images
                ])

        available_pids = list(self.pids)
        final_indices = []
        while len(available_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(available_pids, self.num_pids_per_batch)
            for pid in selected_pids:
                final_indices.extend(groups_by_pid[pid].pop())
                if not groups_by_pid[pid]:
                    available_pids.remove(pid)
        return iter(final_indices)

    def __len__(self):
        return self.length


class RandomPositiveMixedSampler(Sampler):
    """Build batches with distinct-image PID pairs and singleton PIDs."""

    def __init__(self, data_source, batch_size, positive_pairs_per_batch):
        if positive_pairs_per_batch < 1 or 2 * positive_pairs_per_batch >= batch_size:
            raise ValueError(
                "positive_pairs_per_batch must be positive and leave at least one singleton"
            )

        self.batch_size = batch_size
        self.positive_pairs_per_batch = positive_pairs_per_batch
        self.num_singleton_pids = batch_size - 2 * positive_pairs_per_batch
        self.pid_image_indices = defaultdict(lambda: defaultdict(list))
        for index, (pid, image_id, _, _) in enumerate(data_source):
            self.pid_image_indices[pid][image_id].append(index)
        self.pids = list(self.pid_image_indices)
        self.pair_pids = [
            pid for pid in self.pids
            if len(self.pid_image_indices[pid]) >= 2
        ]

        required_pids = positive_pairs_per_batch + self.num_singleton_pids
        if len(self.pair_pids) < positive_pairs_per_batch or len(self.pids) < required_pids:
            raise ValueError("dataset has too few identities to form a mixed batch")

        self.num_batches = len(data_source) // batch_size

    def __iter__(self):
        remaining = {
            pid: {image_id: list(indices) for image_id, indices
                  in image_indices.items()}
            for pid, image_indices in self.pid_image_indices.items()
        }
        for image_indices in remaining.values():
            for indices in image_indices.values():
                random.shuffle(indices)

        active_pids = set(self.pids)
        final_indices = []
        for _ in range(self.num_batches):
            if len(active_pids) - self.positive_pairs_per_batch < self.num_singleton_pids:
                break
            active_pid_list = sorted(active_pids)
            pair_pool = [
                pid for pid in active_pid_list
                if sum(bool(indices) for indices in remaining[pid].values()) >= 2
            ]
            if len(pair_pool) < self.positive_pairs_per_batch:
                break

            pair_pids = random.sample(pair_pool, self.positive_pairs_per_batch)
            used_pids = set(pair_pids)
            batch_indices = []
            for pid in pair_pids:
                available_images = [
                    image_id for image_id, indices in remaining[pid].items()
                    if indices
                ]
                selected_images = random.sample(available_images, 2)
                for image_id in selected_images:
                    batch_indices.append(remaining[pid][image_id].pop())

            singleton_pool = [pid for pid in active_pid_list if pid not in used_pids]
            for pid in random.sample(singleton_pool, self.num_singleton_pids):
                available_images = [
                    image_id for image_id, indices in remaining[pid].items()
                    if indices
                ]
                image_id = random.choice(available_images)
                batch_indices.append(remaining[pid][image_id].pop())
                if not any(remaining[pid].values()):
                    active_pids.remove(pid)
            for pid in pair_pids:
                if not any(remaining[pid].values()):
                    active_pids.remove(pid)

            random.shuffle(batch_indices)
            final_indices.extend(batch_indices)
        return iter(final_indices)

    def __len__(self):
        return self.num_batches * self.batch_size

