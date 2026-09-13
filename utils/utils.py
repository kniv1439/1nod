import random
import os
import torch
import torch.nn as nn
import numpy as np
import heapq
from concurrent.futures import ThreadPoolExecutor

from utils.DataLoader import Data

# ---------------------------------------------------------------------------
# Optional Numba JIT acceleration for 2-hop neighbor sampling.
# If numba is not installed the code falls back to the Python/ThreadPool path.
# ---------------------------------------------------------------------------
try:
    from numba import njit as _njit

    @_njit(cache=True)
    def _sample_2hop_numba(n1_ids, n1_edges, n1_times,
                           flat_ids, flat_edges, flat_times, node_ptr, max_k):
        """
        Collect the most-recent max_k 2-hop neighbors for ONE center node.

        n1_ids / n1_edges / n1_times : int64 / float64 arrays (first-hop neighbors)
        flat_* / node_ptr            : CSR representation of the whole graph
        Returns (ids, edges, times) sorted by time ascending, dtype int64 / float64.
        """
        heap_t = np.empty(max_k, dtype=np.float64)
        heap_n = np.empty(max_k, dtype=np.int64)
        heap_e = np.empty(max_k, dtype=np.int64)
        size   = 0

        for i in range(len(n1_ids)):
            mid = n1_ids[i]
            mt  = n1_times[i]
            lo  = node_ptr[mid]
            hi  = node_ptr[mid + 1]
            cut = np.searchsorted(flat_times[lo:hi], mt)  # neighbors of mid before mt

            for j in range(lo, lo + cut):
                t = flat_times[j]
                n = flat_ids[j]
                e = flat_edges[j]

                if size < max_k:
                    # --- heap push (sift-up) ---
                    heap_t[size] = t
                    heap_n[size] = n
                    heap_e[size] = e
                    k = size
                    size += 1
                    while k > 0:
                        parent = (k - 1) // 2
                        if heap_t[parent] > heap_t[k]:
                            heap_t[parent], heap_t[k] = heap_t[k], heap_t[parent]
                            heap_n[parent], heap_n[k] = heap_n[k], heap_n[parent]
                            heap_e[parent], heap_e[k] = heap_e[k], heap_e[parent]
                            k = parent
                        else:
                            break
                elif t > heap_t[0]:
                    # --- heap replace-root (sift-down) ---
                    heap_t[0] = t
                    heap_n[0] = n
                    heap_e[0] = e
                    k = 0
                    while True:
                        left    = 2 * k + 1
                        right   = 2 * k + 2
                        smallest = k
                        if left  < size and heap_t[left]  < heap_t[smallest]:
                            smallest = left
                        if right < size and heap_t[right] < heap_t[smallest]:
                            smallest = right
                        if smallest != k:
                            heap_t[smallest], heap_t[k] = heap_t[k], heap_t[smallest]
                            heap_n[smallest], heap_n[k] = heap_n[k], heap_n[smallest]
                            heap_e[smallest], heap_e[k] = heap_e[k], heap_e[smallest]
                            k = smallest
                        else:
                            break

        if size == 0:
            return (np.empty(0, dtype=np.int64),
                    np.empty(0, dtype=np.int64),
                    np.empty(0, dtype=np.float64))

        order = np.argsort(heap_t[:size])
        return heap_n[:size][order], heap_e[:size][order], heap_t[:size][order]

    _NUMBA_AVAILABLE = True

    @_njit(cache=True)
    def _sample_1hop_batch_numba(
        node_ids,        # int64, shape (batch_size,)
        interact_times,  # float64, shape (batch_size,)
        flat_ids,        # CSR flat node ids
        flat_edges,      # CSR flat edge ids
        flat_times,      # CSR flat timestamps
        node_ptr,        # CSR row pointers, shape (num_nodes+1,)
        out_ptr,         # output row pointers, shape (batch_size+1,) — filled here
        out_ids_buf,     # pre-allocated output buffer for neighbor ids
        out_edges_buf,   # pre-allocated output buffer for edge ids
        out_times_buf,   # pre-allocated output buffer for timestamps
    ):
        """
        Batch 1-hop neighbor lookup using CSR. Fills out_* buffers in-place.
        Returns the total number of neighbors written across all nodes.
        """
        total = np.int64(0)
        out_ptr[0] = np.int64(0)
        for i in range(len(node_ids)):
            nid = node_ids[i]
            t   = interact_times[i]
            lo  = node_ptr[nid]
            hi  = node_ptr[nid + 1]
            cut = np.int64(np.searchsorted(flat_times[lo:hi], t))
            for j in range(lo, lo + cut):
                out_ids_buf[total]   = flat_ids[j]
                out_edges_buf[total] = flat_edges[j]
                out_times_buf[total] = flat_times[j]
                total += np.int64(1)
            out_ptr[i + 1] = total
        return total

except ImportError:
    _NUMBA_AVAILABLE = False
    _sample_2hop_numba = None


def set_random_seed(seed: int = 0):
    """
    set random seed
    :param seed: int, random seed
    :return:
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def convert_to_gpu(*data, device: str):
    """
    convert data from cpu to gpu, accelerate the running speed
    :param data: can be any type, including Tensor, Module, ...
    :param device: str
    """
    res = []
    for item in data:
        item = item.to(device)
        res.append(item)
    if len(res) > 1:
        res = tuple(res)
    else:
        res = res[0]
    return res


def get_parameter_sizes(model: nn.Module):
    """
    get parameter size of trainable parameters in model
    :param model: nn.Module
    :return:
    """
    return sum([p.numel() for p in model.parameters() if p.requires_grad])


def count_unique_parameters(model: nn.Module) -> int:
    """Count unique stored parameters (respects weight tying). Includes frozen parameters."""
    seen = set()
    total = 0
    for p in model.parameters():
        pid = id(p)
        if pid not in seen:
            seen.add(pid)
            total += p.numel()
    return total


def count_trainable_unique_parameters(model: nn.Module) -> int:
    """Count unique trainable parameters (respects weight tying)."""
    seen = set()
    total = 0
    for p in model.parameters():
        if not p.requires_grad:
            continue
        pid = id(p)
        if pid not in seen:
            seen.add(pid)
            total += p.numel()
    return total


def create_optimizer(model: nn.Module, optimizer_name: str, learning_rate: float, weight_decay: float = 0.0):
    """
    create optimizer
    :param model: nn.Module
    :param optimizer_name: str, optimizer name
    :param learning_rate: float, learning rate
    :param weight_decay: float, weight decay
    :return:
    """
    if optimizer_name == 'Adam':
        optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == 'SGD':
        optimizer = torch.optim.SGD(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == 'RMSprop':
        optimizer = torch.optim.RMSprop(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        raise ValueError(f"Wrong value for optimizer {optimizer_name}!")

    return optimizer


class NeighborSampler:

    def __init__(self, adj_list: list, sample_neighbor_strategy: str = 'uniform', time_scaling_factor: float = 0.0, seed: int = None):
        """
        Neighbor sampler.
        :param adj_list: list, list of list, where each element is a list of triple tuple (node_id, edge_id, timestamp)
        :param sample_neighbor_strategy: str, how to sample historical neighbors, 'uniform', 'recent', or 'time_interval_aware'
        :param time_scaling_factor: float, a hyper-parameter that controls the sampling preference with time interval,
        a large time_scaling_factor tends to sample more on recent links, this parameter works when sample_neighbor_strategy == 'time_interval_aware'
        :param seed: int, random seed
        """
        self.sample_neighbor_strategy = sample_neighbor_strategy
        self.seed = seed

        # list of each node's neighbor ids, edge ids and interaction times, which are sorted by interaction times
        self.nodes_neighbor_ids = []
        self.nodes_edge_ids = []
        self.nodes_neighbor_times = []

        if self.sample_neighbor_strategy == 'time_interval_aware':
            self.nodes_neighbor_sampled_probabilities = []
            self.time_scaling_factor = time_scaling_factor

        # the list at the first position in adj_list is empty, hence, sorted() will return an empty list for the first position
        # its corresponding value in self.nodes_neighbor_ids, self.nodes_edge_ids, self.nodes_neighbor_times will also be empty with length 0
        for node_idx, per_node_neighbors in enumerate(adj_list):
            # per_node_neighbors is a list of tuples (neighbor_id, edge_id, timestamp)
            # sort the list based on timestamps, sorted() function is stable
            # Note that sort the list based on edge id is also correct, as the original data file ensures the interactions are chronological
            sorted_per_node_neighbors = sorted(per_node_neighbors, key=lambda x: x[2])
            self.nodes_neighbor_ids.append(np.array([x[0] for x in sorted_per_node_neighbors]))
            self.nodes_edge_ids.append(np.array([x[1] for x in sorted_per_node_neighbors]))
            self.nodes_neighbor_times.append(np.array([x[2] for x in sorted_per_node_neighbors]))

            # additional for time interval aware sampling strategy (proposed in CAWN paper)
            if self.sample_neighbor_strategy == 'time_interval_aware':
                self.nodes_neighbor_sampled_probabilities.append(self.compute_sampled_probabilities(np.array([x[2] for x in sorted_per_node_neighbors])))

        if self.seed is not None:
            self.random_state = np.random.RandomState(self.seed)

        # Build CSR representation for Numba-accelerated 2-hop sampling.
        if _NUMBA_AVAILABLE:
            _n = len(self.nodes_neighbor_ids)
            _ptr = np.zeros(_n + 1, dtype=np.int64)
            for _i, _arr in enumerate(self.nodes_neighbor_ids):
                _ptr[_i + 1] = _ptr[_i] + len(_arr)
            _total = int(_ptr[-1])
            if _total > 0:
                self._csr_flat_ids   = np.concatenate(
                    [a.astype(np.int64)   for a in self.nodes_neighbor_ids])
                self._csr_flat_edges = np.concatenate(
                    [a.astype(np.int64)   for a in self.nodes_edge_ids])
                self._csr_flat_times = np.concatenate(
                    [a.astype(np.float64) for a in self.nodes_neighbor_times])
            else:
                self._csr_flat_ids   = np.empty(0, dtype=np.int64)
                self._csr_flat_edges = np.empty(0, dtype=np.int64)
                self._csr_flat_times = np.empty(0, dtype=np.float64)
            self._csr_ptr = _ptr

    def compute_sampled_probabilities(self, node_neighbor_times: np.ndarray):
        """
        compute the sampled probabilities of historical neighbors based on their interaction times
        :param node_neighbor_times: ndarray, shape (num_historical_neighbors, )
        :return:
        """
        if len(node_neighbor_times) == 0:
            return np.array([])
        # compute the time delta with regard to the last time in node_neighbor_times
        node_neighbor_times = node_neighbor_times - np.max(node_neighbor_times)
        # compute the normalized sampled probabilities of historical neighbors
        exp_node_neighbor_times = np.exp(self.time_scaling_factor * node_neighbor_times)
        sampled_probabilities = exp_node_neighbor_times / np.cumsum(exp_node_neighbor_times)
        # note that the first few values in exp_node_neighbor_times may be all zero, which make the corresponding values in sampled_probabilities
        # become nan (divided by zero), so we replace the nan by a very large negative number -1e10 to denote the sampled probabilities
        sampled_probabilities[np.isnan(sampled_probabilities)] = -1e10
        return sampled_probabilities

    def find_neighbors_before(self, node_id: int, interact_time: float, return_sampled_probabilities: bool = False):
        """
        extracts all the interactions happening before interact_time (less than interact_time) for node_id in the overall interaction graph
        the returned interactions are sorted by time.
        :param node_id: int, node id
        :param interact_time: float, interaction time
        :param return_sampled_probabilities: boolean, whether return the sampled probabilities of neighbors
        :return: neighbors, edge_ids, timestamps and sampled_probabilities (if return_sampled_probabilities is True) with shape (historical_nodes_num, )
        """
        # 确保 interact_time 是 Python 标量（处理 CUDA tensor 的情况）
        if isinstance(interact_time, (torch.Tensor, np.ndarray)):
            interact_time = float(interact_time.item() if hasattr(interact_time, 'item') else interact_time)
        
        # 确保 node_id 是 Python 整数
        if isinstance(node_id, (torch.Tensor, np.ndarray)):
            node_id = int(node_id.item() if hasattr(node_id, 'item') else node_id)
        
        # return index i, which satisfies list[i - 1] < v <= list[i]
        # return 0 for the first position in self.nodes_neighbor_times since the value at the first position is empty
        i = np.searchsorted(self.nodes_neighbor_times[node_id], interact_time)

        if return_sampled_probabilities:
            return self.nodes_neighbor_ids[node_id][:i], self.nodes_edge_ids[node_id][:i], self.nodes_neighbor_times[node_id][:i], \
                   self.nodes_neighbor_sampled_probabilities[node_id][:i]
        else:
            return self.nodes_neighbor_ids[node_id][:i], self.nodes_edge_ids[node_id][:i], self.nodes_neighbor_times[node_id][:i], None

    def get_historical_neighbors(self, node_ids: np.ndarray, node_interact_times: np.ndarray, num_neighbors: int = 20):
        """
        get historical neighbors of nodes in node_ids with interactions before the corresponding time in node_interact_times
        :param node_ids: ndarray, shape (batch_size, ) or (*, ), node ids
        :param node_interact_times: ndarray, shape (batch_size, ) or (*, ), node interaction times
        :param num_neighbors: int, number of neighbors to sample for each node
        :return:
        """
        assert num_neighbors > 0, 'Number of sampled neighbors for each node should be greater than 0!'
        # All interactions described in the following three matrices are sorted in each row by time
        # each entry in position (i,j) represents the id of the j-th dst node of src node node_ids[i] with an interaction before node_interact_times[i]
        # ndarray, shape (batch_size, num_neighbors)
        nodes_neighbor_ids = np.zeros((len(node_ids), num_neighbors)).astype(np.longlong)
        # each entry in position (i,j) represents the id of the edge with src node node_ids[i] and dst node nodes_neighbor_ids[i][j] with an interaction before node_interact_times[i]
        # ndarray, shape (batch_size, num_neighbors)
        nodes_edge_ids = np.zeros((len(node_ids), num_neighbors)).astype(np.longlong)
        # each entry in position (i,j) represents the interaction time between src node node_ids[i] and dst node nodes_neighbor_ids[i][j], before node_interact_times[i]
        # ndarray, shape (batch_size, num_neighbors)
        nodes_neighbor_times = np.zeros((len(node_ids), num_neighbors)).astype(np.float32)

        # extracts all neighbors ids, edge ids and interaction times of nodes in node_ids, which happened before the corresponding time in node_interact_times
        for idx, (node_id, node_interact_time) in enumerate(zip(node_ids, node_interact_times)):
            # find neighbors that interacted with node_id before time node_interact_time
            node_neighbor_ids, node_edge_ids, node_neighbor_times, node_neighbor_sampled_probabilities = \
                self.find_neighbors_before(node_id=node_id, interact_time=node_interact_time, return_sampled_probabilities=self.sample_neighbor_strategy == 'time_interval_aware')

            if len(node_neighbor_ids) > 0:
                if self.sample_neighbor_strategy in ['uniform', 'time_interval_aware']:
                    # when self.sample_neighbor_strategy == 'uniform', we shuffle the data before sampling with node_neighbor_sampled_probabilities as None
                    # when self.sample_neighbor_strategy == 'time_interval_aware', we sample neighbors based on node_neighbor_sampled_probabilities
                    # for time_interval_aware sampling strategy, we additionally use softmax to make the sum of sampled probabilities be 1
                    if node_neighbor_sampled_probabilities is not None:
                        # for extreme case that node_neighbor_sampled_probabilities only contains -1e10, which will make the denominator of softmax be zero,
                        # torch.softmax() function can tackle this case
                        node_neighbor_sampled_probabilities = torch.softmax(torch.from_numpy(node_neighbor_sampled_probabilities).float(), dim=0).numpy()
                    if self.seed is None:
                        sampled_indices = np.random.choice(a=len(node_neighbor_ids), size=num_neighbors, p=node_neighbor_sampled_probabilities)
                    else:
                        sampled_indices = self.random_state.choice(a=len(node_neighbor_ids), size=num_neighbors, p=node_neighbor_sampled_probabilities)

                    nodes_neighbor_ids[idx, :] = node_neighbor_ids[sampled_indices]
                    nodes_edge_ids[idx, :] = node_edge_ids[sampled_indices]
                    nodes_neighbor_times[idx, :] = node_neighbor_times[sampled_indices]

                    # resort based on timestamps, return the ids in sorted increasing order, note this maybe unstable when multiple edges happen at the same time
                    # (we still do this though this is unnecessary for TGAT or CAWN to guarantee the order of nodes,
                    # since TGAT computes in an order-agnostic manner with relative time encoding, and CAWN computes for each walk while the sampled nodes are in different walks)
                    sorted_position = nodes_neighbor_times[idx, :].argsort()
                    nodes_neighbor_ids[idx, :] = nodes_neighbor_ids[idx, :][sorted_position]
                    nodes_edge_ids[idx, :] = nodes_edge_ids[idx, :][sorted_position]
                    nodes_neighbor_times[idx, :] = nodes_neighbor_times[idx, :][sorted_position]
                elif self.sample_neighbor_strategy == 'recent':
                    # Take most recent interactions with number num_neighbors
                    node_neighbor_ids = node_neighbor_ids[-num_neighbors:]
                    node_edge_ids = node_edge_ids[-num_neighbors:]
                    node_neighbor_times = node_neighbor_times[-num_neighbors:]

                    # put the neighbors' information at the back positions
                    nodes_neighbor_ids[idx, num_neighbors - len(node_neighbor_ids):] = node_neighbor_ids
                    nodes_edge_ids[idx, num_neighbors - len(node_edge_ids):] = node_edge_ids
                    nodes_neighbor_times[idx, num_neighbors - len(node_neighbor_times):] = node_neighbor_times
                else:
                    raise ValueError(f'Not implemented error for sample_neighbor_strategy {self.sample_neighbor_strategy}!')

        # three ndarrays, with shape (batch_size, num_neighbors)
        return nodes_neighbor_ids, nodes_edge_ids, nodes_neighbor_times

    def get_multi_hop_neighbors(self, num_hops: int, node_ids: np.ndarray, node_interact_times: np.ndarray, num_neighbors: int = 20):
        """
        get historical neighbors of nodes in node_ids within num_hops hops
        :param num_hops: int, number of sampled hops
        :param node_ids: ndarray, shape (batch_size, ), node ids
        :param node_interact_times: ndarray, shape (batch_size, ), node interaction times
        :param num_neighbors: int, number of neighbors to sample for each node
        :return:
        """
        assert num_hops > 0, 'Number of sampled hops should be greater than 0!'

        # get the temporal neighbors at the first hop
        # nodes_neighbor_ids, nodes_edge_ids, nodes_neighbor_times -> ndarray, shape (batch_size, num_neighbors)
        nodes_neighbor_ids, nodes_edge_ids, nodes_neighbor_times = self.get_historical_neighbors(node_ids=node_ids,
                                                                                                 node_interact_times=node_interact_times,
                                                                                                 num_neighbors=num_neighbors)
        # three lists to store the neighbor ids, edge ids and interaction timestamp information
        nodes_neighbor_ids_list = [nodes_neighbor_ids]
        nodes_edge_ids_list = [nodes_edge_ids]
        nodes_neighbor_times_list = [nodes_neighbor_times]
        for hop in range(1, num_hops):
            # get information of neighbors sampled at the current hop
            # three ndarrays, with shape (batch_size * num_neighbors ** hop, num_neighbors)
            nodes_neighbor_ids, nodes_edge_ids, nodes_neighbor_times = self.get_historical_neighbors(node_ids=nodes_neighbor_ids_list[-1].flatten(),
                                                                                                     node_interact_times=nodes_neighbor_times_list[-1].flatten(),
                                                                                                     num_neighbors=num_neighbors)
            # three ndarrays with shape (batch_size, num_neighbors ** (hop + 1))
            nodes_neighbor_ids = nodes_neighbor_ids.reshape(len(node_ids), -1)
            nodes_edge_ids = nodes_edge_ids.reshape(len(node_ids), -1)
            nodes_neighbor_times = nodes_neighbor_times.reshape(len(node_ids), -1)

            nodes_neighbor_ids_list.append(nodes_neighbor_ids)
            nodes_edge_ids_list.append(nodes_edge_ids)
            nodes_neighbor_times_list.append(nodes_neighbor_times)

        # tuple, each element in the tuple is a list of num_hops ndarrays, each with shape (batch_size, num_neighbors ** current_hop)
        return nodes_neighbor_ids_list, nodes_edge_ids_list, nodes_neighbor_times_list

    def get_all_first_hop_neighbors(self, node_ids: np.ndarray, node_interact_times: np.ndarray):
        """
        get historical neighbors of nodes in node_ids at the first hop with max_num_neighbors as the maximal number of neighbors (make the computation feasible)
        :param node_ids: ndarray, shape (batch_size, ), node ids
        :param node_interact_times: ndarray, shape (batch_size, ), node interaction times
        :return:
        """
        # 确保输入是 numpy 数组（处理 CUDA tensor 的情况）
        if isinstance(node_ids, torch.Tensor):
            node_ids = node_ids.detach().cpu().numpy()
        if isinstance(node_interact_times, torch.Tensor):
            node_interact_times = node_interact_times.detach().cpu().numpy()
        
        # three lists to store the first-hop neighbor ids, edge ids and interaction timestamp information, with batch_size as the list length
        nodes_neighbor_ids_list, nodes_edge_ids_list, nodes_neighbor_times_list = [], [], []
        # get the temporal neighbors at the first hop
        for idx, (node_id, node_interact_time) in enumerate(zip(node_ids, node_interact_times)):
            # 确保 node_id 和 node_interact_time 是 Python 标量
            if isinstance(node_id, (torch.Tensor, np.ndarray)):
                node_id = int(node_id.item() if hasattr(node_id, 'item') else node_id)
            if isinstance(node_interact_time, (torch.Tensor, np.ndarray)):
                node_interact_time = float(node_interact_time.item() if hasattr(node_interact_time, 'item') else node_interact_time)
            
            # find neighbors that interacted with node_id before time node_interact_time
            node_neighbor_ids, node_edge_ids, node_neighbor_times, _ = self.find_neighbors_before(node_id=node_id,
                                                                                                  interact_time=node_interact_time,
                                                                                                  return_sampled_probabilities=False)
            nodes_neighbor_ids_list.append(node_neighbor_ids)
            nodes_edge_ids_list.append(node_edge_ids)
            nodes_neighbor_times_list.append(node_neighbor_times)

        return nodes_neighbor_ids_list, nodes_edge_ids_list, nodes_neighbor_times_list

    def get_all_first_hop_neighbors_fast(self, node_ids: np.ndarray, node_interact_times: np.ndarray):
        """
        Numba-accelerated 1-hop neighbor lookup (for DyGFormerTW / DyGFormerFF only).
        Falls back to get_all_first_hop_neighbors when Numba is unavailable.
        """
        if isinstance(node_ids, torch.Tensor):
            node_ids = node_ids.detach().cpu().numpy()
        if isinstance(node_interact_times, torch.Tensor):
            node_interact_times = node_interact_times.detach().cpu().numpy()

        if not (_NUMBA_AVAILABLE and hasattr(self, '_csr_ptr')):
            return self.get_all_first_hop_neighbors(node_ids, node_interact_times)

        node_ids_i64 = node_ids.astype(np.int64)
        times_f64    = node_interact_times.astype(np.float64)
        batch_size   = len(node_ids_i64)

        # compute exact buffer size needed (one searchsorted per node)
        max_total = 0
        for i in range(batch_size):
            nid = int(node_ids_i64[i])
            t   = float(times_f64[i])
            lo  = int(self._csr_ptr[nid])
            hi  = int(self._csr_ptr[nid + 1])
            max_total += int(np.searchsorted(self._csr_flat_times[lo:hi], t))

        out_ptr       = np.empty(batch_size + 1, dtype=np.int64)
        if max_total == 0:
            return ([np.array([], dtype=np.longlong)] * batch_size,
                    [np.array([], dtype=np.longlong)] * batch_size,
                    [np.array([], dtype=np.float32)]  * batch_size)

        out_ids_buf   = np.empty(max_total, dtype=np.int64)
        out_edges_buf = np.empty(max_total, dtype=np.int64)
        out_times_buf = np.empty(max_total, dtype=np.float64)

        _sample_1hop_batch_numba(
            node_ids_i64, times_f64,
            self._csr_flat_ids, self._csr_flat_edges, self._csr_flat_times,
            self._csr_ptr,
            out_ptr, out_ids_buf, out_edges_buf, out_times_buf,
        )

        ids_list, edges_list, times_list = [], [], []
        for i in range(batch_size):
            lo, hi = int(out_ptr[i]), int(out_ptr[i + 1])
            ids_list.append(out_ids_buf[lo:hi].astype(np.longlong).copy())
            edges_list.append(out_edges_buf[lo:hi].astype(np.longlong).copy())
            times_list.append(out_times_buf[lo:hi].astype(np.float32).copy())

        return ids_list, edges_list, times_list

    def get_all_second_hop_neighbors(self, node_ids: np.ndarray, node_interact_times: np.ndarray,
                                    nodes_first_hop_neighbor_ids_list: list, nodes_first_hop_edge_ids_list: list,
                                    nodes_first_hop_neighbor_times_list: list,
                                    max_second_hop_length: int = None):
        """
        Get second-hop neighbors for each node: for each first-hop (mid, eid, mt), collect neighbors of mid before time mt.
        Return format same as get_all_first_hop_neighbors (list of arrays per node), so 1-hop and 2-hop can be concatenated.
        Optimized version: uses heap to maintain only the most recent N neighbors when max_second_hop_length is small.
        :param node_ids: ndarray, shape (batch_size, )
        :param node_interact_times: ndarray, shape (batch_size, )
        :param nodes_first_hop_neighbor_ids_list: list of ndarrays, first-hop neighbor ids per node
        :param nodes_first_hop_edge_ids_list: list of ndarrays, first-hop edge ids per node
        :param nodes_first_hop_neighbor_times_list: list of ndarrays, first-hop interaction times per node
        :param max_second_hop_length: int or None, max number of 2-hop neighbors to keep per node (most recent); None = no truncation
        :return: nodes_second_hop_ids_list, nodes_second_hop_edge_ids_list, nodes_second_hop_times_list (same list format)
        """
        if isinstance(node_ids, torch.Tensor):
            node_ids = node_ids.detach().cpu().numpy()
        if isinstance(node_interact_times, torch.Tensor):
            node_interact_times = node_interact_times.detach().cpu().numpy()

        use_numba = (_NUMBA_AVAILABLE
                     and hasattr(self, '_csr_ptr')
                     and max_second_hop_length is not None)
        use_heap_optimization = (not use_numba
                                 and max_second_hop_length is not None
                                 and max_second_hop_length <= 20)

        if use_numba:
            _csr_ptr   = self._csr_ptr
            _csr_ids   = self._csr_flat_ids
            _csr_edges = self._csr_flat_edges
            _csr_times = self._csr_flat_times
            _max_k     = max_second_hop_length

        def _process_one_node(idx):
            n1_ids   = nodes_first_hop_neighbor_ids_list[idx]
            n1_edges = nodes_first_hop_edge_ids_list[idx]
            n1_times = nodes_first_hop_neighbor_times_list[idx]

            if use_numba:
                if len(n1_ids) == 0:
                    return (np.array([], dtype=np.longlong),
                            np.array([], dtype=np.longlong),
                            np.array([], dtype=np.float32))
                nb_ids, nb_edges, nb_times = _sample_2hop_numba(
                    n1_ids.astype(np.int64),
                    n1_edges.astype(np.int64),
                    n1_times.astype(np.float64),
                    _csr_ids, _csr_edges, _csr_times, _csr_ptr, _max_k,
                )
                return (nb_ids.astype(np.longlong),
                        nb_edges.astype(np.longlong),
                        nb_times.astype(np.float32))

            heap = []
            if use_heap_optimization:
                for (mid, eid, mt) in zip(n1_ids, n1_edges, n1_times):
                    nb_ids, nb_edges, nb_times, _ = self.find_neighbors_before(
                        node_id=int(mid), interact_time=float(mt),
                        return_sampled_probabilities=False)
                    if len(nb_ids) > max_second_hop_length:
                        nb_ids   = nb_ids[-max_second_hop_length:]
                        nb_edges = nb_edges[-max_second_hop_length:]
                        nb_times = nb_times[-max_second_hop_length:]
                    for nb_id, nb_edge, nb_time in zip(nb_ids, nb_edges, nb_times):
                        if len(heap) < max_second_hop_length:
                            heapq.heappush(heap, (nb_time, nb_id, nb_edge))
                        elif nb_time > heap[0][0]:
                            heapq.heapreplace(heap, (nb_time, nb_id, nb_edge))
            else:
                for (mid, eid, mt) in zip(n1_ids, n1_edges, n1_times):
                    nb_ids, nb_edges, nb_times, _ = self.find_neighbors_before(
                        node_id=int(mid), interact_time=float(mt),
                        return_sampled_probabilities=False)
                    for nb_id, nb_edge, nb_time in zip(nb_ids, nb_edges, nb_times):
                        if max_second_hop_length is None:
                            heapq.heappush(heap, (nb_time, nb_id, nb_edge))
                        elif len(heap) < max_second_hop_length:
                            heapq.heappush(heap, (nb_time, nb_id, nb_edge))
                        elif nb_time > heap[0][0]:
                            heapq.heapreplace(heap, (nb_time, nb_id, nb_edge))

            if len(heap) > 0:
                sorted_items = sorted(heap)
                return (
                    np.array([x[1] for x in sorted_items], dtype=np.longlong),
                    np.array([x[2] for x in sorted_items], dtype=np.longlong),
                    np.array([x[0] for x in sorted_items], dtype=np.float32),
                )
            return (
                np.array([], dtype=np.longlong),
                np.array([], dtype=np.longlong),
                np.array([], dtype=np.float32),
            )

        n_workers = min(len(node_ids), os.cpu_count() or 4)
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            results = list(executor.map(_process_one_node, range(len(node_ids))))

        nodes_second_hop_ids_list      = [r[0] for r in results]
        nodes_second_hop_edge_ids_list = [r[1] for r in results]
        nodes_second_hop_times_list    = [r[2] for r in results]

        return nodes_second_hop_ids_list, nodes_second_hop_edge_ids_list, nodes_second_hop_times_list

    def reset_random_state(self):
        """
        reset the random state by self.seed
        :return:
        """
        self.random_state = np.random.RandomState(self.seed)


def get_neighbor_sampler(data: Data, sample_neighbor_strategy: str = 'uniform', time_scaling_factor: float = 0.0, seed: int = None):
    """
    get neighbor sampler
    :param data: Data
    :param sample_neighbor_strategy: str, how to sample historical neighbors, 'uniform', 'recent', or 'time_interval_aware''
    :param time_scaling_factor: float, a hyper-parameter that controls the sampling preference with time interval,
    a large time_scaling_factor tends to sample more on recent links, this parameter works when sample_neighbor_strategy == 'time_interval_aware'
    :param seed: int, random seed
    :return:
    """
    max_node_id = max(data.src_node_ids.max(), data.dst_node_ids.max())
    # the adjacency vector stores edges for each node (source or destination), undirected
    # adj_list, list of list, where each element is a list of triple tuple (node_id, edge_id, timestamp)
    # the list at the first position in adj_list is empty
    adj_list = [[] for _ in range(max_node_id + 1)]
    for src_node_id, dst_node_id, edge_id, node_interact_time in zip(data.src_node_ids, data.dst_node_ids, data.edge_ids, data.node_interact_times):
        adj_list[src_node_id].append((dst_node_id, edge_id, node_interact_time))
        adj_list[dst_node_id].append((src_node_id, edge_id, node_interact_time))

    return NeighborSampler(adj_list=adj_list, sample_neighbor_strategy=sample_neighbor_strategy, time_scaling_factor=time_scaling_factor, seed=seed)


class DirectedNeighborSampler(NeighborSampler):
    """
    在 NeighborSampler 基础上额外记录每条邻接边的方向（出边/入边），
    用于分别统计入度和出度。其他模型继续使用父类 NeighborSampler，不受影响。
    """

    def __init__(self, adj_list: list, adj_list_directed: list,
                 sample_neighbor_strategy: str = 'uniform',
                 time_scaling_factor: float = 0.0, seed: int = None):
        """
        :param adj_list: list of list of (neighbor_id, edge_id, timestamp)，传给父类
        :param adj_list_directed: list of list of (neighbor_id, edge_id, timestamp, is_outgoing)，is_outgoing=True 表示出边
        :param sample_neighbor_strategy: str
        :param time_scaling_factor: float
        :param seed: int
        """
        super(DirectedNeighborSampler, self).__init__(
            adj_list=adj_list,
            sample_neighbor_strategy=sample_neighbor_strategy,
            time_scaling_factor=time_scaling_factor,
            seed=seed
        )
        # 与父类的 nodes_neighbor_times 严格按相同方式排序，保证索引对齐
        self.nodes_neighbor_is_outgoing = []
        for per_node_neighbors in adj_list_directed:
            sorted_neighbors = sorted(per_node_neighbors, key=lambda x: x[2])
            self.nodes_neighbor_is_outgoing.append(
                np.array([x[3] for x in sorted_neighbors], dtype=bool)
            )

    def find_in_out_degrees_before(self, node_id: int, interact_time: float):
        """
        返回 node_id 在 interact_time 之前（严格小于）的入度和出度。
        与父类 find_neighbors_before 使用同一个 searchsorted 切片，保证对齐。
        :param node_id: int
        :param interact_time: float
        :return: (in_degree, out_degree), both int
        """
        import torch as _torch
        if isinstance(interact_time, (_torch.Tensor, np.ndarray)):
            interact_time = float(interact_time.item() if hasattr(interact_time, 'item') else interact_time)
        if isinstance(node_id, (_torch.Tensor, np.ndarray)):
            node_id = int(node_id.item() if hasattr(node_id, 'item') else node_id)

        i = np.searchsorted(self.nodes_neighbor_times[node_id], interact_time)
        is_outgoing = self.nodes_neighbor_is_outgoing[node_id][:i]
        out_degree = int(is_outgoing.sum())
        in_degree = i - out_degree
        return in_degree, out_degree


def get_directed_neighbor_sampler(data: Data, sample_neighbor_strategy: str = 'uniform',
                                   time_scaling_factor: float = 0.0, seed: int = None):
    """
    构建 DirectedNeighborSampler，同时维护无向邻接表（供父类使用）和带方向标志的邻接表（供入/出度计算使用）。
    :param data: Data
    :param sample_neighbor_strategy: str
    :param time_scaling_factor: float
    :param seed: int
    :return: DirectedNeighborSampler
    """
    max_node_id = max(data.src_node_ids.max(), data.dst_node_ids.max())
    adj_list = [[] for _ in range(max_node_id + 1)]
    adj_list_directed = [[] for _ in range(max_node_id + 1)]

    for src_node_id, dst_node_id, edge_id, node_interact_time in zip(
            data.src_node_ids, data.dst_node_ids, data.edge_ids, data.node_interact_times):
        adj_list[src_node_id].append((dst_node_id, edge_id, node_interact_time))
        adj_list[dst_node_id].append((src_node_id, edge_id, node_interact_time))
        adj_list_directed[src_node_id].append((dst_node_id, edge_id, node_interact_time, True))   # src 视角：出边
        adj_list_directed[dst_node_id].append((src_node_id, edge_id, node_interact_time, False))  # dst 视角：入边

    return DirectedNeighborSampler(
        adj_list=adj_list,
        adj_list_directed=adj_list_directed,
        sample_neighbor_strategy=sample_neighbor_strategy,
        time_scaling_factor=time_scaling_factor,
        seed=seed
    )


class HistoricalNeighborSampler:
    """
    Edge-centric neighbor sampler for RepeatMixer.
    For each (node_id, interact_node) pair, returns historical edges between them before a given time.
    """

    def __init__(self, data: Data, sample_neighbor_strategy: str = 'recent', time_scaling_factor: float = 0.0, seed: int = None):
        self.sample_neighbor_strategy = sample_neighbor_strategy
        self.seed = seed
        self.time_scaling_factor = time_scaling_factor

        # edge_pair_to_list: (src, dst) -> [(edge_id, timestamp), ...] sorted by time
        self.edge_pair_edges = {}
        for src, dst, eid, t in zip(data.src_node_ids, data.dst_node_ids, data.edge_ids, data.node_interact_times):
            key = (int(src), int(dst))
            if key not in self.edge_pair_edges:
                self.edge_pair_edges[key] = []
            self.edge_pair_edges[key].append((int(eid), float(t)))
            # also add reverse (dst, src) for undirected query
            key_rev = (int(dst), int(src))
            if key_rev not in self.edge_pair_edges:
                self.edge_pair_edges[key_rev] = []
            self.edge_pair_edges[key_rev].append((int(eid), float(t)))

        for k in self.edge_pair_edges:
            self.edge_pair_edges[k] = sorted(self.edge_pair_edges[k], key=lambda x: x[1])

        if self.seed is not None:
            self.random_state = np.random.RandomState(self.seed)

    def get_historical_neighbors(self, node_ids: np.ndarray, interact_nodes: np.ndarray,
                                node_interact_times: np.ndarray, num_neighbors: int = 20):
        """
        For each (node_id, interact_node), get historical edges between them before node_interact_time.
        Returns: neighbor_node_ids (the interact_node), edge_ids, timestamps.
        """
        batch_size = len(node_ids)
        nodes_neighbor_ids = np.zeros((batch_size, num_neighbors), dtype=np.longlong)
        nodes_edge_ids = np.zeros((batch_size, num_neighbors), dtype=np.longlong)
        nodes_neighbor_times = np.zeros((batch_size, num_neighbors), dtype=np.float32)

        for idx in range(batch_size):
            nid = int(node_ids[idx]) if not isinstance(node_ids[idx], (np.integer, int)) else node_ids[idx]
            inid = int(interact_nodes[idx]) if not isinstance(interact_nodes[idx], (np.integer, int)) else interact_nodes[idx]
            t = float(node_interact_times[idx]) if not isinstance(node_interact_times[idx], (np.floating, float)) else node_interact_times[idx]

            key = (nid, inid)
            if key not in self.edge_pair_edges:
                continue

            edges = self.edge_pair_edges[key]
            # filter by time
            before = [(eid, ts) for eid, ts in edges if ts < t]
            if len(before) == 0:
                continue

            if self.sample_neighbor_strategy == 'recent':
                selected = before[-num_neighbors:]
            else:
                if self.seed is not None:
                    indices = self.random_state.choice(len(before), size=min(num_neighbors, len(before)), replace=len(before) < num_neighbors)
                else:
                    indices = np.random.choice(len(before), size=min(num_neighbors, len(before)), replace=len(before) < num_neighbors)
                selected = [before[i] for i in indices]
                selected = sorted(selected, key=lambda x: x[1])

            pad = num_neighbors - len(selected)
            for j, (eid, ts) in enumerate(selected):
                nodes_neighbor_ids[idx, pad + j] = inid
                nodes_edge_ids[idx, pad + j] = eid
                nodes_neighbor_times[idx, pad + j] = ts

        return nodes_neighbor_ids, nodes_edge_ids, nodes_neighbor_times

    def get_two_hop_historical_neighbors(self, node_ids: np.ndarray, scopes: np.ndarray,
                                         node_interact_times: np.ndarray, num_neighbors: int = 20):
        """Placeholder for two-hop sampling; returns empty arrays for compatibility."""
        batch_size = len(node_ids)
        return (np.zeros((batch_size, num_neighbors), dtype=np.longlong),
                np.zeros((batch_size, num_neighbors), dtype=np.longlong),
                np.zeros((batch_size, num_neighbors), dtype=np.float32))

    def reset_random_state(self):
        if self.seed is not None:
            self.random_state = np.random.RandomState(self.seed)


def get_edge_neighbor_sampler(data: Data, sample_neighbor_strategy: str = 'recent', time_scaling_factor: float = 0.0, seed: int = None):
    """Get HistoricalNeighborSampler for RepeatMixer."""
    return HistoricalNeighborSampler(data=data, sample_neighbor_strategy=sample_neighbor_strategy,
                                    time_scaling_factor=time_scaling_factor, seed=seed)


class NegativeEdgeSampler(object):

    def __init__(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray, interact_times: np.ndarray = None, last_observed_time: float = None,
                 negative_sample_strategy: str = 'random', seed: int = None):
        """
        Negative Edge Sampler, which supports three strategies: "random", "historical", "inductive".
        :param src_node_ids: ndarray, (num_src_nodes, ), source node ids, num_src_nodes == num_dst_nodes
        :param dst_node_ids: ndarray, (num_dst_nodes, ), destination node ids
        :param interact_times: ndarray, (num_src_nodes, ), interaction timestamps
        :param last_observed_time: float, time of the last observation (for inductive negative sampling strategy)
        :param negative_sample_strategy: str, negative sampling strategy, can be "random", "historical", "inductive"
        :param seed: int, random seed
        """
        self.seed = seed
        self.negative_sample_strategy = negative_sample_strategy
        self.src_node_ids = src_node_ids
        self.dst_node_ids = dst_node_ids
        self.interact_times = interact_times
        self.unique_src_node_ids = np.unique(src_node_ids)
        self.unique_dst_node_ids = np.unique(dst_node_ids)
        self.unique_interact_times = np.unique(interact_times)
        self.earliest_time = min(self.unique_interact_times)
        self.last_observed_time = last_observed_time

        if self.negative_sample_strategy != 'random':
            # all the possible edges that connect source nodes in self.unique_src_node_ids with destination nodes in self.unique_dst_node_ids
            self.possible_edges = set((src_node_id, dst_node_id) for src_node_id in self.unique_src_node_ids for dst_node_id in self.unique_dst_node_ids)

        if self.negative_sample_strategy == 'inductive':
            # set of observed edges
            self.observed_edges = self.get_unique_edges_between_start_end_time(self.earliest_time, self.last_observed_time)

        if self.seed is not None:
            self.random_state = np.random.RandomState(self.seed)

    def get_unique_edges_between_start_end_time(self, start_time: float, end_time: float):
        """
        get unique edges happened between start and end time
        :param start_time: float, start timestamp
        :param end_time: float, end timestamp
        :return: a set of edges, where each edge is a tuple of (src_node_id, dst_node_id)
        """
        selected_time_interval = np.logical_and(self.interact_times >= start_time, self.interact_times <= end_time)
        # return the unique select source and destination nodes in the selected time interval
        return set((src_node_id, dst_node_id) for src_node_id, dst_node_id in zip(self.src_node_ids[selected_time_interval], self.dst_node_ids[selected_time_interval]))

    def sample(self, size: int, batch_src_node_ids: np.ndarray = None, batch_dst_node_ids: np.ndarray = None,
               current_batch_start_time: float = 0.0, current_batch_end_time: float = 0.0):
        """
        sample negative edges, support random, historical and inductive sampling strategy
        :param size: int, number of sampled negative edges
        :param batch_src_node_ids: ndarray, shape (batch_size, ), source node ids in the current batch
        :param batch_dst_node_ids: ndarray, shape (batch_size, ), destination node ids in the current batch
        :param current_batch_start_time: float, start time in the current batch
        :param current_batch_end_time: float, end time in the current batch
        :return:
        """
        if self.negative_sample_strategy == 'random':
            negative_src_node_ids, negative_dst_node_ids = self.random_sample(size=size)
        elif self.negative_sample_strategy == 'historical':
            negative_src_node_ids, negative_dst_node_ids = self.historical_sample(size=size, batch_src_node_ids=batch_src_node_ids,
                                                                                  batch_dst_node_ids=batch_dst_node_ids,
                                                                                  current_batch_start_time=current_batch_start_time,
                                                                                  current_batch_end_time=current_batch_end_time)
        elif self.negative_sample_strategy == 'inductive':
            negative_src_node_ids, negative_dst_node_ids = self.inductive_sample(size=size, batch_src_node_ids=batch_src_node_ids,
                                                                                 batch_dst_node_ids=batch_dst_node_ids,
                                                                                 current_batch_start_time=current_batch_start_time,
                                                                                 current_batch_end_time=current_batch_end_time)
        else:
            raise ValueError(f'Not implemented error for negative_sample_strategy {self.negative_sample_strategy}!')
        return negative_src_node_ids, negative_dst_node_ids

    def random_sample(self, size: int):
        """
        random sampling strategy, which is used by previous works
        :param size: int, number of sampled negative edges
        :return:
        """
        if self.seed is None:
            random_sample_edge_src_node_indices = np.random.randint(0, len(self.unique_src_node_ids), size)
            random_sample_edge_dst_node_indices = np.random.randint(0, len(self.unique_dst_node_ids), size)
        else:
            random_sample_edge_src_node_indices = self.random_state.randint(0, len(self.unique_src_node_ids), size)
            random_sample_edge_dst_node_indices = self.random_state.randint(0, len(self.unique_dst_node_ids), size)
        return self.unique_src_node_ids[random_sample_edge_src_node_indices], self.unique_dst_node_ids[random_sample_edge_dst_node_indices]

    def random_sample_with_collision_check(self, size: int, batch_src_node_ids: np.ndarray, batch_dst_node_ids: np.ndarray):
        """
        random sampling strategy with collision check, which guarantees that the sampled edges do not appear in the current batch,
        used for historical and inductive sampling strategy
        :param size: int, number of sampled negative edges
        :param batch_src_node_ids: ndarray, shape (batch_size, ), source node ids in the current batch
        :param batch_dst_node_ids: ndarray, shape (batch_size, ), destination node ids in the current batch
        :return:
        """
        assert batch_src_node_ids is not None and batch_dst_node_ids is not None
        batch_edges = set((batch_src_node_id, batch_dst_node_id) for batch_src_node_id, batch_dst_node_id in zip(batch_src_node_ids, batch_dst_node_ids))
        possible_random_edges = list(self.possible_edges - batch_edges)
        assert len(possible_random_edges) > 0
        # if replace is True, then a value in the list can be selected multiple times, otherwise, a value can be selected only once at most
        random_edge_indices = self.random_state.choice(len(possible_random_edges), size=size, replace=len(possible_random_edges) < size)
        return np.array([possible_random_edges[random_edge_idx][0] for random_edge_idx in random_edge_indices]), \
               np.array([possible_random_edges[random_edge_idx][1] for random_edge_idx in random_edge_indices])

    def historical_sample(self, size: int, batch_src_node_ids: np.ndarray, batch_dst_node_ids: np.ndarray,
                          current_batch_start_time: float, current_batch_end_time: float):
        """
        historical sampling strategy, first randomly samples among historical edges that are not in the current batch,
        if number of historical edges is smaller than size, then fill in remaining edges with randomly sampled edges
        :param size: int, number of sampled negative edges
        :param batch_src_node_ids: ndarray, shape (batch_size, ), source node ids in the current batch
        :param batch_dst_node_ids: ndarray, shape (batch_size, ), destination node ids in the current batch
        :param current_batch_start_time: float, start time in the current batch
        :param current_batch_end_time: float, end time in the current batch
        :return:
        """
        assert self.seed is not None
        # get historical edges up to current_batch_start_time
        historical_edges = self.get_unique_edges_between_start_end_time(start_time=self.earliest_time, end_time=current_batch_start_time)
        # get edges in the current batch
        current_batch_edges = self.get_unique_edges_between_start_end_time(start_time=current_batch_start_time, end_time=current_batch_end_time)
        # get source and destination node ids of unique historical edges
        unique_historical_edges = historical_edges - current_batch_edges
        unique_historical_edges_src_node_ids = np.array([edge[0] for edge in unique_historical_edges])
        unique_historical_edges_dst_node_ids = np.array([edge[1] for edge in unique_historical_edges])

        # if sample size is larger than number of unique historical edges, then fill in remaining edges with randomly sampled edges with collision check
        if size > len(unique_historical_edges):
            num_random_sample_edges = size - len(unique_historical_edges)
            random_sample_src_node_ids, random_sample_dst_node_ids = self.random_sample_with_collision_check(size=num_random_sample_edges,
                                                                                                             batch_src_node_ids=batch_src_node_ids,
                                                                                                             batch_dst_node_ids=batch_dst_node_ids)

            negative_src_node_ids = np.concatenate([random_sample_src_node_ids, unique_historical_edges_src_node_ids])
            negative_dst_node_ids = np.concatenate([random_sample_dst_node_ids, unique_historical_edges_dst_node_ids])
        else:
            historical_sample_edge_node_indices = self.random_state.choice(len(unique_historical_edges), size=size, replace=False)
            negative_src_node_ids = unique_historical_edges_src_node_ids[historical_sample_edge_node_indices]
            negative_dst_node_ids = unique_historical_edges_dst_node_ids[historical_sample_edge_node_indices]

        # Note that if one of the input of np.concatenate is empty, the output will be composed of floats.
        # Hence, convert the type to long to guarantee valid index
        return negative_src_node_ids.astype(np.longlong), negative_dst_node_ids.astype(np.longlong)

    def inductive_sample(self, size: int, batch_src_node_ids: np.ndarray, batch_dst_node_ids: np.ndarray,
                         current_batch_start_time: float, current_batch_end_time: float):
        """
        inductive sampling strategy, first randomly samples among inductive edges that are not in self.observed_edges and the current batch,
        if number of inductive edges is smaller than size, then fill in remaining edges with randomly sampled edges
        :param size: int, number of sampled negative edges
        :param batch_src_node_ids: ndarray, shape (batch_size, ), source node ids in the current batch
        :param batch_dst_node_ids: ndarray, shape (batch_size, ), destination node ids in the current batch
        :param current_batch_start_time: float, start time in the current batch
        :param current_batch_end_time: float, end time in the current batch
        :return:
        """
        assert self.seed is not None
        # get historical edges up to current_batch_start_time
        historical_edges = self.get_unique_edges_between_start_end_time(start_time=self.earliest_time, end_time=current_batch_start_time)
        # get edges in the current batch
        current_batch_edges = self.get_unique_edges_between_start_end_time(start_time=current_batch_start_time, end_time=current_batch_end_time)
        # get source and destination node ids of historical edges but 1) not in self.observed_edges; 2) not in the current batch
        unique_inductive_edges = historical_edges - self.observed_edges - current_batch_edges
        unique_inductive_edges_src_node_ids = np.array([edge[0] for edge in unique_inductive_edges])
        unique_inductive_edges_dst_node_ids = np.array([edge[1] for edge in unique_inductive_edges])

        # if sample size is larger than number of unique inductive edges, then fill in remaining edges with randomly sampled edges
        if size > len(unique_inductive_edges):
            num_random_sample_edges = size - len(unique_inductive_edges)
            random_sample_src_node_ids, random_sample_dst_node_ids = self.random_sample_with_collision_check(size=num_random_sample_edges,
                                                                                                             batch_src_node_ids=batch_src_node_ids,
                                                                                                             batch_dst_node_ids=batch_dst_node_ids)

            negative_src_node_ids = np.concatenate([random_sample_src_node_ids, unique_inductive_edges_src_node_ids])
            negative_dst_node_ids = np.concatenate([random_sample_dst_node_ids, unique_inductive_edges_dst_node_ids])
        else:
            inductive_sample_edge_node_indices = self.random_state.choice(len(unique_inductive_edges), size=size, replace=False)
            negative_src_node_ids = unique_inductive_edges_src_node_ids[inductive_sample_edge_node_indices]
            negative_dst_node_ids = unique_inductive_edges_dst_node_ids[inductive_sample_edge_node_indices]

        # Note that if one of the input of np.concatenate is empty, the output will be composed of floats.
        # Hence, convert the type to long to guarantee valid index
        return negative_src_node_ids.astype(np.longlong), negative_dst_node_ids.astype(np.longlong)

    def reset_random_state(self):
        """
        reset the random state by self.seed
        :return:
        """
        self.random_state = np.random.RandomState(self.seed)
