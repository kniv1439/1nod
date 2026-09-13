# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import networkx as nx
from collections import Counter
from typing import Tuple, Dict, Any, List
from scipy.stats import skew, kurtosis
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
import matplotlib.pyplot as plt

# 尝试不同的导入方式
try:
    from community import community_louvain
except ImportError:
    try:
        import community.community_louvain as community_louvain
    except ImportError:
        try:
            from community import best_partition as community_louvain_best_partition
            class MockCommunityLouvain:
                @staticmethod
                def best_partition(G):
                    return community_louvain_best_partition(G)
            community_louvain = MockCommunityLouvain()
        except ImportError:
            print("警告: 无法导入community_louvain，请安装: pip install python-louvain")
            class SimpleCommunityLouvain:
                @staticmethod
                def best_partition(G):
                    return {node: i for i, node in enumerate(G.nodes())}
            community_louvain = SimpleCommunityLouvain()

# ============================== 数据集配置 ==============================
DATASETS = {
    "USLegis": {"path": "processed_data/USLegis/ml_USLegis.csv", "train_ts_start": 0.0, "train_ts_end": 8.0, "test_ts_start": 10.0, "test_ts_end": 11.0, "type": "political"},
    "CanParl": {"path": "processed_data/CanParl/ml_CanParl.csv", "train_ts_start": 0.0, "train_ts_end": 283996800.0, "test_ts_start": 347155200.0, "test_ts_end": 410227200.0, "type": "political"},
    "UNvote":  {"path": "processed_data/UNvote/ml_UNvote.csv",  "train_ts_start": 0.0, "train_ts_end": 1830297600.0, "test_ts_start": 2019686400.0, "test_ts_end": 2240611200.0, "type": "political"},
    "UNtrade": {"path": "processed_data/UNtrade/ml_UNtrade.csv","train_ts_start": 0.0, "train_ts_end": 757382400.0, "test_ts_start": 883612800.0, "test_ts_end": 978307200.0, "type": "political"},
    "wikipedia":{"path": "processed_data/wikipedia/ml_wikipedia.csv","train_ts_start": 0.0,"train_ts_end": 1862645.0,"test_ts_start": 2218288.0,"test_ts_end": 2678373.0,"type": "social"},
    "reddit":   {"path": "processed_data/reddit/ml_reddit.csv","train_ts_start": 0.0,"train_ts_end": 1882467.555,"test_ts_start": 2261811.988,"test_ts_end": 2678390.016,"type": "social"},
    "uci":      {"path": "processed_data/uci/ml_uci.csv","train_ts_start": 0.0,"train_ts_end": 3834783.0,"test_ts_start": 6714637.0,"test_ts_end": 16736181.0,"type": "social"},
    "enron":    {"path": "processed_data/enron/ml_enron.csv","train_ts_start": 0.0,"train_ts_end": 83842852.0,"test_ts_start": 93431801.0,"test_ts_end": 113740399.0,"type": "social"},
    "mooc":     {"path": "processed_data/mooc/ml_mooc.csv","train_ts_start": 0.0,"train_ts_end": 1917226.0,"test_ts_start": 2250152.0,"test_ts_end": 2572086.0,"type": "mooc"}
}


def _register_lastfm_from_processed_csv(
    datasets: Dict[str, Dict[str, Any]],
    path: str = "processed_data/lastfm/ml_lastfm.csv",
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> None:
    """
    若已预处理出 ml_lastfm.csv，则注册 lastfm。
    时间窗与 utils/DataLoader.py 一致：按时间戳分位数切分（默认 train/val/test = 70%%/15%%/15%%），
    训练段 ts<=val_time，测试段 ts>test_time（中间为验证段，与本脚本 train/test 对比一致地排除）。
    """
    if not os.path.isfile(path):
        return
    df = pd.read_csv(path)
    if "ts" not in df.columns:
        return
    ts = df["ts"].to_numpy(dtype=np.float64)
    if ts.size == 0:
        return
    val_time, test_time = np.quantile(ts, [1.0 - val_ratio - test_ratio, 1.0 - test_ratio])
    datasets["lastfm"] = {
        "path": path,
        "train_ts_start": 0.0,
        "train_ts_end": float(val_time),
        "test_ts_start": float(test_time),
        "test_ts_end": float(ts.max()),
        "type": "social",
    }


_register_lastfm_from_processed_csv(DATASETS)

# ============================== 基础 I/O ==============================
def load_dataframe(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df[["u", "i", "ts"]]

def get_political_time_windows(df, train_ts_start, train_ts_end, test_ts_start, test_ts_end):
    train_df = df[(df["ts"] >= train_ts_start) & (df["ts"] <= train_ts_end)]
    test_df  = df[(df["ts"] >= test_ts_start)  & (df["ts"] <= test_ts_end)]
    return train_df, test_df

def get_windows_for_dataset(name: str, cfg: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = load_dataframe(cfg["path"])
    return get_political_time_windows(df, cfg["train_ts_start"], cfg["train_ts_end"], cfg["test_ts_start"], cfg["test_ts_end"])

# ============================== 建图（无向/有向） ==============================
def build_graph_undirected(df: pd.DataFrame) -> nx.Graph:
    edge_list = list(zip(df["u"], df["i"]))
    edge_weights = Counter(edge_list)
    G = nx.Graph()
    for (u, v), w in edge_weights.items():
        if u != v:
            G.add_edge(u, v, weight=float(w))
    return G

def build_graph_directed(df: pd.DataFrame) -> nx.DiGraph:
    edge_list = list(zip(df["u"], df["i"]))
    edge_weights = Counter(edge_list)
    G = nx.DiGraph()
    for (u, v), w in edge_weights.items():
        if u != v:
            G.add_edge(u, v, weight=float(w), count=int(w))
    return G

# ============================== 通用 5 统计量 ==============================
def safe_stats_for_feature(vec: np.ndarray,
                           fisher: bool = False,
                           use_unbiased: bool = True) -> np.ndarray:
    """
    返回 [median, mean, stdev, skewness, kurtosis]
    - fisher=False: Pearson 峰度（正态=3）
    - use_unbiased=True: skew/kurt 无偏估计
    """
    vec = np.asarray(vec, dtype=float)
    if vec.size == 0:
        return np.zeros(5, dtype=float)
    med = float(np.median(vec))
    mean = float(np.mean(vec))
    std = float(np.std(vec, ddof=0))
    skw = skew(vec, bias=not use_unbiased, nan_policy="omit")
    krt = kurtosis(vec, fisher=fisher, bias=not use_unbiased, nan_policy="omit")
    if np.isnan(skw): skw = 0.0
    if np.isnan(krt): krt = 3.0 if not fisher else 0.0
    return np.array([med, mean, std, float(skw), float(krt)], dtype=float)

# ============================== NetSimile 6×5（无向，去掉度数特征） ==============================
def compute_netsimile_node_features(G: nx.Graph) -> pd.DataFrame:
    nodes = list(G.nodes())
    deg = dict(G.degree())
    clust = nx.clustering(G)
    neigh = {u: set(G.neighbors(u)) for u in nodes}

    avg_neigh_deg = {}
    avg_neigh_clust = {}
    ego_internal_edges, ego_outgoing_edges, ego_neighbor_nodes = {}, {}, {}
    for u in nodes:
        Nu = neigh[u]
        if not Nu:
            avg_neigh_deg[u] = 0.0
            avg_neigh_clust[u] = 0.0
            ego_internal_edges[u] = 0; ego_outgoing_edges[u] = 0; ego_neighbor_nodes[u] = 0
            continue
        avg_neigh_deg[u] = float(np.mean([deg[v] for v in Nu]))
        avg_neigh_clust[u] = float(np.mean([clust[v] for v in Nu]))

        egoG = G.subgraph(Nu)
        ego_internal_edges[u] = egoG.number_of_edges()

        ext_edges = 0; ext_nodes = set()
        for v in Nu:
            for w in G.neighbors(v):
                if w not in Nu:
                    ext_edges += 1
                    ext_nodes.add(w)
        ego_outgoing_edges[u] = ext_edges
        ego_neighbor_nodes[u] = len(ext_nodes)

    return pd.DataFrame({
        "node": nodes,
        "deg": [deg[u] for u in nodes],
        "clust": [clust[u] for u in nodes],
        "avg_neigh_deg": [avg_neigh_deg[u] for u in nodes],
        "avg_neigh_clust": [avg_neigh_clust[u] for u in nodes],
        "ego_internal_edges": [ego_internal_edges[u] for u in nodes],
        "ego_outgoing_edges": [ego_outgoing_edges[u] for u in nodes],
        "ego_neighbor_nodes": [ego_neighbor_nodes[u] for u in nodes],
    })

def compute_netsimile_node_features_weighted(G: nx.DiGraph) -> pd.DataFrame:
    """计算加权版本的NetSimile特征，包含平均邻居加权入度和加权出度"""
    nodes = list(G.nodes())
    deg = dict(G.degree())
    clust = nx.clustering(G)
    neigh = {u: set(G.neighbors(u)) for u in nodes}

    avg_neigh_deg = {}
    avg_neigh_clust = {}
    avg_neigh_weighted_indeg = {}
    avg_neigh_weighted_outdeg = {}
    ego_internal_edges, ego_outgoing_edges, ego_neighbor_nodes = {}, {}, {}
    
    for u in nodes:
        Nu = neigh[u]
        if not Nu:
            avg_neigh_deg[u] = 0.0
            avg_neigh_clust[u] = 0.0
            avg_neigh_weighted_indeg[u] = 0.0
            avg_neigh_weighted_outdeg[u] = 0.0
            ego_internal_edges[u] = 0; ego_outgoing_edges[u] = 0; ego_neighbor_nodes[u] = 0
            continue
        
        avg_neigh_deg[u] = float(np.mean([deg[v] for v in Nu]))
        avg_neigh_clust[u] = float(np.mean([clust[v] for v in Nu]))
        
        # 计算平均邻居加权入度和加权出度
        neigh_weighted_indeg = []
        neigh_weighted_outdeg = []
        for v in Nu:
            # 加权入度：所有指向v的边的权重和
            indeg_weight = sum(data.get('weight', 1) for _, _, data in G.in_edges(v, data=True))
            # 加权出度：所有从v出发的边的权重和
            outdeg_weight = sum(data.get('weight', 1) for _, _, data in G.out_edges(v, data=True))
            neigh_weighted_indeg.append(indeg_weight)
            neigh_weighted_outdeg.append(outdeg_weight)
        
        avg_neigh_weighted_indeg[u] = float(np.mean(neigh_weighted_indeg))
        avg_neigh_weighted_outdeg[u] = float(np.mean(neigh_weighted_outdeg))

        egoG = G.subgraph(Nu)
        ego_internal_edges[u] = egoG.number_of_edges()

        ext_edges = 0; ext_nodes = set()
        for v in Nu:
            for w in G.neighbors(v):
                if w not in Nu:
                    ext_edges += 1
                    ext_nodes.add(w)
        ego_outgoing_edges[u] = ext_edges
        ego_neighbor_nodes[u] = len(ext_nodes)

    return pd.DataFrame({
        "node": nodes,
        "deg": [deg[u] for u in nodes],
        "clust": [clust[u] for u in nodes],
        "avg_neigh_deg": [avg_neigh_deg[u] for u in nodes],
        "avg_neigh_clust": [avg_neigh_clust[u] for u in nodes],
        "avg_neigh_weighted_indeg": [avg_neigh_weighted_indeg[u] for u in nodes],
        "avg_neigh_weighted_outdeg": [avg_neigh_weighted_outdeg[u] for u in nodes],
        "ego_internal_edges": [ego_internal_edges[u] for u in nodes],
        "ego_outgoing_edges": [ego_outgoing_edges[u] for u in nodes],
        "ego_neighbor_nodes": [ego_neighbor_nodes[u] for u in nodes],
    })

def netsimile_signature(G: nx.DiGraph) -> np.ndarray:
    if G.number_of_nodes() == 0:
        return np.zeros(10, dtype=float)  # 2个特征 × 5个统计量 = 10维
    feats = compute_netsimile_node_features_weighted(G)
    # 去掉聚类系数特征和ego特征：clust, avg_neigh_clust, ego_internal_edges, ego_outgoing_edges, ego_neighbor_nodes
    cols = ["avg_neigh_weighted_indeg","avg_neigh_weighted_outdeg"]
    parts = [safe_stats_for_feature(feats[c].to_numpy(dtype=float), fisher=False) for c in cols]
    return np.concatenate(parts).astype(float)

# ============================== Canberra 距离 ==============================
def canberra_distance(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    diff = np.abs(x - y); denom = np.abs(x) + np.abs(y)
    mask = denom > 1e-12
    terms = np.zeros_like(diff); terms[mask] = diff[mask] / denom[mask]
    return float(np.sum(terms))

def canberra_distance_with_contributions(x: np.ndarray, y: np.ndarray) -> Tuple[float, np.ndarray]:
    """计算Canberra距离并返回每个维度的贡献"""
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    diff = np.abs(x - y); denom = np.abs(x) + np.abs(y)
    mask = denom > 1e-12
    terms = np.zeros_like(diff); terms[mask] = diff[mask] / denom[mask]
    return float(np.sum(terms)), terms

# ============================== 新增指标（有向图/边列表上计算） ==============================
# 1) 加权入度分布（节点级 5）
def weighted_indegree_stats(Gd: nx.DiGraph) -> np.ndarray:
    weighted_indeg = np.array([sum(data.get('weight', 1) for _, _, data in Gd.in_edges(n, data=True)) for n in Gd.nodes()], dtype=float)
    return safe_stats_for_feature(weighted_indeg, fisher=False)

# 2) 加权出度分布（节点级 5）
def weighted_outdegree_stats(Gd: nx.DiGraph) -> np.ndarray:
    weighted_outdeg = np.array([sum(data.get('weight', 1) for _, _, data in Gd.out_edges(n, data=True)) for n in Gd.nodes()], dtype=float)
    return safe_stats_for_feature(weighted_outdeg, fisher=False)

# 2.5) Burstiness 参数
def burstiness_parameter(tau: np.ndarray) -> float:
    """
    B = (sigma_tau - mu_tau) / (sigma_tau + mu_tau)
    其中 tau 是 inter-event times。
    """
    tau = np.asarray(tau, dtype=float)
    if tau.size == 0:
        return 0.0
    mu = float(np.mean(tau))
    sigma = float(np.std(tau, ddof=0))
    denom = sigma + mu
    if denom <= 1e-12:
        return 0.0
    return float((sigma - mu) / denom)

# 2.6) 节点 burstness（节点级 5）
def node_burstiness_stats(df: pd.DataFrame) -> np.ndarray:
    """
    B(u): 对每个节点 u，收集其参与的全部交互（不区分源/目标）时间戳，
    排序后计算 inter-event times，再计算 B(u)。
    最后对所有节点的 B(u) 做 5 统计量。
    """
    if df is None or df.empty:
        return np.zeros(5, dtype=float)

    b_vals = []
    all_nodes = pd.Index(df["u"]).union(pd.Index(df["i"]))
    for n in all_nodes:
        ts = df.loc[(df["u"] == n) | (df["i"] == n), "ts"].to_numpy(dtype=float)
        if ts.size <= 1:
            b_vals.append(0.0)
            continue
        ts.sort()
        tau = np.diff(ts)
        b_vals.append(burstiness_parameter(tau))

    return safe_stats_for_feature(np.array(b_vals, dtype=float), fisher=False)

# 2.7) 边 burstness（边级 5）
def edge_burstiness_stats(df: pd.DataFrame) -> np.ndarray:
    """
    B(u,v): 对每个有向边 (u,v)，收集其全部交互时间戳并排序，
    计算 inter-event times，再计算 B(u,v)。
    最后对所有边的 B(u,v) 做 5 统计量。
    """
    if df is None or df.empty:
        return np.zeros(5, dtype=float)

    b_vals = []
    df_sorted = df.sort_values(["u", "i", "ts"])
    for _, g in df_sorted.groupby(["u", "i"]):
        ts = g["ts"].to_numpy(dtype=float)
        tau = np.diff(ts) if ts.size > 1 else np.array([], dtype=float)
        b_vals.append(burstiness_parameter(tau))

    return safe_stats_for_feature(np.array(b_vals, dtype=float), fisher=False)

# 3) 重复交互强度（节点级：入/出各 5）
def node_repeat_strength_out(df: pd.DataFrame) -> np.ndarray:
    if df is None or df.empty: return np.zeros(5, dtype=float)
    events_out = df.groupby("u").size().astype(float)
    uniq_out   = df.groupby("u")["i"].nunique().astype(float)
    idx = events_out.index.union(uniq_out.index)
    e = events_out.reindex(idx, fill_value=0.0).to_numpy()
    k = uniq_out.reindex(idx,   fill_value=0.0).to_numpy()
    strength = e / np.maximum(1.0, k)
    return safe_stats_for_feature(strength, fisher=False)

def node_repeat_strength_in(df: pd.DataFrame) -> np.ndarray:
    if df is None or df.empty: return np.zeros(5, dtype=float)
    events_in = df.groupby("i").size().astype(float)
    uniq_in   = df.groupby("i")["u"].nunique().astype(float)
    idx = events_in.index.union(uniq_in.index)
    e = events_in.reindex(idx, fill_value=0.0).to_numpy()
    k = uniq_in.reindex(idx,   fill_value=0.0).to_numpy()
    strength = e / np.maximum(1.0, k)
    return safe_stats_for_feature(strength, fisher=False)

# 4) 边重复率（边级：重复率分布 5）
def edge_reuse_features(df: pd.DataFrame) -> np.ndarray:
    """
    n_ui: 每个 (u,i) 出现次数
    ρ_ui = 1 - 1/n_ui ∈ [0,1]（n=1 => 0；n→∞ => 1）
    返回: ρ 的 5 统计量 => 共 5 维
    """
    if df is None or df.empty:
        return np.zeros(5, dtype=float)
    cnt = df.groupby(["u","i"]).size().astype(float).to_numpy()
    rho = 1.0 - 1.0 / np.maximum(cnt, 1.0)
    return safe_stats_for_feature(rho, fisher=False)

# 5) 互惠率（节点级 5）
def reciprocity_stats(Gd: nx.DiGraph) -> np.ndarray:
    """
    对每个节点 v：互惠率 r(v) = 双向邻居数 / 唯一邻居总数（入∪出）
    """
    if Gd.number_of_nodes() == 0:
        return np.zeros(5, dtype=float)
    r_list = []
    for v in Gd.nodes():
        outN = set(Gd.successors(v))
        inN  = set(Gd.predecessors(v))
        totalN = outN | inN
        if len(totalN) == 0:
            r_list.append(0.0)
        else:
            mutual = sum(1 for u in outN if Gd.has_edge(u, v))
            r_list.append(mutual / float(len(totalN)))
    return safe_stats_for_feature(np.array(r_list, dtype=float), fisher=False)

# 6) 邻居对称度（节点级 5）：in/out 邻居集合 Jaccard
def neighbor_symmetry_stats(Gd: nx.DiGraph) -> np.ndarray:
    if Gd.number_of_nodes() == 0:
        return np.zeros(5, dtype=float)
    s_list = []
    for v in Gd.nodes():
        outN = set(Gd.successors(v))
        inN  = set(Gd.predecessors(v))
        union = outN | inN
        if len(union) == 0:
            s_list.append(0.0)
        else:
            inter = len(outN & inN)
            s_list.append(inter / float(len(union)))
    return safe_stats_for_feature(np.array(s_list, dtype=float), fisher=False)

# ============================== 新增高级特征 ==============================
def importer_temporal_burstiness_stats(df: pd.DataFrame, bins: int = 8) -> np.ndarray:
    """
    对于每个目的地 i，把它在该时间窗内的事件按等宽时间 bins 划分，计算峰值区间占比 (max_bin / total)
    作为该 i 的 burstiness；然后对所有 i 做 safe_stats。
    这个特征检测在窗口内部是否有短时爆发，贸易数据常见。
    """
    if df is None or df.empty:
        return np.zeros(5, dtype=float)
    ts_min, ts_max = df["ts"].min(), df["ts"].max()
    if ts_max <= ts_min:
        # 单时间点，burstiness 等于 1 for nodes with events
        counts = df.groupby("i").size().astype(float).to_numpy()
        burst = np.ones_like(counts)
        return safe_stats_for_feature(burst, fisher=False)
    b_edges = np.linspace(ts_min, ts_max + 1e-6, bins + 1)
    burst_vals = []
    for i, group in df.groupby("i"):
        arr = group["ts"].to_numpy(dtype=float)
        if arr.size == 0:
            continue
        hist, _ = np.histogram(arr, bins=b_edges)
        m = hist.max()
        total = hist.sum()
        if total <= 0:
            burst_vals.append(0.0)
        else:
            burst_vals.append(float(m / total))
    return safe_stats_for_feature(np.array(burst_vals, dtype=float), fisher=False)

def exporter_temporal_burstiness_stats(df: pd.DataFrame, bins: int = 8) -> np.ndarray:
    """
    出口端（按 u 分组）时间突发性：
    在窗口内把时间轴等分为 bins 段，对每个 u 计算 (最大bin计数 / 总计数)，然后对所有 u 做 5统计量。
    """
    if df is None or df.empty:
        return np.zeros(5, dtype=float)
    ts_min, ts_max = df["ts"].min(), df["ts"].max()
    if ts_max <= ts_min:
        counts = df.groupby("u").size().astype(float).to_numpy()
        burst = np.ones_like(counts)
        return safe_stats_for_feature(burst, fisher=False)
    b_edges = np.linspace(ts_min, ts_max + 1e-6, bins + 1)
    burst_vals = []
    for u, group in df.groupby("u"):
        arr = group["ts"].to_numpy(dtype=float)
        if arr.size == 0:
            continue
        hist, _ = np.histogram(arr, bins=b_edges)
        m = hist.max()
        total = hist.sum()
        burst_vals.append(float(m / total) if total > 0 else 0.0)
    return safe_stats_for_feature(np.array(burst_vals, dtype=float), fisher=False)

def bilateral_repeat_weight_ratio_stats(df: pd.DataFrame) -> np.ndarray:
    """
    接收端（in）版本：
    对每个目的地 i，计算来自重复边 (u,i)（出现 >=2 次）的权重占比：
      ratio_i = sum_{(u,i): cnt>=2} cnt(u,i) / sum_{(u,i)} cnt(u,i)
    然后对 ratio_i 做 safe_stats。
    """
    if df is None or df.empty:
        return np.zeros(5, dtype=float)
    cnt_series = df.groupby(["u", "i"]).size().astype(float)
    total_by_i = cnt_series.groupby(level=1).sum().astype(float)
    repeat_mask = cnt_series >= 2.0
    repeat_series = cnt_series[repeat_mask].groupby(level=1).sum().astype(float)  # index level1 is i
    # ensure alignment
    idx = total_by_i.index
    repeat_vals = repeat_series.reindex(idx, fill_value=0.0).to_numpy(dtype=float)
    total_vals = total_by_i.reindex(idx, fill_value=0.0).to_numpy(dtype=float)
    ratio = np.zeros_like(total_vals, dtype=float)
    mask = total_vals > 0
    ratio[mask] = repeat_vals[mask] / total_vals[mask]
    return safe_stats_for_feature(ratio, fisher=False)

def bilateral_repeat_weight_ratio_stats_out(df: pd.DataFrame) -> np.ndarray:
    """
    发出端（out）版本：
    对每个源 u，计算投向重复边 (u,i)（出现 >=2 次）的权重占比：
      ratio_u = sum_{(u,i): cnt>=2} cnt(u,i) / sum_{(u,i)} cnt(u,i)
    然后对 ratio_u 做 safe_stats。
    """
    if df is None or df.empty:
        return np.zeros(5, dtype=float)
    cnt_series = df.groupby(["u", "i"]).size().astype(float)
    total_by_u = cnt_series.groupby(level=0).sum().astype(float)
    repeat_mask = cnt_series >= 2.0
    repeat_series = cnt_series[repeat_mask].groupby(level=0).sum().astype(float)  # index level0 is u
    idx = total_by_u.index
    repeat_vals = repeat_series.reindex(idx, fill_value=0.0).to_numpy(dtype=float)
    total_vals = total_by_u.reindex(idx, fill_value=0.0).to_numpy(dtype=float)
    ratio = np.zeros_like(total_vals, dtype=float)
    mask = total_vals > 0
    ratio[mask] = repeat_vals[mask] / total_vals[mask]
    return safe_stats_for_feature(ratio, fisher=False)

def exporter_importer_imbalance_stats(Gd: nx.DiGraph) -> np.ndarray:
    """
    对每个节点计算 (out_w - in_w) / (out_w + in_w + eps)，衡量节点的净出口/净进口角色。
    返回所有节点该不平衡值的 5 统计量。
    """
    if Gd.number_of_nodes() == 0:
        return np.zeros(5, dtype=float)
    eps = 1e-12
    inw = {n: w for n, w in Gd.in_degree(weight="weight")}
    outw = {n: w for n, w in Gd.out_degree(weight="weight")}
    vals = []
    for n in Gd.nodes():
        iw = float(inw.get(n, 0.0))
        ow = float(outw.get(n, 0.0))
        vals.append((ow - iw) / (ow + iw + eps))
    return safe_stats_for_feature(np.array(vals, dtype=float), fisher=False)


# ============================== 扩展签名（35 维） ==============================
def full_signature(df: pd.DataFrame) -> np.ndarray:
    """
    生成一个时间窗（df）对应的 35 维签名：
    [加权入度5, 加权出度5, 边重复分布5, 节点burstness5, 边burstness5, 加权NetSimile(10)]
    """
    Gd = build_graph_directed(df)

    parts = []
    parts.append(weighted_indegree_stats(Gd))             # 0-4
    parts.append(weighted_outdegree_stats(Gd))            # 5-9
    parts.append(edge_reuse_features(df))                 # 10-14 (加回边重复分布)
    parts.append(node_burstiness_stats(df))               # 15-19
    parts.append(edge_burstiness_stats(df))               # 20-24
    parts.append(netsimile_signature(Gd))                 # 25-34 (包含加权邻居特征)
    sig = np.concatenate(parts).astype(float)
    return sig

def analyze_feature_contributions(dataset_name: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """分析每个数据集的特征贡献"""
    try:
        df = load_dataframe(cfg["path"])
        train_df = df[(df["ts"] >= cfg["train_ts_start"]) & (df["ts"] <= cfg["train_ts_end"])]
        test_df  = df[(df["ts"] >= cfg["test_ts_start"])  & (df["ts"] <= cfg["test_ts_end"])]

        # 计算35维签名
        sig_train = full_signature(train_df)
        sig_test  = full_signature(test_df)
        
        # 计算Canberra距离和每个维度的贡献
        canberra_dist, contributions = canberra_distance_with_contributions(sig_train, sig_test)
        
        # 定义特征名称和对应的维度范围（35维）
        feature_names = [
            "weighted_indegree",      # 0-4
            "weighted_outdegree",     # 5-9
            "edge_reuse",            # 10-14
            "burstness_node",        # 15-19
            "burstness_edge",        # 20-24
            "netsimile_avg_neigh_weighted_indeg", # 25-29
            "netsimile_avg_neigh_weighted_outdeg"  # 30-34
        ]
        
        feature_ranges = [
            (0, 5), (5, 10), (10, 15), (15, 20), (20, 25), (25, 30), (30, 35)
        ]
        
        stat_names = ["median", "mean", "std", "skew", "kurtosis"]

        # 计算每个特征的总贡献
        feature_contributions = {}
        for name, (start, end) in zip(feature_names, feature_ranges):
            if end <= contributions.size:
                contrib_slice = contributions[start:end]
                feature_contributions[name] = {stat_names[i]: float(contrib_slice[i]) for i in range(len(contrib_slice))}
            else:
                feature_contributions[name] = {stat_name: 0.0 for stat_name in stat_names}
        
        # 计算每个特征的相对贡献百分比
        total_canberra = sum(contributions)
        feature_percentages = {}
        for name, (start, end) in zip(feature_names, feature_ranges):
            if end <= contributions.size:
                contrib_slice = contributions[start:end]
                feature_percentages[name] = {
                    stat_names[i]: (contrib_slice[i] / total_canberra * 100) if total_canberra > 0 else 0 
                    for i in range(len(contrib_slice))
                }
            else:
                feature_percentages[name] = {stat_name: 0.0 for stat_name in stat_names}
            
        return {
            "dataset": dataset_name,
            "dataset_type": cfg["type"],
            "canberra_distance": canberra_dist,
            "feature_contributions": feature_contributions,
            "feature_percentages": feature_percentages,
            "total_contribution": total_canberra
        }
    except Exception as e:
        return {
            "dataset": dataset_name,
            "dataset_type": cfg.get("type", "unknown"),
            "canberra_distance": float("inf"),
            "error": str(e)
        }

def analyze_all_feature_contributions(datasets: Dict[str, Dict[str, Any]]) -> None:
    """分析所有数据集的特征对Canberra距离的具体数值贡献"""
    print("=" * 120)
    print("分析每个数据集的特征对Canberra距离的具体数值贡献（35维）")
    print("=" * 120)
    
    for name, cfg in datasets.items():
        print(f"\n=== {name} ({cfg['type']}) ===")
        result = analyze_feature_contributions(name, cfg)
        
        if "error" in result:
            print(f"错误: {result['error']}")
            continue
            
        print(f"Canberra距离: {result['canberra_distance']:.6f}")
        
        print("\n特征贡献 (具体数值):")
        for feature, stats_contrib in result['feature_contributions'].items():
            total_feature_contrib = sum(stats_contrib.values())
            print(f"  {feature:35s}: {total_feature_contrib:8.4f}")
            for stat, contrib in stats_contrib.items():
                print(f"    - {stat:10s}: {contrib:8.4f}")
        
        # 找出贡献最大的前10个特征
        feature_total_contributions = {name: sum(p.values()) for name, p in result['feature_contributions'].items()}
        sorted_features = sorted(feature_total_contributions.items(), key=lambda x: x[1], reverse=True)

        print(f"\n贡献最大的前10个特征:")
        for i, (feature, contrib) in enumerate(sorted_features[:10], 1):
            print(f"  {i:2d}. {feature:35s}: {contrib:8.4f}")
        print("-" * 120)

# ============================== Canberra：使用 25 维签名 ==============================
def calculate_dataset_metrics(dataset_name: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
    try:
        df = load_dataframe(cfg["path"])
        train_df = df[(df["ts"] >= cfg["train_ts_start"]) & (df["ts"] <= cfg["train_ts_end"])]
        test_df  = df[(df["ts"] >= cfg["test_ts_start"])  & (df["ts"] <= cfg["test_ts_end"])]
        
        train_nodes = len(set(train_df['u']) | set(train_df['i']))
        train_edges = len(train_df)
        test_nodes = len(set(test_df['u']) | set(test_df['i']))
        test_edges = len(test_df)

        print(f"\n=== {dataset_name} 完整数据集信息 ===")
        print(f"训练集: {train_nodes} 个节点, {train_edges} 条边")
        print(f"测试集: {test_nodes} 个节点, {test_edges} 条边")

        # 25 维签名
        sig_train = full_signature(train_df)
        sig_test  = full_signature(test_df)
        canberra_dist = canberra_distance(sig_train, sig_test)

        result = {
            "dataset": dataset_name, "dataset_type": cfg["type"],
            "train_nodes": train_nodes, "train_edges": train_edges,
            "test_nodes": test_nodes,   "test_edges": test_edges,
            "NetSimile_Canberra": canberra_dist,
        }

        print(f"扩展签名维度: {sig_train.size}")
        print(f"Canberra 距离: {canberra_dist:.6f}")
        print("=" * 50)
        return result
    except Exception as e:
        print(f"计算 {dataset_name} 时出错: {e}")
        return {
            "dataset": dataset_name, "dataset_type": cfg.get("type","unknown"),
            "train_nodes": 0, "train_edges": 0, "test_nodes": 0, "test_edges": 0,
            "NetSimile_Canberra": float("inf"),
            "error": str(e)
        }

# ============================== 主流程 ==============================
def analyze_all_datasets_canberra(datasets: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    print("=" * 80)
    print("计算所有数据集：Canberra Distance (共 35 维)")
    print("=" * 80)
    rows = []
    for name, cfg in datasets.items():
        rows.append(calculate_dataset_metrics(name, cfg))
    return pd.DataFrame(rows)

# ============================== 入口 ==============================
if __name__ == "__main__":
    print("=" * 80)
    print("计算：扩展 Canberra（35 维）")
    print("=" * 80)

    _register_lastfm_from_processed_csv(DATASETS)
    if "lastfm" not in DATASETS:
        print("提示: 未找到 processed_data/lastfm/ml_lastfm.csv，已跳过 lastfm（可先运行 preprocess_data/preprocess_data.py --dataset_name lastfm）")
    
    # 首先分析每个数据集的特征贡献
    analyze_all_feature_contributions(DATASETS)
    
    # 然后运行 Canberra 距离分析
    canberra_df = analyze_all_datasets_canberra(DATASETS)

    print("\n" + "=" * 80)
    print("完整数据集 Canberra 距离分析结果")
    print("=" * 80)
    print(canberra_df.to_string(index=False))

    os.makedirs("new", exist_ok=True)
    canberra_df.to_csv("new/canberra_distance_analysis.csv", index=False)
    print("\n已保存：new/canberra_distance_analysis.csv")

    if not canberra_df[canberra_df["error"].isna()].empty:
        canberra_summary_df = canberra_df[canberra_df["error"].isna()][[
            "dataset","dataset_type","train_nodes","train_edges","test_nodes","test_edges",
            "NetSimile_Canberra"
        ]].copy().sort_values("NetSimile_Canberra", ascending=False)
        canberra_summary_df.to_csv("new/canberra_distance_summary.csv", index=False)
        print("已保存：new/canberra_distance_summary.csv")
    
    print("\n✅ 分析完成")
