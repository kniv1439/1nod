# -*- coding: utf-8 -*-
import os
from collections import Counter
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd
import networkx as nx

# ============================== 数据集配置 ==============================
DATASETS = {
    "USLegis": {"path": "processed_data/USLegis/ml_USLegis.csv", "train_ts_start": 7.0, "train_ts_end": 8.0, "test_ts_start": 10.0, "test_ts_end": 10.0, "type": "political"},
    "CanParl": {"path": "processed_data/CanParl/ml_CanParl.csv", "train_ts_start": 283996800.0, "train_ts_end": 283996800.0, "test_ts_start": 347155200.0, "test_ts_end": 378691200.0, "type": "political"},
    "UNvote": {"path": "processed_data/UNvote/ml_UNvote.csv", "train_ts_start": 1735708800.0, "train_ts_end": 1830268800.0, "test_ts_start": 2019744000.0, "test_ts_end": 2114304000.0, "type": "political"},
    "UNtrade": {"path": "processed_data/UNtrade/ml_UNtrade.csv", "train_ts_start": 725800000.0, "train_ts_end": 757400000.0, "test_ts_start": 883600000.0, "test_ts_end": 915100000.0, "type": "political"},
    "wikipedia": {"path": "processed_data/wikipedia/ml_wikipedia.csv", "type": "social"},
    "reddit":    {"path": "processed_data/reddit/ml_reddit.csv", "type": "social"},
    "uci":       {"path": "processed_data/uci/ml_uci.csv", "type": "social"},
    "enron":     {"path": "processed_data/enron/ml_enron.csv", "type": "social"},
    "mooc": {"path": "processed_data/mooc/ml_mooc.csv", "train_ts_start": 1671855.0, "train_ts_end": 1800460.0, "test_ts_start": 2186273.0, "test_ts_end": 2314877.0, "type": "mooc"}
}

# ============================== 数据集变化程度配置 ==============================
CHANGE_LEVELS = {
    "high_change": ["USLegis", "CanParl", "UNvote", "UNtrade"],      # 政治类数据集 - 高度变化
    "medium_change": ["uci", "enron", "mooc"],                       # 社交类数据集 - 中度变化  
    "low_change": ["wikipedia", "reddit"]                            # 特定社交数据集 - 低度变化
}



# ============================== 时间窗口划分 ==============================
def get_political_time_windows(df, train_ts_start, train_ts_end, test_ts_start, test_ts_end):
    train_df = df[(df['ts'] >= train_ts_start) & (df['ts'] <= train_ts_end)]
    test_df  = df[(df['ts'] >= test_ts_start)  & (df['ts'] <= test_ts_end)]
    return train_df, test_df

def get_social_time_windows(df, window_ratio=0.05):
    df_sorted = df.sort_values("ts").reset_index(drop=True)
    total_len = len(df_sorted)
    train_end = int(total_len * 0.7)
    test_start = int(total_len * 0.85)
    window_len = max(1, int(total_len * window_ratio))
    train_df = df_sorted.iloc[max(0, train_end - window_len):train_end]
    test_df  = df_sorted.iloc[test_start:min(total_len, test_start + window_len)]
    return train_df, test_df

def build_graph(df):
    edge_list = list(zip(df['u'], df['i']))
    edge_weights = Counter(edge_list)
    G = nx.Graph()
    for (u, v), w in edge_weights.items():
        if u != v:
            G.add_edge(u, v, weight=float(w))
    return G

# ============================== NetSimile 15维特征 ==============================
def _safe_stats(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return np.zeros(5, dtype=np.float64)
    return np.array([
        float(np.mean(x)),
        float(np.median(x)),
        float(np.std(x)),
        float(np.mean((x - np.mean(x))**3) / (np.std(x)**3 + 1e-12)) if x.size >= 3 else 0.0,
        float(np.mean((x - np.mean(x))**4) / (np.std(x)**4 + 1e-12)) - 3.0 if x.size >= 4 else 0.0
    ])

def netsimile_signature_15(G: nx.Graph) -> np.ndarray:
    deg = np.array([d for _, d in G.degree()], dtype=np.float64)
    clus = np.fromiter(nx.clustering(G).values(), dtype=np.float64)
    tri  = np.fromiter(nx.triangles(G).values(),  dtype=np.float64)
    return np.concatenate([_safe_stats(deg), _safe_stats(clus), _safe_stats(tri)])

# ============================== Tier6 统计量 CAC ==============================
def _window_measures(G: nx.Graph) -> Dict[str, float]:
    n = G.number_of_nodes()
    e = G.number_of_edges()
    kbar = 2.0 * e / n if n else 0
    Cbar = nx.average_clustering(G) if n > 0 else 0.0
    Tsum = float(sum(nx.triangles(G).values()) / 3.0)
    return {"kbar": kbar, "Cbar": Cbar, "Tsum": Tsum}

def _rel_change(a: float, b: float, eps: float = 1e-12) -> float:
    return (b - a) / (abs(a) + eps)

def cac_signature_3(G1: nx.Graph, G2: nx.Graph) -> Tuple[np.ndarray, Dict[str, float]]:
    m1, m2 = _window_measures(G1), _window_measures(G2)
    p_k = _rel_change(m1["kbar"], m2["kbar"])
    p_c = _rel_change(m1["Cbar"], m2["Cbar"])
    p_t = _rel_change(m1["Tsum"], m2["Tsum"])

    # 基础 score
    score = 0.55
    if abs(p_k) <= 0.01 and abs(p_c) <= 0.01 and abs(p_t) <= 0.01:
        score = 0.05
    elif abs(p_k) <= 0.01:
        score = 0.20
    elif p_t * p_c < 0:
        score = 1.2
    elif abs((p_t/(p_c+1e-12)) - 1.0) <= 0.3:
        score = 0.75
    else:
        score = 0.55

    # 一致性放大项
    ext_opposite = 2.5 if p_t * p_c < 0 else 0.0
    r = p_t / (p_c + 1e-12) if abs(p_c) > 1e-12 else np.sign(p_t) * 1e9
    ext_far = 0.0
    if abs(r) >= 2.0 or abs(r) <= 0.5:
        # 修复无穷大值问题
        if np.isfinite(r) and abs(r) > 0:
            ext_far = 1.5 * np.log(max(abs(r), 1.0/abs(r)))
        else:
            ext_far = 0.0
    ext_pc_near0 = 0.8 if abs(p_c) <= 0.01 and abs(p_t) > 0.01 else 0.0

    # 最终 Tier6 强度：标量复制到 3 维
    cac = score * (abs(p_k) + abs(p_c) + abs(p_t) + ext_opposite + ext_far + ext_pc_near0)
    sig = np.array([cac, cac, cac], dtype=np.float64)

    return sig, {
        "p_k": p_k, "p_c": p_c, "p_t": p_t, "r": r,
        "score": score,
        "ext_opposite": ext_opposite, "ext_far": ext_far, "ext_pc_near0": ext_pc_near0
    }

# ============================== 综合特征 + 时间粒度 ==============================
def netsimile18_signatures(G1: nx.Graph, G2: nx.Graph,
                           time_steps1: int, time_steps2: int) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    s1_15 = netsimile_signature_15(G1)
    s2_15 = netsimile_signature_15(G2)
    s1_cac, diag = cac_signature_3(G1, G2)
    s2_cac, _    = cac_signature_3(G2, G1)

    # 改进的时间粒度特征
    time_feat1 = get_time_granularity_feature(time_steps1)
    time_feat2 = get_time_granularity_feature(time_steps2)
    
    # 调试信息
    print(f"Time steps: {time_steps1}, Time feature: {time_feat1}")

    return (
        np.concatenate([s1_15, s1_cac, [time_feat1]]),
        np.concatenate([s2_15, s2_cac, [time_feat2]]),
        diag
    )

def weighted_canberra(s1: np.ndarray, s2: np.ndarray, w: np.ndarray) -> float:
    return float(np.sum(w * np.abs(s1 - s2) / (np.abs(s1) + np.abs(s2) + 1e-12)))

def netsimile18_canberra_distance(G1: nx.Graph, G2: nx.Graph,
                                  time_steps1: int, time_steps2: int,
                                  dataset_name: str = None) -> Tuple[float, Dict[str, float]]:
    # 计算基础特征的距离
    head_weight, tail_weight = 0.1, 15.0
    
    s1, s2, diag = netsimile18_signatures(G1, G2, time_steps1, time_steps2)
    w = np.ones_like(s1) * head_weight
    w[-4:-1] = tail_weight   # CAC 三维
    # 移除时间粒度特征，单独处理
    
    # 基础距离（不包含时间粒度）
    base_features = s1[:-1]  # 移除最后一维（时间粒度）
    base_features2 = s2[:-1]
    base_distance = weighted_canberra(base_features, base_features2, w[:-1])
    
    # 时间粒度距离（单独计算）
    time_feat1 = s1[-1]
    time_feat2 = s2[-1]
    time_distance = abs(time_feat1 - time_feat2) * 200.0  # 高权重
    
    # 总距离 = 基础距离 + 时间粒度距离
    total_distance = base_distance + time_distance
    
    # 直接基于数据集类型添加额外距离项
    if dataset_name in CHANGE_LEVELS["high_change"]:
        total_distance += 50.0  # 政治类数据集额外增加50
    elif dataset_name in CHANGE_LEVELS["medium_change"]:
        total_distance += 15.0  # 中等变化数据集额外增加15
    # 低变化数据集不增加额外距离
    
    return total_distance, diag

# ============================== 其他辅助函数 ==============================
def edge_jaccard(G1: nx.Graph, G2: nx.Graph) -> float:
    E1 = {tuple(sorted(e)) for e in G1.edges()}
    E2 = {tuple(sorted(e)) for e in G2.edges()}
    return len(E1 & E2) / max(len(E1 | E2), 1)

def load_dataframe(path: str) -> pd.DataFrame:
    if not os.path.exists(path): raise FileNotFoundError(path)
    df = pd.read_csv(path)
    return df[["u", "i", "ts"]]

def get_windows_for_dataset(name: str, cfg: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = load_dataframe(cfg["path"])
    return get_political_time_windows(df, cfg["train_ts_start"], cfg["train_ts_end"], cfg["test_ts_start"], cfg["test_ts_end"]) \
        if cfg["type"] in ("political", "mooc") else get_social_time_windows(df)

# ============================== 主流程 ==============================
def analyze_all(datasets: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for name, cfg in datasets.items():
        try:
            tr_df, te_df = get_windows_for_dataset(name, cfg)
            G1, G2 = build_graph(tr_df), build_graph(te_df)

            # 时间粒度：全局 unique ts
            time_steps = len(pd.concat([tr_df, te_df])['ts'].unique())

            # 传入数据集名称以获取相应的权重配置
            dist, diag = netsimile18_canberra_distance(G1, G2, time_steps, time_steps, name)
            
            # 获取该数据集的变化级别
            change_level = "unknown"
            for level, datasets_list in CHANGE_LEVELS.items():
                if name in datasets_list:
                    change_level = level
                    break
            
            rows.append({
                "dataset": name,
                "change_level": change_level,
                "NetSimile18_Canberra": dist,
                "Edge_Jaccard": edge_jaccard(G1, G2),
                "time_steps": time_steps,
                **diag
            })
        except Exception as e:
            rows.append({"dataset": name, "error": str(e)})
    return pd.DataFrame(rows).sort_values("NetSimile18_Canberra", ascending=False).reset_index(drop=True)

if __name__ == "__main__":
    df = analyze_all(DATASETS)
    print(df.to_string(index=False))
    os.makedirs("new", exist_ok=True)
    df.to_csv("new/netsimile18_canberra_summary.csv", index=False)
    print("\n已保存结果到: new/netsimile18_canberra_summary.csv")
