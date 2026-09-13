# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import networkx as nx
from collections import Counter
from typing import Tuple, Dict, Any
from scipy.stats import ks_2samp
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
import matplotlib.pyplot as plt
import seaborn as sns

# 尝试不同的导入方式
try:
    from community import community_louvain
except ImportError:
    try:
        import community.community_louvain as community_louvain
    except ImportError:
        try:
            from community import best_partition as community_louvain_best_partition
            # 创建一个模拟的community_louvain对象
            class MockCommunityLouvain:
                @staticmethod
                def best_partition(G):
                    return community_louvain_best_partition(G)
            community_louvain = MockCommunityLouvain()
        except ImportError:
            print("警告: 无法导入community_louvain，请安装: pip install python-louvain")
            # 创建一个简单的替代实现
            class SimpleCommunityLouvain:
                @staticmethod
                def best_partition(G):
                    # 简单的社区检测：每个节点一个社区
                    return {node: i for i, node in enumerate(G.nodes())}
            community_louvain = SimpleCommunityLouvain()

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

def load_dataframe(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df[["u", "i", "ts"]]

def get_windows_for_dataset(name: str, cfg: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = load_dataframe(cfg["path"])
    if cfg["type"] in ("political", "mooc"):
        return get_political_time_windows(df, cfg["train_ts_start"], cfg["train_ts_end"], cfg["test_ts_start"], cfg["test_ts_end"])
    else:
        return get_social_time_windows(df)

# ============================== 五类特征（移除星形motif） ==============================
def safe_stats(arr: np.ndarray) -> np.ndarray:
    if len(arr) == 0:
        return np.zeros(3)
    return np.array([np.mean(arr), np.median(arr), np.std(arr)])

# def star_motif_freq(G: nx.Graph) -> np.ndarray:
#     # star motif: 节点度数高的星形结构，统计每个节点的度数作为 proxy
#     deg = np.array([d for _, d in G.degree()], dtype=float)
#     return safe_stats(deg)

def degree_distribution_stats(G1, G2) -> np.ndarray:
    d1 = [d for _, d in G1.degree()]
    d2 = [d for _, d in G2.degree()]
    if len(d1) == 0 or len(d2) == 0:
        ks_stat = 0.0
    else:
        ks_stat, _ = ks_2samp(d1, d2)
    return np.array([ks_stat, ks_stat, ks_stat])

def community_similarity(G1, G2) -> np.ndarray:
    if G1.number_of_nodes() == 0 or G2.number_of_nodes() == 0:
        return np.zeros(3)
    comm1 = community_louvain.best_partition(G1) if G1.number_of_edges() > 0 else {n: 0 for n in G1.nodes()}
    comm2 = community_louvain.best_partition(G2) if G2.number_of_edges() > 0 else {n: 0 for n in G2.nodes()}
    nodes = list(set(G1.nodes()) | set(G2.nodes()))
    labels1 = [comm1.get(n, -1) for n in nodes]
    labels2 = [comm2.get(n, -1) for n in nodes]
    nmi = normalized_mutual_info_score(labels1, labels2)
    ari = adjusted_rand_score(labels1, labels2)
    mod_diff = abs(nx.algorithms.community.quality.modularity(G1, [set([n for n, c in comm1.items() if c == cid]) for cid in set(comm1.values())]) -
                   nx.algorithms.community.quality.modularity(G2, [set([n for n, c in comm2.items() if c == cid]) for cid in set(comm2.values())]))
    return np.array([nmi, ari, mod_diff])

def netsimile18_signature(G1: nx.Graph, G2: nx.Graph) -> np.ndarray:
    # 获取两个图的所有节点
    all_nodes = list(set(G1.nodes()) | set(G2.nodes()))
    
    # 计算三角形数量，对于不存在的节点设为0
    tri1_dict = nx.triangles(G1)
    tri2_dict = nx.triangles(G2)
    tri1 = np.array([tri1_dict.get(node, 0) for node in all_nodes], dtype=float)
    tri2 = np.array([tri2_dict.get(node, 0) for node in all_nodes], dtype=float)
    
    # 计算聚类系数，对于不存在的节点设为0
    clus1_dict = nx.clustering(G1)
    clus2_dict = nx.clustering(G2)
    clus1 = np.array([clus1_dict.get(node, 0) for node in all_nodes], dtype=float)
    clus2 = np.array([clus2_dict.get(node, 0) for node in all_nodes], dtype=float)
    
    # 计算度数，对于不存在的节点设为0
    deg1_dict = dict(G1.degree())
    deg2_dict = dict(G2.degree())
    deg1 = np.array([deg1_dict.get(node, 0) for node in all_nodes], dtype=float)
    deg2 = np.array([deg2_dict.get(node, 0) for node in all_nodes], dtype=float)



    f1 = safe_stats(tri1 - tri2)
    f2 = safe_stats(clus1 - clus2)
    f3 = safe_stats(deg1 - deg2)
    f4 = degree_distribution_stats(G1, G2)
    f5 = community_similarity(G1, G2)
    # f6 = safe_stats(star_motif_freq(G1) - star_motif_freq(G2))  # 移除星形motif统计

    return np.concatenate([f1, f2, f3, f4, f5])  # 移除f6

# ============================== Canberra 距离 ==============================
def canberra_distance(x: np.ndarray, y: np.ndarray) -> float:
    # 定义权重向量：15个特征维度的权重（每个特征3个统计量）
    # f1: [均值, 中位数, 标准差] - 增大中位数权重
    # f2: [均值, 中位数, 标准差] - 增大中位数权重  
    # f3: [均值, 中位数, 标准差] - 增大所有统计量权重
    # f4: [KS统计量, KS统计量, KS统计量]
    # f5: [NMI, ARI, 模块度差异]
    weights = np.array([
        1.0, 3.0, 1.0,  # f1: 三角形差异 [均值, 中位数(权重3), 标准差]
        1.0, 3.0, 1.0,  # f2: 聚类系数差异 [均值, 中位数(权重3), 标准差]
        2.0, 2.0, 2.0,  # f3: 度数差异 [均值(权重2), 中位数(权重2), 标准差(权重2)]
        1.0, 1.0, 1.0,  # f4: 度分布差异 [KS统计量, KS统计量, KS统计量]
        1.0, 1.0, 1.0   # f5: 社区相似性 [NMI, ARI, 模块度差异]
    ])
    
    # 加权Canberra距离
    return float(np.sum(weights * np.abs(x - y) / (np.abs(x) + np.abs(y) + 1e-12)))

# ============================== 图结构变化指标分析 ==============================
def calculate_graph_change_metrics(G1: nx.Graph, G2: nx.Graph) -> Dict[str, float]:
    """
    计算两个图之间的结构变化指标
    """
    # 1. 三角形变化
    tri1 = sum(nx.triangles(G1).values()) / 3.0  # 三角形总数
    tri2 = sum(nx.triangles(G2).values()) / 3.0
    triangle_change = (tri2 - tri1) / max(tri1, 1e-12)  # 相对变化
    
    # 2. 聚类系数变化
    clust1 = nx.average_clustering(G1) if G1.number_of_nodes() > 0 else 0.0
    clust2 = nx.average_clustering(G2) if G2.number_of_nodes() > 0 else 0.0
    clustering_change = (clust2 - clust1) / max(clust1, 1e-12)  # 相对变化
    
    # 3. 节点平均度数变化
    avg_deg1 = 2.0 * G1.number_of_edges() / max(G1.number_of_nodes(), 1)
    avg_deg2 = 2.0 * G2.number_of_edges() / max(G2.number_of_nodes(), 1)
    degree_change = (avg_deg2 - avg_deg1) / max(avg_deg1, 1e-12)  # 相对变化
    
    # 4. 绝对变化值
    triangle_abs_change = tri2 - tri1
    clustering_abs_change = clust2 - clust1
    degree_abs_change = avg_deg2 - avg_deg1
    
    return {
        "triangle_relative_change": triangle_change,
        "clustering_relative_change": clustering_change,
        "degree_relative_change": degree_change,
        "triangle_absolute_change": triangle_abs_change,
        "clustering_absolute_change": clustering_abs_change,
        "degree_absolute_change": degree_abs_change,
        "train_triangles": tri1,
        "test_triangles": tri2,
        "train_clustering": clust1,
        "test_clustering": clust2,
        "train_avg_degree": avg_deg1,
        "test_avg_degree": avg_deg2
    }

# ============================== 主流程 ==============================
def analyze_all(datasets: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for name, cfg in datasets.items():
        try:
            tr_df, te_df = get_windows_for_dataset(name, cfg)
            G1, G2 = build_graph(tr_df), build_graph(te_df)

            # 计算图结构变化指标
            change_metrics = calculate_graph_change_metrics(G1, G2)
            
            # 原有的NetSimile分析
            s1 = netsimile18_signature(G1, G1)   # baseline: 自身
            s2 = netsimile18_signature(G1, G2)
            dist = canberra_distance(s1, s2)
            
            # 合并所有指标
            row_data = {
                "dataset": name,
                "dataset_type": cfg["type"],
                "NetSimile_Canberra": dist,
                "error": None,
                **change_metrics
            }
            rows.append(row_data)
            
        except Exception as e:
            rows.append({
                "dataset": name, 
                "dataset_type": cfg.get("type", "unknown"),
                "NetSimile_Canberra": float('inf'),  # 出错时设为无穷大
                "error": str(e)
            })
    
    df = pd.DataFrame(rows)
    # 只对没有错误的数据集进行排序
    valid_df = df[df['error'].isna()].sort_values("NetSimile_Canberra", ascending=False)
    error_df = df[df['error'].notna()]
    return pd.concat([valid_df, error_df]).reset_index(drop=True)

# ============================== 可视化函数 ==============================
def create_change_histograms(df: pd.DataFrame, output_dir: str = "new"):
    """
    为图结构变化指标创建直方图
    """
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 过滤掉有错误的数据集
    valid_df = df[df['error'].isna()].copy()
    
    if valid_df.empty:
        print("没有有效数据可以绘制直方图")
        return
    
    # 创建子图
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('九个数据集的图结构变化指标分布', fontsize=16, fontweight='bold')
    
    # 1. 三角形相对变化直方图
    ax1 = axes[0, 0]
    colors = ['#FF6B6B' if t == 'political' else '#4ECDC4' if t == 'social' else '#45B7D1' 
              for t in valid_df['dataset_type']]
    bars1 = ax1.bar(range(len(valid_df)), valid_df['triangle_relative_change'], color=colors, alpha=0.7)
    ax1.set_title('三角形相对变化 (%)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('数据集')
    ax1.set_ylabel('相对变化')
    ax1.set_xticks(range(len(valid_df)))
    ax1.set_xticklabels(valid_df['dataset'], rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)
    
    # 添加数值标签
    for i, (bar, val) in enumerate(zip(bars1, valid_df['triangle_relative_change'])):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f'{val:.2f}', ha='center', va='bottom', fontsize=9)
    
    # 2. 聚类系数相对变化直方图
    ax2 = axes[0, 1]
    bars2 = ax2.bar(range(len(valid_df)), valid_df['clustering_relative_change'], color=colors, alpha=0.7)
    ax2.set_title('聚类系数相对变化 (%)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('数据集')
    ax2.set_ylabel('相对变化')
    ax2.set_xticks(range(len(valid_df)))
    ax2.set_xticklabels(valid_df['dataset'], rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)
    
    # 添加数值标签
    for i, (bar, val) in enumerate(zip(bars2, valid_df['clustering_relative_change'])):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    # 3. 平均度数相对变化直方图
    ax3 = axes[1, 0]
    bars3 = ax3.bar(range(len(valid_df)), valid_df['degree_relative_change'], color=colors, alpha=0.7)
    ax3.set_title('平均度数相对变化 (%)', fontsize=12, fontweight='bold')
    ax3.set_xlabel('数据集')
    ax3.set_ylabel('相对变化')
    ax3.set_xticks(range(len(valid_df)))
    ax3.set_xticklabels(valid_df['dataset'], rotation=45, ha='right')
    ax3.grid(True, alpha=0.3)
    
    # 添加数值标签
    for i, (bar, val) in enumerate(zip(bars3, valid_df['degree_relative_change'])):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    # 4. 综合变化指标散点图
    ax4 = axes[1, 1]
    scatter = ax4.scatter(valid_df['triangle_relative_change'], 
                         valid_df['clustering_relative_change'],
                         c=[1 if t == 'political' else 2 if t == 'social' else 3 for t in valid_df['dataset_type']],
                         s=valid_df['degree_relative_change'].abs() * 100 + 50,
                         alpha=0.7, cmap='viridis')
    
    # 添加数据集标签
    for i, row in valid_df.iterrows():
        ax4.annotate(row['dataset'], 
                    (row['triangle_relative_change'], row['clustering_relative_change']),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    ax4.set_title('三角形变化 vs 聚类系数变化\n(气泡大小表示度数变化)', fontsize=12, fontweight='bold')
    ax4.set_xlabel('三角形相对变化')
    ax4.set_ylabel('聚类系数相对变化')
    ax4.grid(True, alpha=0.3)
    
    # 添加图例
    legend_elements = [
        plt.Rectangle((0,0),1,1, facecolor='#FF6B6B', alpha=0.7, label='政治类'),
        plt.Rectangle((0,0),1,1, facecolor='#4ECDC4', alpha=0.7, label='社交类'),
        plt.Rectangle((0,0),1,1, facecolor='#45B7D1', alpha=0.7, label='教育类')
    ]
    ax1.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    
    # 保存图片
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f'{output_dir}/graph_change_histograms.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/graph_change_histograms.pdf', bbox_inches='tight')
    print(f"已保存直方图到: {output_dir}/graph_change_histograms.png 和 .pdf")
    
    # 显示图片
    plt.show()

def create_detailed_comparison_plots(df: pd.DataFrame, output_dir: str = "new"):
    """
    创建详细的对比图
    """
    valid_df = df[df['error'].isna()].copy()
    
    if valid_df.empty:
        return
    
    # 创建训练vs测试的对比图
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('训练集 vs 测试集 图结构指标对比', fontsize=16, fontweight='bold')
    
    # 1. 三角形数量对比
    ax1 = axes[0]
    x = np.arange(len(valid_df))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, valid_df['train_triangles'], width, label='训练集', alpha=0.8, color='#FF9999')
    bars2 = ax1.bar(x + width/2, valid_df['test_triangles'], width, label='测试集', alpha=0.8, color='#66B2FF')
    
    ax1.set_title('三角形数量对比', fontweight='bold')
    ax1.set_xlabel('数据集')
    ax1.set_ylabel('三角形数量')
    ax1.set_xticks(x)
    ax1.set_xticklabels(valid_df['dataset'], rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 聚类系数对比
    ax2 = axes[1]
    bars3 = ax2.bar(x - width/2, valid_df['train_clustering'], width, label='训练集', alpha=0.8, color='#FF9999')
    bars4 = ax2.bar(x + width/2, valid_df['test_clustering'], width, label='测试集', alpha=0.8, color='#66B2FF')
    
    ax2.set_title('聚类系数对比', fontweight='bold')
    ax2.set_xlabel('数据集')
    ax2.set_ylabel('聚类系数')
    ax2.set_xticks(x)
    ax2.set_xticklabels(valid_df['dataset'], rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 平均度数对比
    ax3 = axes[2]
    bars5 = ax3.bar(x - width/2, valid_df['train_avg_degree'], width, label='训练集', alpha=0.8, color='#FF9999')
    bars6 = ax3.bar(x + width/2, valid_df['test_avg_degree'], width, label='测试集', alpha=0.8, color='#66B2FF')
    
    ax3.set_title('平均度数对比', fontweight='bold')
    ax3.set_xlabel('数据集')
    ax3.set_ylabel('平均度数')
    ax3.set_xticks(x)
    ax3.set_xticklabels(valid_df['dataset'], rotation=45, ha='right')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图片
    plt.savefig(f'{output_dir}/train_vs_test_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/train_vs_test_comparison.pdf', bbox_inches='tight')
    print(f"已保存对比图到: {output_dir}/train_vs_test_comparison.png 和 .pdf")
    
    plt.show()

if __name__ == "__main__":
    df = analyze_all(DATASETS)
    
    # 打印详细的分析结果
    print("=" * 80)
    print("九个数据集的图结构变化分析结果")
    print("=" * 80)
    print(df.to_string(index=False))
    
    # 创建输出目录
    os.makedirs("new", exist_ok=True)
    
    # 保存完整结果
    df.to_csv("new/graph_change_analysis.csv", index=False)
    print(f"\n已保存完整结果到: new/graph_change_analysis.csv")
    
    # 创建简化的变化指标摘要
    if not df[df['error'].isna()].empty:
        summary_df = df[df['error'].isna()][[
            'dataset', 'dataset_type', 
            'triangle_relative_change', 'clustering_relative_change', 'degree_relative_change',
            'triangle_absolute_change', 'clustering_absolute_change', 'degree_absolute_change'
        ]].copy()
        
        # 按三角形相对变化排序
        summary_df = summary_df.sort_values('triangle_relative_change', ascending=False)
        summary_df.to_csv("new/graph_change_summary.csv", index=False)
        print(f"已保存变化指标摘要到: new/graph_change_summary.csv")
        
        # 打印变化最大的数据集
        print("\n" + "=" * 60)
        print("变化最大的数据集 (按三角形相对变化排序):")
        print("=" * 60)
        for _, row in summary_df.head(3).iterrows():
            print(f"数据集: {row['dataset']} ({row['dataset_type']})")
            print(f"  三角形相对变化: {row['triangle_relative_change']:.4f}")
            print(f"  聚类系数相对变化: {row['clustering_relative_change']:.4f}")
            print(f"  平均度数相对变化: {row['degree_relative_change']:.4f}")
            print()
        
        # 生成可视化图表
        print("\n" + "=" * 60)
        print("正在生成可视化图表...")
        print("=" * 60)
        
        try:
            # 生成变化指标直方图
            create_change_histograms(df, "new")
            
            # 生成训练vs测试对比图
            create_detailed_comparison_plots(df, "new")
            
            print("\n✅ 所有可视化图表已生成完成！")
            print("📊 生成的文件:")
            print("   - new/graph_change_histograms.png/pdf (变化指标直方图)")
            print("   - new/train_vs_test_comparison.png/pdf (训练vs测试对比图)")
            
        except Exception as e:
            print(f"❌ 生成可视化图表时出错: {e}")
            print("请确保已安装 matplotlib 和 seaborn: pip install matplotlib seaborn")
