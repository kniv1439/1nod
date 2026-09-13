"""
分析不同数据集性能差异的原因
即使正负样本比例都是1:1，不同数据集的性能仍然会不同
"""
import pandas as pd
import numpy as np

def analyze_dataset_characteristics(csv_path, dataset_name):
    """分析数据集的特征"""
    df = pd.read_csv(csv_path)
    
    # 基本统计
    num_edges = len(df)
    num_src_nodes = df['u'].nunique()
    num_dst_nodes = df['i'].nunique()
    num_total_nodes = len(set(df['u'].unique()) | set(df['i'].unique()))
    
    # 计算图的密度
    max_possible_edges = num_src_nodes * num_dst_nodes
    density = num_edges / max_possible_edges if max_possible_edges > 0 else 0
    
    # 计算平均度
    avg_out_degree = num_edges / num_src_nodes if num_src_nodes > 0 else 0
    avg_in_degree = num_edges / num_dst_nodes if num_dst_nodes > 0 else 0
    
    # 时间跨度
    time_span = df['ts'].max() - df['ts'].min()
    
    # 计算负样本的"难度"
    # 在稀疏图中，随机采样的负样本更容易区分（因为大部分节点对之间本来就没有边）
    # 在密集图中，随机采样的负样本可能更接近正样本，更难区分
    
    print(f"\n{'='*60}")
    print(f"{dataset_name} 数据集特征分析")
    print(f"{'='*60}")
    print(f"正样本边数: {num_edges:,}")
    print(f"源节点数: {num_src_nodes:,}")
    print(f"目标节点数: {num_dst_nodes:,}")
    print(f"总唯一节点数: {num_total_nodes:,}")
    print(f"最大可能边数: {max_possible_edges:,}")
    print(f"图密度: {density:.6f} ({density*100:.4f}%)")
    print(f"平均每个源节点的出度: {avg_out_degree:.2f}")
    print(f"平均每个目标节点的入度: {avg_in_degree:.2f}")
    print(f"时间跨度: {time_span:.2f}")
    
    # 分析负样本的难度
    print(f"\n负样本难度分析:")
    if density < 0.01:
        difficulty = "容易"
        reason = "图非常稀疏，随机采样的负样本（不存在的边）与正样本（存在的边）差异很大，模型容易区分"
    elif density < 0.1:
        difficulty = "中等"
        reason = "图相对稀疏，负样本与正样本有一定差异，模型可以较好地区分"
    elif density < 0.5:
        difficulty = "较难"
        reason = "图相对密集，很多节点对之间都有边，随机采样的负样本可能更接近正样本，区分难度增加"
    else:
        difficulty = "很难"
        reason = "图非常密集，大部分节点对之间都有边，负样本与正样本非常相似，模型难以区分"
    
    print(f"  难度等级: {difficulty}")
    print(f"  原因: {reason}")
    
    return {
        'num_edges': num_edges,
        'num_src_nodes': num_src_nodes,
        'num_dst_nodes': num_dst_nodes,
        'density': density,
        'avg_out_degree': avg_out_degree,
        'avg_in_degree': avg_in_degree,
        'difficulty': difficulty
    }

if __name__ == "__main__":
    print("="*60)
    print("为什么即使正负样本比例都是1:1，不同数据集的性能仍然不同？")
    print("="*60)
    
    print("\n关键原因:")
    print("1. 图的密度不同")
    print("   - 稀疏图：随机采样的负样本（不存在的边）与正样本（存在的边）差异很大")
    print("   - 密集图：很多节点对之间都有边，负样本可能更接近正样本，更难区分")
    print("\n2. 负采样的难度不同")
    print("   - 在稀疏图中，大部分节点对之间本来就没有边，模型只需要识别'存在边'即可")
    print("   - 在密集图中，随机采样的负样本可能在历史中曾经存在过，或者节点之间有很多共同邻居")
    print("\n3. 数据集的特性不同")
    print("   - 节点数量、边数量、时间跨度、演化模式等都会影响模型的学习难度")
    print("   - 不同数据集的节点和边特征分布不同")
    print("\n4. 评估指标的相对性")
    print("   - AUC和AP是相对指标，衡量的是模型区分正负样本的能力")
    print("   - 在'容易'的数据集上，模型可能获得更高的AUC")
    print("   - 在'困难'的数据集上，即使模型性能很好，AUC也可能相对较低")
    
    # 分析 reddit 数据集
    try:
        analyze_dataset_characteristics('processed_data/reddit/ml_reddit.csv', 'Reddit')
    except Exception as e:
        print(f"\n无法分析 reddit 数据集: {e}")
    
    print("\n" + "="*60)
    print("总结：")
    print("="*60)
    print("虽然正负样本比例都是1:1，但不同数据集的'难度'不同。")
    print("在稀疏图中，负样本更容易区分，模型可能获得更高的性能。")
    print("在密集图中，负样本更难区分，模型性能可能相对较低。")
    print("这就是为什么即使使用相同的评估方法（1:1正负样本），")
    print("不同数据集的性能仍然会不同的原因。")

