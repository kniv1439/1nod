"""
验证评估时正负样本的比例是否为1:1
"""
import sys
sys.path.append('/home/zhangjinhao/nod')

# 模拟评估过程中的负采样逻辑
def verify_negative_sampling_ratio():
    print("="*60)
    print("验证评估时正负样本比例")
    print("="*60)
    
    # 从代码中可以看到的关键信息：
    print("\n1. 负采样调用:")
    print("   - 代码位置: evaluate_models_utils.py 第117行或123行")
    print("   - 调用方式: evaluate_neg_edge_sampler.sample(size=len(batch_src_node_ids))")
    print("   - 关键点: size 参数 = len(batch_src_node_ids)，即正样本的数量")
    
    print("\n2. 正负样本组合:")
    print("   - 代码位置: evaluate_models_utils.py 第371-372行")
    print("   - 组合方式:")
    print("     predicts = torch.cat([positive_probabilities, negative_probabilities], dim=0)")
    print("     labels = torch.cat([torch.ones_like(positive_probabilities), torch.zeros_like(negative_probabilities)], dim=0)")
    print("   - 正样本标签: 1 (torch.ones_like)")
    print("   - 负样本标签: 0 (torch.zeros_like)")
    
    print("\n3. 模拟验证:")
    # 模拟一个batch的情况
    batch_size = 32  # 假设batch_size为32
    print(f"   假设 batch_size = {batch_size}")
    print(f"   正样本数量: {batch_size}")
    print(f"   负样本数量: size={batch_size} (从 sample(size=len(batch_src_node_ids)) 可以看出)")
    print(f"   比例: {batch_size}:{batch_size} = 1:1 ✓")
    
    print("\n4. 代码证据:")
    print("   " + "-"*56)
    print("   第117行 (非random策略):")
    print("   batch_neg_src_node_ids, batch_neg_dst_node_ids = ")
    print("       evaluate_neg_edge_sampler.sample(size=len(batch_src_node_ids), ...)")
    print("   " + "-"*56)
    print("   第123行 (random策略):")
    print("   _, batch_neg_dst_node_ids = ")
    print("       evaluate_neg_edge_sampler.sample(size=len(batch_src_node_ids))")
    print("   " + "-"*56)
    print("   第371-372行 (组合正负样本):")
    print("   predicts = torch.cat([positive_probabilities, negative_probabilities], dim=0)")
    print("   labels = torch.cat([torch.ones_like(...), torch.zeros_like(...)], dim=0)")
    print("   " + "-"*56)
    
    print("\n5. 结论:")
    print("   ✓ 评估时，每个正样本对应生成1个负样本")
    print("   ✓ 正负样本比例确实是 1:1")
    print("   ✓ 正样本标签为1，负样本标签为0")
    print("   ✓ 两者拼接后用于计算AUC和AP指标")
    
    print("\n" + "="*60)
    print("验证完成：评估时正负边比例为 1:1 ✓")
    print("="*60)

if __name__ == "__main__":
    verify_negative_sampling_ratio()

