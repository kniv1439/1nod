"""
展示如何从代码中看出评估时正负样本比例是1:1
"""
import sys

def show_evidence():
    print("="*70)
    print("如何从代码中看出评估时正负样本比例是1:1")
    print("="*70)
    
    print("\n【证据1：负采样时的 size 参数】")
    print("-"*70)
    print("文件：evaluate_models_utils.py")
    print("位置：第117行或第123行")
    print()
    print("代码片段：")
    print("""
    if evaluate_neg_edge_sampler.negative_sample_strategy != 'random':
        batch_neg_src_node_ids, batch_neg_dst_node_ids = \\
            evaluate_neg_edge_sampler.sample(size=len(batch_src_node_ids), ...)
    else:
        _, batch_neg_dst_node_ids = \\
            evaluate_neg_edge_sampler.sample(size=len(batch_src_node_ids))
    """)
    print()
    print("关键点：")
    print("  • size=len(batch_src_node_ids)")
    print("  • batch_src_node_ids 是正样本的数量（从 evaluate_data 中获取）")
    print("  • sample() 方法会根据 size 参数生成对应数量的负样本")
    print("  • 因此：负样本数量 = len(batch_src_node_ids) = 正样本数量")
    print("  • 结论：正负样本比例 = 1:1 ✓")
    
    print("\n【证据2：正负样本的拼接方式】")
    print("-"*70)
    print("文件：evaluate_models_utils.py")
    print("位置：第371-372行")
    print()
    print("代码片段：")
    print("""
    # 拼接预测结果和真实标签
    predicts = torch.cat([positive_probabilities, negative_probabilities], dim=0)
    labels = torch.cat([torch.ones_like(positive_probabilities), torch.zeros_like(negative_probabilities)], dim=0)
    """)
    print()
    print("关键点：")
    print("  • positive_probabilities：正样本的预测概率")
    print("  • negative_probabilities：负样本的预测概率")
    print("  • torch.cat([..., ...], dim=0)：在维度0上拼接")
    print("  • torch.ones_like(positive_probabilities)：正样本标签全为1")
    print("  • torch.zeros_like(negative_probabilities)：负样本标签全为0")
    print("  • 如果两者数量不同，拼接后的 labels 中1和0的数量会不同")
    print("  • 但从代码逻辑看，两者数量应该相同（都是 batch_size）")
    
    print("\n【证据3：查看正负样本的生成过程】")
    print("-"*70)
    print("文件：evaluate_models_utils.py")
    print("位置：第112-114行（正样本）和第117/123行（负样本）")
    print()
    print("正样本获取：")
    print("  batch_src_node_ids, batch_dst_node_ids = \\")
    print("      evaluate_data.src_node_ids[evaluate_data_indices], \\")
    print("      evaluate_data.dst_node_ids[evaluate_data_indices]")
    print("  • 从 evaluate_data 中获取，数量 = len(evaluate_data_indices) = batch_size")
    print()
    print("负样本生成：")
    print("  batch_neg_src_node_ids, batch_neg_dst_node_ids = \\")
    print("      evaluate_neg_edge_sampler.sample(size=len(batch_src_node_ids))")
    print("  • 通过负采样器生成，数量 = size = len(batch_src_node_ids) = batch_size")
    print()
    print("结论：")
    print("  • 正样本数量 = batch_size")
    print("  • 负样本数量 = batch_size")
    print("  • 比例 = 1:1 ✓")
    
    print("\n【证据4：查看 NegativeEdgeSampler 的 sample 方法】")
    print("-"*70)
    print("文件：utils/utils.py")
    print("位置：第373-398行")
    print()
    print("方法签名：")
    print("  def sample(self, size: int, ...):")
    print("      \"\"\"")
    print("      sample negative edges")
    print("      :param size: int, number of sampled negative edges")
    print("      \"\"\"")
    print()
    print("关键点：")
    print("  • size 参数明确表示'要采样的负样本数量'")
    print("  • 当调用 sample(size=len(batch_src_node_ids)) 时")
    print("  • 返回的负样本数量 = size = len(batch_src_node_ids)")
    print("  • 因此：负样本数量 = 正样本数量 = 1:1 ✓")
    
    print("\n【证据5：实际验证】")
    print("-"*70)
    print("可以通过以下方式验证：")
    print("  1. 在代码中添加打印语句：")
    print("     print(f'正样本数量: {len(batch_src_node_ids)}')")
    print("     print(f'负样本数量: {len(batch_neg_src_node_ids)}')")
    print()
    print("  2. 检查拼接后的 labels：")
    print("     print(f'标签中1的数量: {labels.sum()}')")
    print("     print(f'标签中0的数量: {len(labels) - labels.sum()}')")
    print("     print(f'总数量: {len(labels)}')")
    print()
    print("  3. 如果比例是1:1，应该看到：")
    print("     标签中1的数量 = 标签中0的数量 = batch_size")
    print("     总数量 = 2 * batch_size")
    
    print("\n" + "="*70)
    print("总结：")
    print("="*70)
    print("从代码中可以明确看出：")
    print("1. 负采样时 size=len(batch_src_node_ids)，即负样本数量=正样本数量")
    print("2. 正负样本拼接时使用相同的方式，数量应该相等")
    print("3. 标签中1和0的数量应该相等（都是batch_size）")
    print("4. 因此，评估时正负样本比例确实是 1:1")
    print("="*70)

if __name__ == "__main__":
    show_evidence()




