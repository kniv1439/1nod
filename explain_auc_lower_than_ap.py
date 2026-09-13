"""
分析 AUC-ROC 远低于 AP 的情况
"""
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

def explain_auc_lower_than_ap():
    print("="*70)
    print("AUC-ROC 远低于 AP 的情况分析")
    print("="*70)
    
    print("\n1. 理论可能性")
    print("-"*70)
    print("是的，AUC-ROC 可能会远低于 AP！")
    print("这种情况虽然不常见，但在特定场景下确实会发生。")
    
    print("\n2. 什么情况下会出现 AUC-ROC < AP？")
    print("-"*70)
    
    print("\n【场景1：正样本得分集中且高，负样本得分也较高】")
    print("特征：")
    print("  • 正样本得分：集中在高分段 [0.8, 0.9]，分布很集中")
    print("  • 负样本得分：也集中在较高分段 [0.6, 0.8]，与正样本重叠")
    print("  • 结果：")
    print("    - AP 高：因为正样本得分都很高，排序靠前，精确率高")
    print("    - AUC-ROC 低：因为很多负样本得分也高，区分能力差")
    
    # 示例
    print("\n示例1：")
    y_true_1 = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    y_score_1 = np.array([0.9, 0.88, 0.85, 0.82, 0.8, 0.78, 0.75, 0.72, 0.7, 0.65])
    ap_1 = average_precision_score(y_true_1, y_score_1)
    auc_1 = roc_auc_score(y_true_1, y_score_1)
    print(f"  正样本得分: [0.9, 0.88, 0.85, 0.82, 0.8]")
    print(f"  负样本得分: [0.78, 0.75, 0.72, 0.7, 0.65]")
    print(f"  AP: {ap_1:.4f}, AUC-ROC: {auc_1:.4f}")
    print(f"  差异: AP - AUC-ROC = {ap_1 - auc_1:.4f}")
    print(f"  分析: 虽然正样本得分都高于负样本，但差距不大，AUC-ROC较低")
    print(f"        而正样本得分集中且高，AP较高")
    
    print("\n【场景2：正样本得分分布集中，负样本得分分布分散且重叠】")
    print("特征：")
    print("  • 正样本得分：集中在 [0.7, 0.8]，分布很集中")
    print("  • 负样本得分：分布在 [0.3, 0.9]，分布很分散")
    print("  • 结果：")
    print("    - AP 高：正样本得分集中，排序靠前，精确率高")
    print("    - AUC-ROC 低：很多负样本得分高于部分正样本，区分能力差")
    
    # 示例
    print("\n示例2：")
    y_true_2 = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    y_score_2 = np.array([0.8, 0.78, 0.75, 0.72, 0.7, 0.9, 0.5, 0.4, 0.3, 0.2])
    ap_2 = average_precision_score(y_true_2, y_score_2)
    auc_2 = roc_auc_score(y_true_2, y_score_2)
    print(f"  正样本得分: [0.8, 0.78, 0.75, 0.72, 0.7]")
    print(f"  负样本得分: [0.9, 0.5, 0.4, 0.3, 0.2]")
    print(f"  AP: {ap_2:.4f}, AUC-ROC: {auc_2:.4f}")
    print(f"  差异: AP - AUC-ROC = {ap_2 - auc_2:.4f}")
    print(f"  分析: 有一个负样本得分(0.9)高于所有正样本，导致AUC-ROC很低")
    print(f"        但正样本得分集中，排序靠前，AP仍然较高")
    
    print("\n【场景3：类别不平衡 + 得分重叠】")
    print("特征：")
    print("  • 正样本很少，但得分都很高且集中")
    print("  • 负样本很多，得分分布广泛，部分得分很高")
    print("  • 结果：")
    print("    - AP 高：正样本得分高且集中，排序靠前")
    print("    - AUC-ROC 低：很多负样本得分也高，区分能力差")
    
    # 示例
    print("\n示例3：类别不平衡")
    np.random.seed(42)
    y_true_3 = np.concatenate([np.ones(5), np.zeros(50)])
    # 正样本得分集中且高
    y_score_3_pos = np.random.uniform(0.85, 0.95, 5)
    # 负样本得分分布广泛，部分很高
    y_score_3_neg = np.concatenate([
        np.random.uniform(0.8, 0.9, 10),  # 部分负样本得分很高
        np.random.uniform(0.3, 0.7, 40)   # 大部分负样本得分中等
    ])
    y_score_3 = np.concatenate([y_score_3_pos, y_score_3_neg])
    # 打乱顺序
    indices = np.random.permutation(len(y_true_3))
    y_true_3 = y_true_3[indices]
    y_score_3 = y_score_3[indices]
    
    ap_3 = average_precision_score(y_true_3, y_score_3)
    auc_3 = roc_auc_score(y_true_3, y_score_3)
    print(f"  正样本数: 5, 负样本数: 50")
    print(f"  正样本得分范围: [{y_score_3_pos.min():.2f}, {y_score_3_pos.max():.2f}]")
    print(f"  负样本得分范围: [{y_score_3_neg.min():.2f}, {y_score_3_neg.max():.2f}]")
    print(f"  AP: {ap_3:.4f}, AUC-ROC: {auc_3:.4f}")
    print(f"  差异: AP - AUC-ROC = {ap_3 - auc_3:.4f}")
    if ap_3 > auc_3:
        print(f"  ✓ AP 高于 AUC-ROC！")
    
    print("\n3. 为什么会出现这种情况？")
    print("-"*70)
    print("【AUC-ROC 的计算方式】")
    print("  • 比较每一对（正样本，负样本）")
    print("  • 如果正样本得分 > 负样本得分，则计数+1")
    print("  • AUC-ROC = 正样本得分高于负样本的比例")
    print("  • 如果很多负样本得分高于正样本，AUC-ROC 会很低")
    
    print("\n【AP 的计算方式】")
    print("  • 按得分从高到低排序")
    print("  • 计算每个召回率下的精确率")
    print("  • AP = PR曲线下的面积")
    print("  • 如果正样本得分集中且高，即使有负样本得分更高，AP 仍然可能较高")
    
    print("\n【关键差异】")
    print("  • AUC-ROC：考虑所有正负样本对的比较")
    print("    - 如果有很多负样本得分高于正样本，AUC-ROC 会很低")
    print("  • AP：主要关注正样本的排序")
    print("    - 如果正样本得分集中且高，即使有负样本得分更高，AP 仍然可能较高")
    
    print("\n4. 实际意义")
    print("-"*70)
    print("当 AUC-ROC < AP 时：")
    print("  • 说明：模型对正样本的识别很准确（正样本得分高且集中）")
    print("  • 但：模型对负样本的识别不够准确（部分负样本得分也很高）")
    print("  • 可能原因：")
    print("    1. 模型对'高置信度正样本'识别很好，但对'困难负样本'识别较差")
    print("    2. 负样本中存在'假负样本'（实际上是正样本但被标记为负样本）")
    print("    3. 模型过拟合，对训练集中的正样本模式记忆很好")
    
    print("\n5. 在链路预测中的含义")
    print("-"*70)
    print("如果 AUC-ROC < AP：")
    print("  • 模型能够很好地识别'明显的正样本'（存在的边）")
    print("  • 但模型难以区分'困难负样本'（不存在的边，但看起来很像正样本）")
    print("  • 这可能意味着：")
    print("    - 模型对正样本的特征学习得很好")
    print("    - 但模型对负样本的多样性理解不足")
    print("    - 或者负样本中存在很多'接近正样本'的情况（密集图）")
    
    print("\n6. 如何理解这种差异？")
    print("-"*70)
    print("假设场景：")
    print("  • 10个正样本，得分都是 0.85（很集中）")
    print("  • 10个负样本，得分分布：")
    print("    - 3个得分 0.9（高于正样本）")
    print("    - 7个得分 0.3（低于正样本）")
    print()
    print("AUC-ROC 计算：")
    print("  • 10个正样本 × 10个负样本 = 100对比较")
    print("  • 正样本得分 > 负样本得分：7个正样本 × 7个低分负样本 = 49对")
    print("  • 正样本得分 < 负样本得分：10个正样本 × 3个高分负样本 = 30对")
    print("  • AUC-ROC ≈ 49/100 = 0.49（很低！）")
    print()
    print("AP 计算：")
    print("  • 按得分排序：0.9(负), 0.9(负), 0.9(负), 0.85(正), 0.85(正), ..., 0.3(负)")
    print("  • 前3个是负样本，精确率 = 0/3 = 0")
    print("  • 前13个中有10个正样本，精确率 = 10/13 ≈ 0.77")
    print("  • AP 会考虑所有召回率下的精确率，可能仍然较高")
    
    print("\n" + "="*70)
    print("总结：")
    print("="*70)
    print("1. AUC-ROC 可能会远低于 AP")
    print("2. 这种情况发生在：正样本得分集中且高，但负样本得分也较高或分布分散")
    print("3. AUC-ROC 关注整体区分能力，AP 关注正样本排序质量")
    print("4. 在链路预测中，这可能意味着模型对正样本识别好，但对困难负样本识别差")
    print("="*70)

if __name__ == "__main__":
    explain_auc_lower_than_ap()




