"""
演示 AUC-ROC 低于 AP 的具体例子
"""
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

def demonstrate_auc_lower_than_ap():
    print("="*70)
    print("演示：AUC-ROC 低于 AP 的具体例子")
    print("="*70)
    
    # 创建一个明确的例子：正样本得分集中且高，但负样本中有很多高分
    print("\n例子：正样本得分集中，但负样本中有很多高分")
    print("-"*70)
    
    # 5个正样本，得分都很高且集中
    y_true = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    # 正样本得分：集中在0.75-0.8
    # 负样本得分：部分很高(0.85-0.9)，部分很低(0.1-0.3)
    y_score = np.array([
        0.8, 0.78, 0.76, 0.75, 0.75,  # 5个正样本，得分集中且高
        0.88, 0.87, 0.86, 0.85, 0.84,  # 5个负样本，得分很高（高于正样本）
        0.3, 0.25, 0.2, 0.15, 0.1     # 5个负样本，得分很低
    ])
    
    ap = average_precision_score(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)
    
    print(f"正样本数: 5, 负样本数: 10")
    print(f"正样本得分: [0.8, 0.78, 0.76, 0.75, 0.75]")
    print(f"负样本得分: [0.88, 0.87, 0.86, 0.85, 0.84, 0.3, 0.25, 0.2, 0.15, 0.1]")
    print(f"\n按得分排序（从高到低）:")
    sorted_indices = np.argsort(y_score)[::-1]
    for i, idx in enumerate(sorted_indices[:10]):
        label = "正" if y_true[idx] == 1 else "负"
        print(f"  {i+1}. 得分 {y_score[idx]:.2f} ({label})")
    
    print(f"\nAP: {ap:.4f}")
    print(f"AUC-ROC: {auc:.4f}")
    print(f"差异: AP - AUC-ROC = {ap - auc:.4f}")
    
    if ap > auc:
        print(f"\n✓ 成功！AP ({ap:.4f}) 高于 AUC-ROC ({auc:.4f})")
        print(f"  差异: {ap - auc:.4f}")
    else:
        print(f"\n✗ 这个例子中 AUC-ROC 仍然高于 AP")
    
    print("\n" + "="*70)
    print("分析：")
    print("="*70)
    print("AUC-ROC 计算：")
    print("  • 5个正样本 × 10个负样本 = 50对比较")
    print("  • 正样本得分 > 负样本得分：")
    print("    - 5个正样本 × 5个低分负样本 = 25对")
    print("  • 正样本得分 < 负样本得分：")
    print("    - 5个正样本 × 5个高分负样本 = 25对")
    print("  • AUC-ROC = 25/50 = 0.5（随机水平！）")
    print()
    print("AP 计算：")
    print("  • 按得分排序后，前5个都是负样本（得分0.88-0.84）")
    print("  • 接下来5个都是正样本（得分0.8-0.75）")
    print("  • 前10个中，有5个正样本，精确率 = 5/10 = 0.5")
    print("  • 但正样本得分集中且高，在召回率增加时精确率会提高")
    print("  • AP 会考虑所有召回率下的精确率，可能高于0.5")
    
    print("\n" + "="*70)
    print("关键理解：")
    print("="*70)
    print("1. AUC-ROC 关注：'有多少正样本得分高于负样本'")
    print("   - 如果一半负样本得分高于正样本，AUC-ROC ≈ 0.5")
    print()
    print("2. AP 关注：'正样本的排序质量'")
    print("   - 如果正样本得分集中且高，即使有负样本得分更高，AP 仍然可能较高")
    print()
    print("3. 因此，当正样本得分集中但负样本得分分布广泛时，")
    print("   可能出现 AUC-ROC < AP 的情况")

if __name__ == "__main__":
    demonstrate_auc_lower_than_ap()




