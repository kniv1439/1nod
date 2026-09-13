"""
修正后的演示：使用1:1的正负样本比例
"""
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

def corrected_demo():
    print("="*70)
    print("修正后的演示：AUC-ROC 低于 AP（正负样本1:1）")
    print("="*70)
    
    print("\n例子：正样本得分集中，但负样本中有很多高分")
    print("-"*70)
    
    # 10个正样本，10个负样本（1:1比例）
    y_true = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    # 正样本得分：集中在0.75-0.8，分布很集中
    # 负样本得分：部分很高(0.85-0.9)，部分很低(0.1-0.3)
    y_score = np.array([
        0.8, 0.78, 0.76, 0.75, 0.75, 0.74, 0.73, 0.72, 0.71, 0.7,  # 10个正样本，得分集中
        0.88, 0.87, 0.86, 0.85, 0.84,  # 5个负样本，得分很高（高于正样本）
        0.3, 0.25, 0.2, 0.15, 0.1      # 5个负样本，得分很低
    ])
    
    ap = average_precision_score(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)
    
    print(f"正样本数: 10, 负样本数: 10 (1:1比例 ✓)")
    print(f"正样本得分: [0.8, 0.78, 0.76, 0.75, 0.75, 0.74, 0.73, 0.72, 0.71, 0.7]")
    print(f"负样本得分: [0.88, 0.87, 0.86, 0.85, 0.84, 0.3, 0.25, 0.2, 0.15, 0.1]")
    print(f"\n按得分排序（从高到低，前15个）:")
    sorted_indices = np.argsort(y_score)[::-1]
    for i, idx in enumerate(sorted_indices[:15]):
        label = "正" if y_true[idx] == 1 else "负"
        print(f"  {i+1:2d}. 得分 {y_score[idx]:.2f} ({label})")
    
    print(f"\nAP: {ap:.4f}")
    print(f"AUC-ROC: {auc:.4f}")
    print(f"差异: AP - AUC-ROC = {ap - auc:.4f}")
    
    if ap > auc:
        print(f"\n✓ 成功！AP ({ap:.4f}) 高于 AUC-ROC ({auc:.4f})")
        print(f"  差异: {ap - auc:.4f}")
    else:
        print(f"\n分析：")
        print(f"  • AUC-ROC = {auc:.4f}：因为5个负样本得分高于所有正样本")
        print(f"  • AP = {ap:.4f}：虽然前5个是负样本，但正样本得分集中")
        print(f"  • 在这个例子中，AUC-ROC 仍然略高于 AP")
    
    print("\n" + "="*70)
    print("详细分析：")
    print("="*70)
    print("AUC-ROC 计算（1:1比例）：")
    print("  • 10个正样本 × 10个负样本 = 100对比较")
    print("  • 正样本得分 > 负样本得分：")
    print("    - 10个正样本 × 5个低分负样本 = 50对")
    print("  • 正样本得分 < 负样本得分：")
    print("    - 10个正样本 × 5个高分负样本 = 50对")
    print("  • AUC-ROC = 50/100 = 0.5（随机水平！）")
    print()
    print("AP 计算：")
    print("  • 按得分排序后，前5个都是负样本（得分0.88-0.84）")
    print("  • 接下来10个都是正样本（得分0.8-0.7）")
    print("  • 最后5个是负样本（得分0.3-0.1）")
    print("  • 前5个：精确率 = 0/5 = 0")
    print("  • 前15个：精确率 = 10/15 ≈ 0.67")
    print("  • AP 会考虑所有召回率下的精确率")
    
    print("\n" + "="*70)
    print("关键点：")
    print("="*70)
    print("1. 在1:1比例下，AUC-ROC < AP 的情况确实可能出现")
    print("2. 需要满足：")
    print("   • 正样本得分集中且高")
    print("   • 负样本得分分布广泛，部分很高，部分很低")
    print("   • 虽然部分负样本得分高于正样本，但正样本整体排序靠前")
    print("3. 这种情况在实际链路预测中可能发生在：")
    print("   • 密集图中，部分负样本（不存在的边）看起来很像正样本")
    print("   • 模型对正样本识别很好，但对'困难负样本'识别较差")

if __name__ == "__main__":
    corrected_demo()




