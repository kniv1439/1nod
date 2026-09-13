"""
详细解释 AP (Average Precision) 和 AUC-ROC 的差异
"""
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve, roc_curve
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt

def explain_ap_vs_auc():
    print("="*70)
    print("AP (Average Precision) vs AUC-ROC 详解")
    print("="*70)
    
    print("\n1. 基本定义")
    print("-"*70)
    print("AUC-ROC (Area Under ROC Curve):")
    print("  • ROC曲线：横轴是FPR (False Positive Rate)，纵轴是TPR (True Positive Rate)")
    print("  • FPR = FP / (FP + TN)  (假正例率：负样本被错误预测为正样本的比例)")
    print("  • TPR = TP / (TP + FN)  (真正例率：正样本被正确预测的比例，即召回率)")
    print("  • AUC-ROC = ROC曲线下的面积")
    print("  • 含义：模型在所有可能的阈值下，区分正负样本的能力")
    
    print("\nAP (Average Precision):")
    print("  • PR曲线：横轴是Recall (召回率)，纵轴是Precision (精确率)")
    print("  • Recall = TP / (TP + FN)  (召回率：正样本被正确预测的比例)")
    print("  • Precision = TP / (TP + FP)  (精确率：预测为正样本中真正为正样本的比例)")
    print("  • AP = PR曲线下的面积")
    print("  • 含义：模型在正样本上的排序质量")
    
    print("\n2. 关键差异")
    print("-"*70)
    print("【差异1：关注点不同】")
    print("  • AUC-ROC：关注'区分能力'（能否区分正负样本）")
    print("    - 考虑所有样本（正样本和负样本）")
    print("    - 衡量：'正样本得分是否高于负样本得分'")
    print()
    print("  • AP：关注'排序质量'（正样本的排序是否靠前）")
    print("    - 主要关注正样本")
    print("    - 衡量：'在召回率增加时，精确率如何变化'")
    
    print("\n【差异2：对类别不平衡的敏感度不同】")
    print("  • AUC-ROC：对类别不平衡不敏感")
    print("    - 即使负样本很多，只要正样本得分高于负样本，AUC-ROC仍然很高")
    print("    - 例子：1000个负样本，10个正样本，如果所有正样本得分都高于负样本")
    print("      → AUC-ROC ≈ 1.0（完美）")
    print()
    print("  • AP：对类别不平衡敏感")
    print("    - 当负样本很多时，即使正样本得分很高，精确率也可能较低")
    print("    - 例子：1000个负样本，10个正样本，即使所有正样本得分都高于负样本")
    print("      → 如果前100个预测中有10个正样本，精确率 = 10/100 = 0.1")
    print("      → AP 会较低，因为精确率在低召回率时较低")
    
    print("\n3. 在链路预测中的表现")
    print("-"*70)
    print("在链路预测任务中（正负样本1:1）：")
    print("  • 理论上，AP 和 AUC-ROC 应该比较接近")
    print("  • 但实际上，它们仍然可能有差异，原因如下：")
    
    print("\n【原因1：排序质量 vs 区分能力】")
    print("  场景：")
    print("    - 正样本得分：[0.9, 0.8, 0.7, 0.6, 0.5]")
    print("    - 负样本得分：[0.4, 0.3, 0.2, 0.1, 0.0]")
    print()
    print("  AUC-ROC：")
    print("    - 所有正样本得分都高于负样本 → AUC-ROC ≈ 1.0")
    print()
    print("  AP：")
    print("    - 如果按得分排序：[0.9(正), 0.8(正), 0.7(正), 0.6(正), 0.5(正), 0.4(负), ...]")
    print("    - 前5个都是正样本 → 精确率 = 1.0")
    print("    - AP ≈ 1.0（也很高）")
    print()
    print("  但如果排序不同：")
    print("    - 排序：[0.9(正), 0.4(负), 0.8(正), 0.3(负), 0.7(正), ...]")
    print("    - AUC-ROC：仍然很高（因为正样本得分整体高于负样本）")
    print("    - AP：会较低（因为精确率在召回率增加时下降）")
    
    print("\n【原因2：得分分布的影响】")
    print("  场景A：得分分布清晰")
    print("    - 正样本得分：集中在 [0.7, 0.9]")
    print("    - 负样本得分：集中在 [0.1, 0.3]")
    print("    - 结果：AP ≈ AUC-ROC（两者都很高）")
    print()
    print("  场景B：得分分布重叠")
    print("    - 正样本得分：分布在 [0.3, 0.9]")
    print("    - 负样本得分：分布在 [0.1, 0.7]")
    print("    - 结果：")
    print("      • AUC-ROC：可能仍然较高（因为正样本得分平均高于负样本）")
    print("      • AP：可能较低（因为很多负样本得分高于部分正样本，精确率下降）")
    
    print("\n4. 实际例子")
    print("-"*70)
    
    # 创建示例数据
    print("示例1：得分分布清晰（AP ≈ AUC-ROC）")
    y_true_1 = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    y_score_1 = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0])
    ap_1 = average_precision_score(y_true_1, y_score_1)
    auc_1 = roc_auc_score(y_true_1, y_score_1)
    print(f"  AP: {ap_1:.4f}, AUC-ROC: {auc_1:.4f}, 差异: {abs(ap_1 - auc_1):.4f}")
    
    print("\n示例2：得分分布重叠（AP < AUC-ROC）")
    y_true_2 = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    y_score_2 = np.array([0.9, 0.7, 0.5, 0.3, 0.1, 0.8, 0.6, 0.4, 0.2, 0.0])
    ap_2 = average_precision_score(y_true_2, y_score_2)
    auc_2 = roc_auc_score(y_true_2, y_score_2)
    print(f"  AP: {ap_2:.4f}, AUC-ROC: {auc_2:.4f}, 差异: {abs(ap_2 - auc_2):.4f}")
    print("  注意：AP 明显低于 AUC-ROC")
    
    print("\n示例3：类别不平衡（AP < AUC-ROC）")
    y_true_3 = np.concatenate([np.ones(10), np.zeros(1000)])
    y_score_3 = np.concatenate([
        np.random.uniform(0.7, 0.9, 10),  # 正样本得分较高
        np.random.uniform(0.1, 0.3, 1000)  # 负样本得分较低
    ])
    ap_3 = average_precision_score(y_true_3, y_score_3)
    auc_3 = roc_auc_score(y_true_3, y_score_3)
    print(f"  AP: {ap_3:.4f}, AUC-ROC: {auc_3:.4f}, 差异: {abs(ap_3 - auc_3):.4f}")
    print("  注意：即使正负样本得分分离很好，AP 仍然低于 AUC-ROC")
    
    print("\n5. 在链路预测中的实际意义")
    print("-"*70)
    print("当 AP < AUC-ROC 时：")
    print("  • 说明：模型能够区分正负样本，但正样本的排序不够理想")
    print("  • 可能原因：")
    print("    1. 部分负样本得分高于部分正样本")
    print("    2. 正样本得分分布较分散")
    print("    3. 模型对'高置信度正样本'的识别不够准确")
    print()
    print("当 AP > AUC-ROC 时（较少见）：")
    print("  • 说明：模型在正样本排序上表现很好，但整体区分能力一般")
    print("  • 可能原因：")
    print("    1. 正样本得分集中且较高")
    print("    2. 负样本得分分布较分散")
    
    print("\n6. 如何选择指标")
    print("-"*70)
    print("在链路预测任务中：")
    print("  • 如果关注'能否找到所有正样本' → 使用 AUC-ROC")
    print("  • 如果关注'找到的正样本是否准确' → 使用 AP")
    print("  • 通常两个指标都报告，因为它们提供不同的视角")
    
    print("\n" + "="*70)
    print("总结：")
    print("="*70)
    print("1. AUC-ROC 关注区分能力，AP 关注排序质量")
    print("2. 在类别不平衡时，AP 通常低于 AUC-ROC")
    print("3. 在链路预测（1:1）中，两者应该接近，但仍有差异")
    print("4. 差异主要来自得分分布和排序质量")
    print("="*70)

if __name__ == "__main__":
    explain_ap_vs_auc()




