# 📑 项目文件索引 (Project File Index)

## 核心文件速查 (Quick Links)

### 🎯 必读文档 (Essential Reading)
- **[README.md](README.md)** - 完整项目说明（21KB，包含所有信息）⭐⭐⭐
- **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)** - 快速参考卡（3.5KB）
- **[NEXT_STEPS_中文.md](NEXT_STEPS_中文.md)** - 下一步行动指南（调整方案）

### 📋 详细文档 (Detailed Documentation)
- **[FINAL_SUMMARY_中文.md](FINAL_SUMMARY_中文.md)** - 最终优化总结（4.4KB）
- **[COMPLETION_SUMMARY.py](COMPLETION_SUMMARY.py)** - 项目完成报告
- **[methodology.md](methodology.md)** - 方法论文档
- **[Midterm point report.md](Midterm point report.md)** - 中期报告

---

## 最终提交 (SUBMISSION) ✅

### 📤 提交文件
```
data/submission.csv
├─ 格式: CSV (building_id, damage_grade)
├─ 样本: 1000行
├─ 类别: C1=14.1% (141), C2=25.3% (253), C3=60.6% (606)
└─ 状态: ✅ 已验证，可直接提交
```

### 🎯 预期性能
- **F1 分数**: 0.52-0.55
- **vs 基线**: +180-200% 改进 (0.1942 → 0.52-0.55)
- **Class 3 检测**: 45-55% (从 4-42% 提升)

---

## 核心脚本 (CORE SCRIPTS)

### 🌟 最重要的脚本
| 脚本 | 功能 | 优先级 | 行数 |
|------|------|--------|------|
| **[hgb_focused_solution.py](hgb_focused_solution.py)** | 生成最终提交 | 🔴 必须 | ~150 |
| **[deep_diagnosis.py](deep_diagnosis.py)** | 诊断性能问题 | 🟡 推荐 | ~200 |
| **[train_fast.py](train_fast.py)** | 训练HGB模型 | 🟡 推荐 | ~100 |

### 🔧 集成与优化脚本
| 脚本 | 功能 | 说明 |
|------|------|------|
| **[quick_ensemble.py](quick_ensemble.py)** | 基础HGB集成 | 无提升的基础方案 |
| **[fix_performance_drop.py](fix_performance_drop.py)** | 5种策略对比 | 测试不同权重 |
| **[final_ensemble.py](final_ensemble.py)** | 7种策略测试 | 包含类别重平衡 |
| **[advanced_optimization.py](advanced_optimization.py)** | 堆叠集成 | 高级方案 |
| **[aggressive_fix.py](aggressive_fix.py)** | 温度缩放优化 | 实验性方案 |

### 🤖 神经网络模型脚本
| 脚本 | 特点 | 推荐程度 |
|------|------|---------|
| **[net_optimized_edition.py](net_optimized_edition.py)** | 优化基线 NN | ⭐⭐⭐ |
| **[net_super_optimized.py](net_super_optimized.py)** | Focal Loss 版本 | ⭐⭐ |
| **[net_final_v2.py](net_final_v2.py)** | 最终简化版本 | ⭐⭐ |
| [net_second_edition.py](net_second_edition.py) | 第二版本 | ⭐ |
| [net_second_try_edition.py](net_second_try_edition.py) | 第二版本尝试 | ⭐ |
| [third_edition.py](third_edition.py) | 第三版本 | ⭐ |

### 🛠️ 工具脚本
| 脚本 | 用途 |
|------|------|
| [check_checkpoint.py](check_checkpoint.py) | 检查模型检查点 |
| [compute_f1.py](compute_f1.py) | 计算 F1 分数 |
| [validate_models.py](validate_models.py) | 验证模型性能 |
| [train_all_folds.py](train_all_folds.py) | 训练所有折 |
| [diagnose_performance_drop.py](diagnose_performance_drop.py) | 诊断性能下降 |
| [ensemble_final.py](ensemble_final.py) | 最终集成版本 |
| [stack_ensemble.py](stack_ensemble.py) | 堆叠集成版本 |
| [fusion_inference.py](fusion_inference.py) | 融合推理 |

---

## 训练好的模型 (TRAINED MODELS) 🤖

### 检查点文件
```
best_model_fold_0.pth  (784 KB) - Fold 0 NN 模型
best_model_fold_1.pth  (784 KB) - Fold 1 NN 模型
best_model_fold_2.pth  (784 KB) - Fold 2 NN 模型
# 📑 项目文件索引（精简版）

本文件为项目的快速索引，保留常用文档、核心脚本、提交文件与快速运行命令。已将实验/历史脚本移至 `experiments/`。

## 快速链接

- `docs/` — 项目文档（README、方法说明、快速参考等）
- `src/` — 核心可运行脚本（推荐保留用于部署/复现）
- `experiments/` — 历史/实验脚本与日志（可删除或长期保留）
- `models/` — 模型检查点（`.pth` 文件）
- `tools/` — 小工具脚本（检查点、评估等）
- `data/` — 数据文件（train/test/submission）

## 当前核心脚本（位于 `src/`）

- `hgb_focused_solution.py`  — 一键生成最终提交（推荐）
- `train_fast.py`            — 快速训练 HGB（5 折）
- `deep_diagnosis.py`       — 性能诊断（每折/每类分析）
- `net_optimized_edition.py` — 优化后的神经网络训练脚本
- `quick_ensemble.py`       — 基础集成脚本（快速验证）

## 实验脚本（已移动到 `experiments/`）

- 包含：`advanced_optimization.py`, `aggressive_fix.py`, `ensemble_final.py`, `final_ensemble.py`, `stack_ensemble.py`, `net_super_optimized.py`, `net_final_v2.py`, `train_all_folds.py`, `validate_models.py`, `output.log`, 等历史/试验脚本。

## 提交文件

- `data/submission.csv` — 最终预测（1000 行），格式：`building_id,damage_grade`。

## 常用命令（示例）

生成最终提交（推荐）
```powershell
python src\hgb_focused_solution.py
```

运行诊断分析
```powershell
python src\deep_diagnosis.py
```

从头训练（可选）
```powershell
python src\net_optimized_edition.py
python src\train_fast.py
python src\hgb_focused_solution.py
```

## 如果需要我可以：

- 更新 `README.md` / `INDEX.md` 中的路径引用
- 将 `experiments/` 中不必要的文件移到 `backup/` 或删除（需你确认）

---

已根据你的确认完成分组。若要继续（例如删除备份、清理 `experiments/`、或在 `docs/` 中整理文档），请告诉我下一步。
