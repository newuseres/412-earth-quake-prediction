项目快速说明 — README.txt

概述:
- 本项目为地震损害等级预测的代码库，已将文件按功能整理为若干目录以便复现与开发。

目录结构（重要位置）:
- `src/`         : 核心可运行脚本（保留用于部署或快速复现）
- `models/`      : 训练好的模型检查点（`.pth` 文件）
- `tools/`       : 小工具脚本（检查点、评估等）
- `data/`        : 数据文件（`train.csv`, `test.csv`, `submission.csv`）

核心脚本（推荐先看并运行）:
- `src\hgb_focused_solution.py`  — 一键生成最终提交（推荐优先运行）
- `src\train_fast.py`            — 快速训练 HGB（5 折）
- `src\deep_diagnosis.py`       — 性能诊断（每折/每类分析）
- `src\net_optimized_edition.py` — 优化后的神经网络训练脚本
- `src\quick_ensemble.py`       — 基础集成脚本（快速验证）

快速运行示例（PowerShell）:
生成最终提交（默认行为会写入 `data/submission.csv`）:
```powershell
cd "c:\Users\simpe\OneDrive\MCS\412 Data Mining\Project"
python src\hgb_focused_solution.py
```

运行诊断分析（生成诊断报告/混淆矩阵）:
```powershell
python src\deep_diagnosis.py
```

从头训练（可选，耗时）:
```powershell
python src\net_optimized_edition.py
python src\train_fast.py
```

模型与数据:
- `models/` 下包含 `best_model_fold_0.pth` … `best_model_fold_4.pth`（已备份）。
- `data/submission.csv` 为最终预测文件，格式：`building_id,damage_grade`。

维护与清理建议:
- 我已将历史/实验脚本全部移到 `experiments/`，如果确认不再需要可在你允许下删除或移动到长期存档。
- `backup/` 包含被移动或覆盖的原始文件；在确认无误后可清理此目录以释放空间。

接下来我可以为你做：
- 更新 `docs/README.md` 中的相对链接以指向新路径（如果需要）
- 帮你清理 `experiments/` 或 `backup/`（需你确认删除规则）

若要我继续，请直接回复你想要的下一步操作（例如："更新 docs 链接"、"清理 experiments 中超过 30 天的文件"、或"不做任何额外操作"）。


Project README

Overview
This repository contains code and artifacts for earthquake damage-grade prediction. The project is arranged to make core scripts easy to run and to keep trained models and data organized.

Top-level layout (important)
- `src/`    : Core runnable scripts for training, evaluation, and submission
- `models/` : Saved model checkpoints (.pth files)
- `tools/`  : Utility scripts (checkpoint checks, evaluation helpers)
- `data/`   : Dataset files (`train.csv`, `test.csv`, `submission.csv`)

Core scripts (located in `src/`)
- `src\hgb_focused_solution.py`   — Generate final submission (recommended)
- `src\train_fast.py`             — Fast HGB training (5-fold)
- `src\deep_diagnosis.py`         — Diagnostic analysis (per-fold, per-class)
- `src\net_optimized_edition.py`  — Optimized neural network training
- `src\quick_ensemble.py`         — Quick ensemble script for validation

Quick start (PowerShell)
Generate the final submission (writes `data/submission.csv`):
```powershell
cd "C:\Users\simpe\OneDrive\MCS\412 Data Mining\Project"
python src\hgb_focused_solution.py
```

Run diagnostic analysis (outputs reports / confusion matrices):
```powershell
python src\deep_diagnosis.py
```

Train from scratch (optional, time-consuming):
```powershell
python src\net_optimized_edition.py
python src\train_fast.py
```

Models and data
- Checkpoints: `models/best_model_fold_0.pth` … `models/best_model_fold_4.pth`
- Submission: `data/submission.csv` (format: `building_id,damage_grade`)

Notes
- This README intentionally lists only the main folders you will need for daily work (`src/`, `models/`, `tools/`, `data/`).

If you want, I can also:
- Update documentation links in `docs/` to point to these paths (if desired)
- Clean or archive historic/experimental files (I will do deletions only after you confirm rules)

Generated on: 2025-12-02
