# 2026 策联杯 C 题可复现工程

本工程实现 C 题 Q1--Q4 的完整数据链：赛前观看人数预测、场馆与时段 COPT 优化、第三轮反馈下的动态资源 COPT 优化，以及 2026 美加墨世界杯的同口径结构比较。正式论文和提交材料均由现有代码、求解日志与独立验收结果生成。

## 目录与输入

项目根目录为 `c_contest/`。开发工作树中的官方 C 题附件保留在项目同级的 `2026策联杯C题/` 中，不改写。程序优先读取支撑材料包内的 `data/official/`；若该目录不存在，再从项目上一级目录递归发现：

- 唯一包含 `historical_matches`、`teams`、`groups_matches`、`base_predictions` 工作表的 `.xlsx`；
- 六份官方 CSV 模板；
- Q4 使用的 OpenFootball 原始快照位于 `research/q4/`，每行结果保留 URL 与采集日期。

提交构建器会将唯一工作簿和六份模板复制到 ZIP 的 `data/official/`，使解压后的支撑材料可独立复现。若外部同名官方文件出现多份，数据发现会主动失败，避免误读附件。

## Python 环境

本地使用两个 Python 3.12 环境，以隔离 COPT 评估版：

```powershell
py -3.12 -m venv .venv-demo
.\.venv-demo\Scripts\python.exe -m pip install -r requirements-demo.txt

py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-copt.txt
```

`requirements-demo.txt` 用于 Q1、Q4、分析、绘图和测试；`requirements-copt.txt` 用于 Q2/Q3 的 COPT 模型。当前本机为 COPT 8.0.6 非商业评估模式，单个 MIP 最多 2000 个变量和 2000 个约束。COPT 安装成功不等于具备商业许可证，复现时须检查求解日志中的许可证、状态和 gap。

## 一键复现

从 PowerShell 运行：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_all.ps1
```

默认顺序为：Q1 正式预测、Q1 七模型比较与报告、Q2 可行基线、Q2 COPT、Q3 COPT、Q4 数据解析与比较、稳健性/收敛分析、论文图、测试、14 项交付验收、两份 PDF 编译。随机种子统一为 `20260814`。

开发时可使用：

```powershell
.\scripts\run_all.ps1 -SkipAnalysis -SkipPdf
```

这只缩短调试时间，不构成最终交付验收。最终版本必须无跳过参数完整运行。

## 正式输出

| 任务 | 正式文件 |
|---|---|
| Q1 隐藏测试预测 | `outputs/q1/demo/result_1_test_prediction.csv` |
| Q1 72 场预测 | `outputs/q1/demo/result_1_match_prediction.csv` |
| Q2 COPT 赛程 | `outputs/q2/copt/result_2_group_schedule.csv` |
| Q3 动态策略 | `outputs/q3/copt/result_3_dynamic_strategy.csv` |
| Q4 真实赛程 | `outputs/q4/actual_schedule.csv` |
| Q4 对比 | `outputs/q4/demo/result_4_schedule_comparison.csv` |
| 主论文 | `paper/build/main.pdf` |
| AI 使用详情 | `paper/build/ai_usage.pdf` |
| 最终验收 | `outputs/validation/deliverable_validation.json` |

Q1 的时间验证 MSE 不是隐藏测试 MSE。Q2 的 0 gap 只证明受限候选集内的整数最优，不证明全部 `72 x 16 x 80` 组合的全局最优。Q4 只比较公开赛程可同口径识别的结构指标，不伪造历史票务、转播、公平或风险字段。

## COPT 证据

Q2/Q3 的完整 COPT 源码、日志与元数据分别位于：

- `src/c_contest_q1/q2_copt.py`、`outputs/q2/copt/solver.log`、`outputs/q2/copt/solve_metadata.json`；
- `src/c_contest_q1/q3_copt.py`、`outputs/q3/copt/solver.log`、`outputs/q3/copt/solve_metadata.json`；
- `docs/project/COPT_EVIDENCE.md` 汇总模型规模、状态、gap、独立目标对账和声明边界。

## 构建提交文件

比赛要求文件名含三位参赛队号。未获得队号时可生成占位版本：

```powershell
.\.venv-demo\Scripts\python.exe scripts\build_submission.py --team-id XXX
```

获得队号后必须重新运行，例如：

```powershell
.\.venv-demo\Scripts\python.exe scripts\build_submission.py --team-id 123
```

脚本使用白名单构建 `submission/<队号>_参赛论文.pdf` 和 `submission/<队号>_支撑材料.zip`，生成 `submission_manifest.json` 与 SHA-256，并在任一文件达到 20 MB 时失败。它不会打包虚拟环境、缓存、旧 Demo 或 PDF QA 图片。
