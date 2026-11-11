# Kimi3.1 Pro v0.28 工作区版本记录

记录日期为 2026-11-11，分支为 `xiesl-028-support-kimi3.1-pro`。本版本保存当前插件工作区，包括 Kimi3.1 Pro 适配、preload/IPC 兼容修复、v1 prefill 路径及显式选择的实验算子。当前生成精度仍有重复 token 问题。

## 实现与使用配置

- 完整模型为 Kimi-K3.1-Pro-CHANNEL-INT4-w4a8，93 层，PP2/TP8，节点为 nmz22 与 nmz20。
- 修复有界 Unix socket 路径、TP/PP daemon workspace 初始化、PP 本地层权重过滤以及 DSA indexer 的单一参数所有权。
- 修复 QuantFP8 HIP fallback 动态派发递归，以及 K31 临时 dense MLA 构造与最终 sparse MLA 配置的兼容性。
- v1 使用 `VLLM_HCU_KIMI_SPARSE_MLA_BACKEND=expanded_masked` 和 `VLLM_HCU_KIMI_SHARE_QKV_QUANT=1`。三路投影实验 `VLLM_HCU_KIMI_MERGE_QKV_GEMM=0`，对应 validation 为 0。
- expanded prefill 的 guard 要求单请求 fresh 4096 token、精确形状、本地 12 heads、BF16 当前输入及投影权重等条件。FP8 cache 本身不能单独判定该分支是否命中。完整服务测试未采集逐次命中计数。
- 默认 sparse provider 仍由源码的 `flashmla` 默认值决定。HIP、migrated HIP 与 CK 实验保留供复现，需要显式选择和构建。本次性能采用 expanded_masked 配置。

## 完整模型 prefill 性能

本次配置为 eager、max_model_len=16384、max_num_seqs=128、max_num_batched_tokens=8192、block_size=64、SSM BF16、prefix caching 开启并使用 Mamba align。公开 FP8 E4M3 KV 通过正式 FLASHMLA_SPARSE 路径归一化为内部 fp8_ds_mla 格式。权重由独立 daemon 预载，engine 使用 strict IPC。

每档输入 4096 token、输出 1 token，请求数等于客户端最大并发。每档排除一波预热，再测三轮。下表 K31 TTFT 为每轮请求均值的跨三轮中位数，输入吞吐为三轮中位数。24 波测量共 765/765 请求成功，实际输入输出长度均符合配置。

| 并发 | K3 历史 TTFT ms | K31 TTFT ms | K3 历史输入 token/s | K31 输入 token/s |
|---:|---:|---:|---:|---:|
| 1 | 1263.98 | 1362.69 | 3239.59 | 3005.01 |
| 2 | 1863.33 | 2038.05 | 3310.78 | 3031.36 |
| 4 | 3132.24 | 3659.88 | 3895.02 | 3566.45 |
| 8 | 4449.73 | 5715.56 | 5052.21 | 4485.58 |
| 16 | 6893.32 | 8736.17 | 5907.80 | 5177.53 |
| 32 | 11314.31 | 14281.01 | 6562.49 | 5604.01 |
| 64 | 20502.96 | 25099.77 | 6884.98 | 5849.04 |
| 128 | 38478.58 | 46603.79 | 7075.57 | 5978.56 |

K3 历史来自 2026-10-29 的完整 PP2/TP8 eager 4096/1 单轮测量，使用当时的 QKV 融合和 AITER KDA state。模型、代码和库版本存在差异，上表用于端到端观测对照。两节点现有 LightOP 二进制也存在差异，本轮保留了各节点运行库。最终服务 KV 容量为 146315 token。全服务计数记录 prefix hits=0/4207314、preemptions=0，不能仅根据客户端并发推断 KV 抢占。

## 精度与重新预载复测

HumanEval 全集 164 题，每题一个候选，temperature=0、top_p=1、seed=20261026、max_tokens=4096、thinking 关闭，并发 32。

| 运行 | 通过数 | pass@1 | 精确重复 ` @` 到 4096 token 的题数 | 请求/评分异常 |
|---|---:|---:|---:|---:|
| K3 历史 | 160/164 | 97.56% | 本表未重新统计 | 0 |
| K31 首轮 | 146/164 | 89.02% | 15 | 0 |
| K31 从磁盘重新 preload 后 | 147/164 | 89.63% | 14 | 0 |

首轮 149 stop / 15 length，复测 150 stop / 14 length。复测的配置文件和 164 个逐题请求与首轮一致，权重重新读盘，两轮版本和模型签名一致。

首轮重复输出题号为 36、45、47、53、55、59、65、75、83、85、98、131、138、155、162。复测其中 13 题仍复现，36 与 45 恢复，82 新增重复输出。整体恢复通过 36、45、93，新增失败 67、82。其余代码错误包括 encode 的大小写转换、tri 的越界访问和 order_by_points 的负数排序。尚未定位重复 token 的根因。

## Sparse MLA 瓶颈与 v1 结果

以下 trace 和 TTFT 来自 nmz18 的 TP8/PP1 前 12 层实验，负载为 4096 输入、1 输出、并发 1。

- K3 原生 Attention 主 kernel 三次合计 1.4317 ms，K31 原生 sparse MLA 三次合计 22.5717 ms。K3 使用展开 K/V 的 dense FlashAttention，K31 在压缩 latent KV 上执行 sparse MLA。
- K31 原生 QK 维度为 576，PV latent 维度为 512。本地 12 heads 在原生路径中补齐到 64，原生 kernel 对当前硬件和形状的执行效率是主要瓶颈。
- v1 将 K/V 展开后在 QK=192、V=128 的空间调用 torch.bmm，配合 Triton 融合 scale/mask/softmax，并沿用原 top-k indexer。
- v1 三层 QK/softmax/PV 核心合计 7.713588 ms，包含 KV 投影、mask 和整理的完整 Attention 路径为 10.072907 ms。这两项与原生“主 kernel 三次”统计范围不同。
- v1 仍物化完整 mask、分数矩阵及概率矩阵，约 1.2 GiB/卡中间张量及其读写仍是后续瓶颈。
- 原 trace 中 TopK 三次合计 2.633916 ms，indexer 打分约 1.3194 ms，v1 保留这些实现。
- 同轮部分层端到端 TTFT，K3 为 155.777 ms，K31 原生约 186 ms，v1 为 167.895 ms。
- 已试验的 HIP 融合、迁移和 CK 路径在单测中慢于 expanded 组合路径，尚未替代 v1。部分层结果与完整 PP2/TP8 结果分别记录。

## 原始证据

原始数据位于共享宿主机 `/public/home/xiesl/inference/send_time/task_dir`。

- `tp8_pp2数据整理/TP8_PP2_Benchmark数据整理.md`
- `support_vllm_preload/runs/bench_prefill_4096_20261029`
- `support_vllm_preload/runs/humaneval_qkv_20261029/humaneval/summary.json`
- `speedup_kimi31pro/versions/v1`
- `kimi31pro_pp2tp8_v1_tests/reports/final_report.txt`
- `kimi31pro_pp2tp8_v1_tests/reports/accuracy_final.json`
- `kimi31pro_pp2tp8_v1_tests/retest_01/reports/retest_report.txt`
- `kimi31pro_pp2tp8_v1_tests/retest_01/reports/comparison.json`

性能和精度记录对应提交前的工作区，版本基点为 97be91ed5399d3723139bf7721152837c906d023 及本文保存的改动。构建产物和大体积模型/trace 文件留在任务目录。

## 提交前检查

本次相关现有测试共 142 项通过。隐藏设备运行时 134 项通过，另 8 项因 LightOP 导入需要设备属性而失败；恢复默认设备可见性重跑后 8 项通过。提交快照沿用上文已完成的完整模型验证。
