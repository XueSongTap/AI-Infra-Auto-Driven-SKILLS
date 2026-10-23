# AI-Infra-Auto-Driven-SKILLS (training fork)

Agent-ready skills for LLM training performance work: torch profiler trace
triage, layer/kernel breakdowns, Perfetto navigation, operator FLOPs/MFU
estimates, and model architecture diagrams.

This is a trimmed fork of
[BBuf/AI-Infra-Auto-Driven-SKILLS](https://github.com/BBuf/AI-Infra-Auto-Driven-SKILLS).
The upstream repository targets inference serving (SGLang, vLLM, TensorRT-LLM,
TokenSpeed). Serving-only skills, the model PR-history knowledge base, diffusion
flows, prompts, and release notes were removed here; they remain available
upstream. The kept skills are trace- and model-level tools that still carry
serving-oriented examples and will be adapted for training step by step.

## Skills

| Skill | Use it when |
| --- | --- |
| [`llm-torch-profiler-analysis`](skills/llm-torch-profiler-analysis/) | Read a torch profiler trace and get kernel, overlap-opportunity, and fusion-opportunity tables. |
| [`llm-pipeline-analysis`](skills/llm-pipeline-analysis/) | Break a trace into passes, layers, and kernels with anchor boundaries and Perfetto ranges. |
| [`torch-profiler-layer-track`](skills/torch-profiler-layer-track/) | Add verified layer-number guides and compact GPU lanes to a trace for Perfetto navigation. |
| [`model-compute-simulation`](skills/model-compute-simulation/) | Estimate operator shapes, FLOPs, and MFU for a shape, or map kernels back to operators. |
| [`model-architecture-diagram`](skills/model-architecture-diagram/) | Return original public architecture diagrams for popular LLM, VLM, MoE, OCR, and diffusion families. |

## Install

Claude Code plugin:

```text
/plugin marketplace add XueSongTap/AI-Infra-Auto-Driven-SKILLS
/plugin install ai-infra-auto-driven-skills@ai-infra-auto-driven-skills
```

Any skill runtime (Claude Code, Codex, Kimi, ...): link or copy the skill
directories into its skill directory.

```bash
git clone https://github.com/XueSongTap/AI-Infra-Auto-Driven-SKILLS.git
cd AI-Infra-Auto-Driven-SKILLS
SKILL_DIR=~/.claude/skills   # or ${CODEX_HOME:-~/.codex}/skills
mkdir -p "$SKILL_DIR"
for d in skills/*/; do
  [ -f "$d/SKILL.md" ] && ln -sfn "$PWD/${d%/}" "$SKILL_DIR/$(basename "$d")"
done
```

## TODO

### 第一阶段：裁剪收尾

- [ ] 确认 GitHub Actions Lint 通过（本地未跑 isort / black / ruff / lychee）
- [ ] 清理保留 skill 中只对 serving 有意义的段落与脚本，例如
      `llm-torch-profiler-analysis` 的 `run_sglang_torch_profile_host.sh`、
      `run_vllm_torch_profile_host.sh`、`probe_llm_server.py`
- [ ] 精简 `fuse-overlap-catalog.md` / `overlap-catalog.md` /
      `vllm-torch-compile-fusions.md`，只保留训练中也会出现的 kernel（GEMM、
      attention、norm、MoE dispatch/combine、通信）
- [ ] 更新 `docs/upstream-source-contracts.md`，改为记录训练框架的源码版本
- [ ] 决定是否改名：仓库名、plugin 名、各 skill 名（去掉 `llm-serving`、
      `sglang` 等前缀）

### 第二阶段：现有 skill 改造为训练版

- [ ] `llm-torch-profiler-analysis`
  - [ ] 训练 trace 抓取脚本：PyTorch 原生、Megatron-LM、FSDP2、DeepSpeed，
        支持 `torch.profiler.schedule` 选取稳态 step 与多 rank 抓取
  - [ ] 按 forward / backward / optimizer step / 梯度通信划分阶段
  - [ ] 通信与计算 overlap 分析：DP all-reduce / reduce-scatter、FSDP
        all-gather 预取、TP/SP 集合通信、EP all-to-all、PP send/recv
  - [ ] 训练侧 fusion 目录：fused Adam、fused cross-entropy、activation
        recompute 开销、FP8 cast / amax 等
- [ ] `llm-pipeline-analysis`
  - [ ] 层边界识别覆盖 backward（反序层、autograd 节点）和 recompute 重算段
  - [ ] Pipeline Parallel：1F1B / interleaved 调度下的 micro-batch 划分与
        bubble 统计
  - [ ] `model_profiles.py` 增加训练相关的 anchor kernel
- [ ] `torch-profiler-layer-track`
  - [ ] backward 层号标注（反向顺序）与 forward/backward 分色
  - [ ] 多 rank trace 合并后的 lane 压缩
- [ ] `model-compute-simulation`
  - [ ] 训练 FLOPs：forward + backward（≈ 3× forward，约 6N·tokens），
        区分开启 / 关闭 activation recompute
  - [ ] 训练 MFU / HFU（按 step time 与 global batch tokens 计算）
  - [ ] 并行维度 what-if：DP / TP / PP / CP / EP / ZeRO stage
  - [ ] 显存估算：参数、梯度、优化器状态、激活（含 recompute 与 offload）
- [ ] `model-architecture-diagram`：基本通用，补充训练常见模型条目即可

### 第三阶段：新增训练 skill

- [ ] 训练吞吐基准与配置搜索（对应上游 `llm-serving-auto-benchmark`）：
      同模型、同硬件下对比并行策略与框架参数，记录 tokens/s/GPU、MFU、显存
- [ ] 训练显存规划与 OOM 分析（对应上游 `llm-serving-capacity-planner`）
- [ ] 训练故障排查：NCCL hang / timeout、loss spike / NaN、慢节点、
      checkpoint 保存与加载失败（对应上游 `sglang-prod-incident-triage`）
- [ ] 训练框架 PR 优化历史库（Megatron-LM、TorchTitan、DeepSpeed 等），
      可复用上游 `tools/rebuild_model_pr_history_from_git.py` 的思路
- [ ] 每个新 skill 配套离线测试（放在 `tests/`）

## Evidence Rules

- Profiler reports name the exact trace they read and never reuse an older
  trace for a new capture.
- Performance claims are scoped to the exact model, hardware, precision,
  workload, parallelism, and framework revisions.
- Historical upstream evidence keeps its audit date; a source refresh is not a
  GPU rerun.
