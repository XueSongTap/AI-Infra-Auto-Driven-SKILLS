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

## Evidence Rules

- Profiler reports name the exact trace they read and never reuse an older
  trace for a new capture.
- Performance claims are scoped to the exact model, hardware, precision,
  workload, parallelism, and framework revisions.
- Historical upstream evidence keeps its audit date; a source refresh is not a
  GPU rerun.
