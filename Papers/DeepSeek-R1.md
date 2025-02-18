# DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning

DeepSeek-R1：通过强化学习激励 LLMs 中的推理能力

DeepSeek-AI

research@deepseek.com

Abstract

We introduce our first-generation reasoning models, DeepSeek-R1-Zero and DeepSeek-R1. DeepSeek-R1-Zero, a model trained via large-scale reinforcement learning (RL) without supervised fine-tuning (SFT) as a preliminary step, demonstrates remarkable reasoning capabilities. Through RL, DeepSeek-R1-Zero naturally emerges with numerous powerful and intriguing reasoning behaviors. However, it encounters challenges such as poor readability, and language mixing. To address these issues and further enhance reasoning performance, we introduce DeepSeek-R1, which incorporates multi-stage training and cold-start data before RL. DeepSeek-R1 achieves performance comparable to OpenAI-o1-1217 on reasoning tasks. To support the research community, we open-source DeepSeek-R1-Zero, DeepSeek-R1, and six dense models (1.5B, 7B, 8B, 14B, 32B, 70B) distilled from DeepSeek-R1 based on Qwen and Llama.

我们推出了第一代推理模型——DeepSeek-R1-Zero 和 DeepSeek-R1。DeepSeek-R1-Zero 是一个通过大规模强化学习（RL）训练而成的模型，无需监督微调（SFT）作为初步步骤，展现了卓越的推理能力。通过 RL，DeepSeek-R1-Zero 自然涌现出众多强大且引人入胜的推理行为。然而，它也面临诸如可读性差和语言混杂等挑战。为解决这些问题并进一步提升推理性能，我们引入了 DeepSeek-R1，它在 RL 之前融入了多阶段训练和冷启动数据。DeepSeek-R1 在推理任务上的表现可与 OpenAI-o1-1217 相媲美。为支持研究社区，我们开源了 DeepSeek-R1-Zero、DeepSeek-R1 以及基于 Qwen 和 Llama 从 DeepSeek-R1 蒸馏出的六个密集模型（1.5B、7B、8B、14B、32B、70B）。

Refer to caption

Figure 1: Benchmark performance of DeepSeek-R1.

## 1 Introduction

In recent years, Large Language Models (LLMs) have been undergoing rapid iteration and evolution (OpenAI, 2024a; Anthropic, 2024; Google, 2024), progressively diminishing the gap towards Artificial General Intelligence (AGI).

近年来，大型语言模型（LLMs）经历了快速的迭代与进化（OpenAI, 2024a; Anthropic, 2024; Google, 2024），逐步缩小了与人工通用智能（AGI）之间的差距。

Recently, post-training has emerged as an important component of the full training pipeline. It has been shown to enhance accuracy on reasoning tasks, align with social values, and adapt to user preferences, all while requiring relatively minimal computational resources against pre-training. In the context of reasoning capabilities, OpenAI’s o1 (OpenAI, 2024b) series models were the first to introduce inference-time scaling by increasing the length of the Chain-of-Thought reasoning process. This approach has achieved significant improvements in various reasoning tasks, such as mathematics, coding, and scientific reasoning. However, the challenge of effective test-time scaling remains an open question for the research community. Several prior works have explored various approaches, including process-based reward models (Uesato et al., 2022; Lightman et al., 2023; Wang et al., 2023), reinforcement learning (Kumar et al., 2024), and search algorithms such as Monte Carlo Tree Search and Beam Search (Feng et al., 2024; Xin et al., 2024; Trinh et al., 2024). However, none of these methods has achieved general reasoning performance comparable to OpenAI’s o1 series models.

最近，后训练已成为完整训练流程中的一个重要组成部分。它被证明能够提高推理任务的准确性、与社会价值观保持一致，并适应用户偏好，同时相对于预训练所需的计算资源相对较少。在推理能力方面，OpenAI 的 o1 系列模型（OpenAI, 2024b）首次通过增加思维链推理过程的长度引入了推理时扩展。这种方法在数学、编程和科学推理等各种推理任务中取得了显著改进。然而，有效的测试时扩展仍然是研究界面临的一个开放性问题。之前的一些研究探索了多种方法，包括基于过程的奖励模型（Uesato 等，2022；Lightman 等，2023；Wang 等，2023）、强化学习（Kumar 等，2024）以及蒙特卡洛树搜索和束搜索等搜索算法（Feng 等，2024；Xin 等，2024；Trinh 等，2024）。然而，这些方法均未达到与 OpenAI 的 o1 系列模型相媲美的通用推理性能。

In this paper, we take the first step toward improving language model reasoning capabilities using pure reinforcement learning (RL). Our goal is to explore the potential of LLMs to develop reasoning capabilities without any supervised data, focusing on their self-evolution through a pure RL process. Specifically, we use DeepSeek-V3-Base as the base model and employ GRPO (Shao et al., 2024) as the RL framework to improve model performance in reasoning. During training, DeepSeek-R1-Zero naturally emerged with numerous powerful and interesting reasoning behaviors. After thousands of RL steps, DeepSeek-R1-Zero exhibits super performance on reasoning benchmarks. For instance, the pass@1 score on AIME 2024 increases from 15.6% to 71.0%, and with majority voting, the score further improves to 86.7%, matching the performance of OpenAI-o1-0912.

本文中，我们迈出了利用纯强化学习（RL）提升语言模型推理能力的第一步。我们的目标是探索 LLMs 在无任何监督数据的情况下发展推理能力的潜力，重点在于通过纯 RL 过程实现其自我进化。具体而言，我们采用 DeepSeek-V3-Base 作为基础模型，并运用 GRPO（Shao 等人，2024 年）作为 RL 框架，以增强模型在推理任务上的表现。训练过程中，DeepSeek-R1-Zero 自然涌现出众多强大且有趣的推理行为。经过数千步 RL 训练后，DeepSeek-R1-Zero 在推理基准测试中展现出卓越性能。例如，在 AIME 2024 上的 pass@1 得分从 15.6%提升至 71.0%，而采用多数投票后，得分进一步跃升至 86.7%，与 OpenAI-o1-0912 的表现相当。

However, DeepSeek-R1-Zero encounters challenges such as poor readability, and language mixing. To address these issues and further enhance reasoning performance, we introduce DeepSeek-R1, which incorporates a small amount of cold-start data and a multi-stage training pipeline. Specifically, we begin by collecting thousands of cold-start data to fine-tune the DeepSeek-V3-Base model. Following this, we perform reasoning-oriented RL like DeepSeek-R1-Zero. Upon nearing convergence in the RL process, we create new SFT data through rejection sampling on the RL checkpoint, combined with supervised data from DeepSeek-V3 in domains such as writing, factual QA, and self-cognition, and then retrain the DeepSeek-V3-Base model. After fine-tuning with the new data, the checkpoint undergoes an additional RL process, taking into account prompts from all scenarios. After these steps, we obtained a checkpoint referred to as DeepSeek-R1, which achieves performance on par with OpenAI-o1-1217.

然而，DeepSeek-R1-Zero 面临诸如可读性差和语言混杂等挑战。为解决这些问题并进一步提升推理性能，我们引入了 DeepSeek-R1，它融合了少量冷启动数据和一个多阶段训练流程。具体而言，我们首先收集数千条冷启动数据来微调 DeepSeek-V3-Base 模型。随后，我们执行类似 DeepSeek-R1-Zero 的推理导向强化学习（RL）。在 RL 过程接近收敛时，我们通过对 RL 检查点进行拒绝采样，结合来自 DeepSeek-V3 在写作、事实问答和自我认知等领域的监督数据，创建新的 SFT 数据，并重新训练 DeepSeek-V3-Base 模型。使用新数据微调后，检查点还需经历一个额外的 RL 过程，考虑所有场景的提示。经过这些步骤，我们获得了称为 DeepSeek-R1 的检查点，其性能与 OpenAI-o1-1217 相当。

We further explore distillation from DeepSeek-R1 to smaller dense models. Using Qwen2.5-32B (Qwen, 2024b) as the base model, direct distillation from DeepSeek-R1 outperforms applying RL on it. This demonstrates that the reasoning patterns discovered by larger base models are crucial for improving reasoning capabilities. We open-source the distilled Qwen and Llama (Dubey et al., 2024) series. Notably, our distilled 14B model outperforms state-of-the-art open-source QwQ-32B-Preview (Qwen, 2024a) by a large margin, and the distilled 32B and 70B models set a new record on the reasoning benchmarks among dense models.

我们进一步探索了从 DeepSeek-R1 到更小密集模型的蒸馏过程。以 Qwen2.5-32B（Qwen, 2024b）为基础模型，直接从 DeepSeek-R1 进行蒸馏的效果优于在其上应用强化学习。这表明，更大基础模型所发现的推理模式对于提升推理能力至关重要。我们开源了蒸馏后的 Qwen 和 Llama（Dubey 等，2024）系列。值得注意的是，我们蒸馏的 14B 模型大幅超越了当前最先进的开源 QwQ-32B-Preview（Qwen, 2024a），而蒸馏的 32B 和 70B 模型在密集模型的推理基准测试中创下了新纪录。

### 1.1 Contributions

##### Post-Training: Large-Scale Reinforcement Learning on the Base Model

训练后：基于模型的大规模强化学习

We directly apply RL to the base model without relying on supervised fine-tuning (SFT) as a preliminary step. This approach allows the model to explore chain-of-thought (CoT) for solving complex problems, resulting in the development of DeepSeek-R1-Zero. DeepSeek-R1-Zero demonstrates capabilities such as self-verification, reflection, and generating long CoTs, marking a significant milestone for the research community. Notably, it is the first open research to validate that reasoning capabilities of LLMs can be incentivized purely through RL, without the need for SFT. This breakthrough paves the way for future advancements in this area.

我们直接将强化学习（RL）应用于基础模型，无需依赖监督微调（SFT）作为初步步骤。这种方法使模型能够探索思维链（CoT）以解决复杂问题，从而开发出 DeepSeek-R1-Zero。DeepSeek-R1-Zero 展示了自我验证、反思和生成长思维链等能力，标志着研究界的一个重要里程碑。值得注意的是，这是首次公开研究验证 LLMs 的推理能力可以仅通过 RL 激励，而无需 SFT。这一突破为该领域的未来进展铺平了道路。

We introduce our pipeline to develop DeepSeek-R1. The pipeline incorporates two RL stages aimed at discovering improved reasoning patterns and aligning with human preferences, as well as two SFT stages that serve as the seed for the model’s reasoning and non-reasoning capabilities. We believe the pipeline will benefit the industry by creating better models.

我们介绍了开发 DeepSeek-R1 的流程。该流程包含两个强化学习阶段，旨在发现改进的推理模式并与人类偏好对齐，以及两个监督微调阶段，作为模型推理和非推理能力的基础。我们相信这一流程将通过创建更好的模型为行业带来益处。

##### Distillation: Smaller Models Can Be Powerful Too

蒸馏：小型模型同样强大

We demonstrate that the reasoning patterns of larger models can be distilled into smaller models, resulting in better performance compared to the reasoning patterns discovered through RL on small models. The open source DeepSeek-R1, as well as its API, will benefit the research community to distill better smaller models in the future.

我们证明了较大模型的推理模式可以被提炼到较小模型中，与通过强化学习在小模型上发现的推理模式相比，能带来更好的性能表现。开源的 DeepSeek-R1 及其 API，将有助于研究社区未来提炼出更优的小型模型。

Using the reasoning data generated by DeepSeek-R1, we fine-tuned several dense models that are widely used in the research community. The evaluation results demonstrate that the distilled smaller dense models perform exceptionally well on benchmarks. DeepSeek-R1-Distill-Qwen-7B achieves 55.5% on AIME 2024, surpassing QwQ-32B-Preview. Additionally, DeepSeek-R1-Distill-Qwen-32B scores 72.6% on AIME 2024, 94.3% on MATH-500, and 57.2% on LiveCodeBench. These results significantly outperform previous open-source models and are comparable to o1-mini. We open-source distilled 1.5B, 7B, 8B, 14B, 32B, and 70B checkpoints based on Qwen2.5 and Llama3 series to the community.

利用 DeepSeek-R1 生成的推理数据，我们对研究界广泛使用的多个密集模型进行了微调。评估结果显示，经过蒸馏的小型密集模型在基准测试中表现尤为出色。DeepSeek-R1-Distill-Qwen-7B 在 AIME 2024 上达到了 55.5%的成绩，超越了 QwQ-32B-Preview。此外，DeepSeek-R1-Distill-Qwen-32B 在 AIME 2024 上获得了 72.6%的分数，在 MATH-500 上为 94.3%，在 LiveCodeBench 上为 57.2%。这些成绩显著超越了之前的开源模型，并与 o1-mini 相媲美。我们向社区开源了基于 Qwen2.5 和 Llama3 系列的 1.5B、7B、8B、14B、32B 及 70B 蒸馏检查点。

### 1.2 Summary of Evaluation Results

Reasoning tasks: (1) DeepSeek-R1 achieves a score of 79.8% Pass@1 on AIME 2024, slightly surpassing OpenAI-o1-1217. On MATH-500, it attains an impressive score of 97.3%, performing on par with OpenAI-o1-1217 and significantly outperforming other models. (2) On coding-related tasks, DeepSeek-R1 demonstrates expert level in code competition tasks, as it achieves 2,029 Elo rating on Codeforces outperforming 96.3% human participants in the competition. For engineering-related tasks, DeepSeek-R1 performs slightly better than DeepSeek-V3, which could help developers in real world tasks.

推理任务：（1）DeepSeek-R1 在 AIME 2024 上获得了 79.8%的 Pass@1 分数，略高于 OpenAI-o1-1217。在 MATH-500 上，它取得了令人印象深刻的 97.3%的分数，与 OpenAI-o1-1217 表现相当，并显著优于其他模型。（2）在编码相关任务上，DeepSeek-R1 展示了代码竞赛任务中的专家水平，它在 Codeforces 上获得了 2,029 的 Elo 评分，超过了 96.3%的人类参赛者。对于工程相关任务，DeepSeek-R1 的表现略优于 DeepSeek-V3，这可能有助于开发者在实际任务中的应用。

Knowledge: On benchmarks such as MMLU, MMLU-Pro, and GPQA Diamond, DeepSeek-R1 achieves outstanding results, significantly outperforming DeepSeek-V3 with scores of 90.8% on MMLU, 84.0% on MMLU-Pro, and 71.5% on GPQA Diamond. While its performance is slightly below that of OpenAI-o1-1217 on these benchmarks, DeepSeek-R1 surpasses other closed-source models, demonstrating its competitive edge in educational tasks. On the factual benchmark SimpleQA, DeepSeek-R1 outperforms DeepSeek-V3, demonstrating its capability in handling fact-based queries. A similar trend is observed where OpenAI-o1 surpasses 4o on this benchmark.

知识：在 MMLU、MMLU-Pro 和 GPQA Diamond 等基准测试中，DeepSeek-R1 取得了卓越的成绩，显著超越了 DeepSeek-V3，得分分别为 MMLU 的 90.8%、MMLU-Pro 的 84.0%和 GPQA Diamond 的 71.5%。尽管在这些基准测试中其表现略逊于 OpenAI-o1-1217，但 DeepSeek-R1 超越了其他闭源模型，展示了其在教育任务中的竞争优势。在事实基准测试 SimpleQA 上，DeepSeek-R1 的表现优于 DeepSeek-V3，证明了其处理基于事实查询的能力。类似趋势也体现在 OpenAI-o1 在该基准测试上超越 4o 的情况。

Others: DeepSeek-R1 also excels in a wide range of tasks, including creative writing, general question answering, editing, summarization, and more. It achieves an impressive length-controlled win-rate of 87.6% on AlpacaEval 2.0 and a win-rate of 92.3% on ArenaHard, showcasing its strong ability to intelligently handle non-exam-oriented queries. Additionally, DeepSeek-R1 demonstrates outstanding performance on tasks requiring long-context understanding, substantially outperforming DeepSeek-V3 on long-context benchmarks.

其他方面：DeepSeek-R1 在创意写作、通用问答、编辑、摘要等多种任务上同样表现出色。它在 AlpacaEval 2.0 上实现了 87.6%的长度控制胜率，在 ArenaHard 上的胜率更是高达 92.3%，充分展现了其智能处理非应试类查询的强大能力。此外，DeepSeek-R1 在需要长上下文理解的任务上表现卓越，在长上下文基准测试中大幅超越 DeepSeek-V3。

## 2 Approach

### 2.1 Overview

Previous work has heavily relied on large amounts of supervised data to enhance model performance. In this study, we demonstrate that reasoning capabilities can be significantly improved through large-scale reinforcement learning (RL), even without using supervised fine-tuning (SFT) as a cold start. Furthermore, performance can be further enhanced with the inclusion of a small amount of cold-start data. In the following sections, we present: (1) DeepSeek-R1-Zero, which applies RL directly to the base model without any SFT data, and (2) DeepSeek-R1, which applies RL starting from a checkpoint fine-tuned with thousands of long Chain-of-Thought (CoT) examples. 3) Distill the reasoning capability from DeepSeek-R1 to small dense models.

先前的研究在很大程度上依赖于大量监督数据来提升模型性能。本研究中，我们展示了即使不使用监督微调（SFT）作为冷启动，通过大规模强化学习（RL）也能显著提升推理能力。此外，加入少量冷启动数据可进一步优化性能。接下来，我们将介绍：（1）DeepSeek-R1-Zero，它直接将 RL 应用于基础模型，无需任何 SFT 数据；（2）DeepSeek-R1，它从经过数千个长链思维（CoT）示例微调的检查点开始应用 RL；（3）将 DeepSeek-R1 的推理能力蒸馏到小型密集模型中。

### 2.2 DeepSeek-R1-Zero: Reinforcement Learning on the Base Model

Reinforcement learning has demonstrated significant effectiveness in reasoning tasks, as evidenced by our previous works(Wang et al.,[2023](https://arxiv.org/html/2501.12948v1#bib.bib34); Shao et al.,[2024](https://arxiv.org/html/2501.12948v1#bib.bib28)). However, these works heavily depended on supervised data, which are time-intensive to gather. In this section, we explore the potential of LLMs to develop reasoning capabilitieswithout any supervised data, focusing on their self-evolution through a pure reinforcement learning process. We start with a brief overview of our RL algorithm, followed by the presentation of some exciting results, and hope this provides the community with valuable insights.

强化学习在推理任务中已展现出显著成效，正如我们先前的研究所证实（Wang 等，2023；Shao 等，2024）。然而，这些研究高度依赖于监督数据，而收集这些数据耗时巨大。本节中，我们探讨了 LLMs 在无任何监督数据情况下发展推理能力的潜力，重点在于其通过纯粹的强化学习过程实现自我进化。我们首先简要概述了我们的强化学习算法，随后展示了一些令人振奋的结果，希望这能为社区提供宝贵的洞见。
