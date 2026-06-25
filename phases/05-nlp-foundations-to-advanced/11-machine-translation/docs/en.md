# 机器翻译

> 机器翻译是三十年来一直为 NLP 研究买单、并且至今仍在买单的任务。

**类型：** 构建
**语言：** Python
**前置知识：** 第 5 阶段 · 10（注意力机制），第 5 阶段 · 04（GloVe、FastText、Subword）
**时间：** 约 75 分钟

## 问题背景

模型读取一种语言的句子，再生成另一种语言的句子。长度不同。语序不同。有些源词对应多个目标词，反之亦然。习语拒绝逐词映射。法语里的 "I miss you" 是 "tu me manques"，字面意思是“你对我来说缺失了”。没有任何词级对齐能在这种表达下存活。

机器翻译迫使 NLP 发明了编码器-解码器、注意力机制、Transformer，最终催生了整个大语言模型范式。每一次进步都源于翻译质量可衡量、而人机差距又始终顽固存在。

本课跳过历史回顾，直接讲授 2026 年可用的工作流：预训练多语言编码器-解码器模型（NLLB-200 或 mBART）、子词分词、束搜索、BLEU 与 chrF 评估，以及那些仍会未经发现就进入生产环境的少数失效模式。

## 核心概念

![机器翻译流程：分词 → 编码 → 带注意力解码 → 去分词](../assets/mt-pipeline.svg)

现代机器翻译是一个在平行语料上训练的 Transformer 编码器-解码器。编码器读取按源语言分词后的输入；解码器借助交叉注意力（见第 10 课）每次生成一个目标子词。解码时使用束搜索以避免贪心解码陷阱。输出经过去分词、去真实大小写化后，再与参考译文对比打分。

三项实际选择决定了真实场景下的机器翻译质量。

- **分词器。** 在混合语言语料上训练的 SentencePiece BPE。跨语言共享词表正是 NLLB 能够实现零样本语言对的原因。
- **模型规模。** NLLB-200 distilled 600M 可在笔记本上运行。NLLB-200 3.3B 是公开的生产默认版本。54.5B 是研究上限。
- **解码。** 一般内容使用束宽 4-5。使用长度惩罚避免输出过短。需要术语一致性时使用约束解码。

## 动手实现

### 步骤 1：调用预训练机器翻译模型

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_id = "facebook/nllb-200-distilled-600M"
tok = AutoTokenizer.from_pretrained(model_id, src_lang="eng_Latn")
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

src = "The cats are running."
inputs = tok(src, return_tensors="pt")

out = model.generate(
    **inputs,
    forced_bos_token_id=tok.convert_tokens_to_ids("fra_Latn"),
    num_beams=5,
    length_penalty=1.0,
    max_new_tokens=64,
)
print(tok.batch_decode(out, skip_special_tokens=True)[0])
```

```text
Les chats courent.
```

这里有三个关键点。`src_lang` 告诉分词器使用哪种文字与切分方式。`forced_bos_token_id` 告诉解码器生成哪种语言。两者都是 NLLB 特有的技巧；mBART 和 M2M-100 有各自的约定，不能互换。

### 步骤 2：BLEU 与 chrF

BLEU 衡量输出与参考译文之间的 n-gram 重叠。使用 1-4 四种参考 n-gram 尺寸、精确率的几何平均，并对过短输出施加简短惩罚。分数落在 [0, 100] 区间。它被广泛使用，但难以解释：30 BLEU 表示“可用”；40 表示“不错”；50 表示“优秀”；1 BLEU 以内的差异属于噪声。

chrF 衡量字符级 F 分数。在形态丰富的语言中，它比 BLEU 更能捕捉匹配。通常与 BLEU 一起报告。

```python
import sacrebleu

hypotheses = ["Les chats courent."]
references = [["Les chats courent."]]

bleu = sacrebleu.corpus_bleu(hypotheses, references)
chrf = sacrebleu.corpus_chrf(hypotheses, references)
print(f"BLEU: {bleu.score:.1f}  chrF: {chrf.score:.1f}")
```

始终使用 `sacrebleu`。它对 tokenization 做了归一化，使不同论文之间的分数具有可比性。自己实现 BLEU 是导致误导性基准的常见原因。

### 三层评估体系（2026）

现代机器翻译评估使用三类互补的指标族。交付时至少携带其中两类。

- **启发式指标**（BLEU、chrF）。快速、需要参考译文、可解释，但对改写不敏感。用于遗留对比和回归检测。
- **学习型指标**（COMET、BLEURT、BERTScore）。基于人类判断训练的神经网络模型，比较译文与源文、参考译文的语义相似度。自 2023 年以来，COMET 与机器翻译研究的人类相关性最高，也是 2026 年注重质量的生产默认指标。
- **大模型作为裁判**（无需参考译文）。提示大模型从流畅度、充分度、语气、文化适当性等维度为译文打分。在评分标准设计良好的情况下，GPT-4-as-judge 与人类一致率约为 80%。用于没有参考译文的开放式内容。

2026 年的实用组合：`sacrebleu` 负责 BLEU 和 chrF，`unbabel-comet` 负责 COMET，再用一个被提示的大模型作为最终面向人类的信号。在把它用于生产数据之前，先用 50-100 条人工标注样例校准每个指标。

无参考指标（COMET-QE、BLEURT-QE、LLM-as-judge）让你在没有参考译文的情况下评估翻译，这在长尾语言对没有参考译文时尤为重要。

### 步骤 3：生产环境中的失效模式

上面的工作流在 80% 的时间里能流畅翻译，但在剩下 20% 里会静默失败。已命名的失效模式包括：

- **幻觉。** 模型捏造源文中不存在的内容。在不熟悉的领域词汇中很常见。症状：输出流畅，但声称了源文未陈述的事实。缓解措施：对领域术语使用约束解码、对受监管内容进行人工审核、监控输出是否远长于输入。
- **目标语言错误。** 模型翻译成了错误的语言。NLLB 在罕见语言对上 surprisingly 容易出现这种情况。缓解措施：验证 `forced_bos_token_id`，并在输出端始终用语言 ID 模型检查。
- **术语漂移。** “Sign up” 在文档 1 中变成 "s'inscrire"，在文档 2 中又变成 "créer un compte"。对 UI 文本和面向用户的字符串来说，一致性比 raw 质量更重要。缓解措施：使用词汇表约束解码或译后编辑词典。
- **正式程度不匹配。** 法语的 "tu" 与 "vous"、日语的敬体与简体。模型会挑选训练集中更常见的那种。对面向客户的内容而言，这通常是错的。缓解措施：如果模型支持，用正式程度 token 作为提示前缀；或在仅含正式语体的语料上微调一个小模型。
- **短输入长度爆炸。** 非常短的源句往往会生成过长的译文，因为长度惩罚在源句少于约 5 个 token 时几乎失效。缓解措施：设置与源句长度成比例的硬最大长度上限。

### 步骤 4：领域微调

预训练模型是通才。法律、医学或游戏对话翻译如果能在领域平行语料上微调，质量会有可测量的提升。配方并不复杂：

```python
from transformers import Trainer, TrainingArguments
from datasets import Dataset

pairs = [
    {"src": "The defendant pleaded guilty.", "tgt": "L'accusé a plaidé coupable."},
]

ds = Dataset.from_list(pairs)


def preprocess(ex):
    return tok(
        ex["src"],
        text_target=ex["tgt"],
        truncation=True,
        max_length=128,
        padding="max_length",
    )


ds = ds.map(preprocess, remove_columns=["src", "tgt"])

args = TrainingArguments(output_dir="out", per_device_train_batch_size=4, num_train_epochs=3, learning_rate=3e-5)
Trainer(model=model, args=args, train_dataset=ds).train()
```

几千条高质量的平行示例胜过几十万条嘈杂的网络抓取数据。训练数据质量是生产环境中最大的单一杠杆。

## 投入使用

2026 年机器翻译生产栈：

| 使用场景 | 推荐起点 |
|---------|---------------------------|
| 任意语言互译，200 种语言 | `facebook/nllb-200-distilled-600M`（笔记本）或 `nllb-200-3.3B`（生产） |
| 以英语为中心，高质量，50 种语言 | `facebook/mbart-large-50-many-to-many-mmt` |
| 短运行、低成本推理、英-法/德/西 | Helsinki-NLP / Marian 模型 |
| 低延迟浏览器端 | 经 ONNX 量化的 Marian（约 50 MB） |
| 最高质量、愿意付费 | GPT-4 / Claude / Gemini 配合翻译提示 |

截至 2026 年，大语言模型在若干语言对上已经超过了专门的机器翻译模型，尤其在习语内容和长上下文方面。代价是每 token 成本和延迟。当上下文长度、风格一致性或通过提示进行领域适配的重要性高于吞吐量时，选择大语言模型。

## 交付

保存为 `outputs/skill-mt-evaluator.md`：

```markdown
---
name: mt-evaluator
description: 评估一份机器翻译输出是否可交付。
version: 1.0.0
phase: 5
lesson: 11
tags: [nlp, translation, evaluation]
---

给定一段源文本和一份候选译文，输出：

1. 自动分数估算。预期的 BLEU 和 chrF 范围。说明是否有参考译文。
2. 五点人工可核验清单：(a) 内容保留（无幻觉），(b) 语言正确，(c) 语域 / 正式程度匹配，(d) 与所提供词汇表的术语一致性，(e) 无截断或长度爆炸。
3. 一个领域特定问题需要进一步探查。例如法律：命名实体与法规引用；医学：药品名称与剂量；UI：占位符变量 `{name}`。
4. 置信度标志。“交付” / “交付并复核” / “不予交付”。与步骤 2 中发现问题的严重程度挂钩。

若未对输出进行语言 ID 检查，拒绝交付。若用户未明确选择无参考评分（COMET-QE、BLEURT-QE），拒绝在无参考的情况下进行评估。任何超过 1000 token 的内容都应标记为可能需要分块翻译。
```

## 练习

1. **简单。** 使用 `nllb-200-distilled-600M` 将一段 5 句英文段落翻译成法语，再译回英文。测量回译结果与原文的接近程度。你会看到语义被保留，但用词发生了漂移。
2. **中等。** 使用 `fasttext lid.176` 或 `langdetect` 对翻译输出实现语言 ID 检查。将其集成到机器翻译调用流程中，以便在返回前捕获目标语言错误的生成结果。
3. **困难。** 在你自选的 5,000 对领域语料上微调 `nllb-200-distilled-600M`。在留出集上测量微调前后的 BLEU。报告哪些句子改善了，哪些退化了。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|-----------------|-----------------------|
| BLEU | 翻译分数 | 带简短惩罚的 n-gram 精确率。[0, 100]。 |
| chrF | 字符 F 分数 | 字符级 F 分数。对形态丰富的语言更敏感。 |
| NMT | 神经机器翻译 | 在平行语料上训练的 Transformer 编码器-解码器。2017 年后的默认方案。 |
| NLLB | 不让任何语言掉队 | Meta 的 200 种语言机器翻译模型族。 |
| 约束解码 | 受控输出 | 强制输出中出现 / 不出现特定 token 或 n-gram。 |
| 幻觉 | 捏造内容 | 模型输出缺乏源文支持。 |

## 延伸阅读

- [Costa-jussà et al. (2022). No Language Left Behind: Scaling Human-Centered Machine Translation](https://arxiv.org/abs/2207.04672) —— NLLB 论文。
- [Post (2018). A Call for Clarity in Reporting BLEU Scores](https://aclanthology.org/W18-6319/) —— 为何报告 BLEU 时应使用 `sacrebleu`。
- [Popović (2015). chrF: character n-gram F-score for automatic MT evaluation](https://aclanthology.org/W15-3049/) —— chrF 论文。
- [Hugging Face MT guide](https://huggingface.co/docs/transformers/tasks/translation) —— 实用的微调教程。
