# Translate en.md to Chinese — 06 & 07 Phases Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:dispatching-parallel-agents to translate each file independently. Each en.md becomes a zh.md in the same `docs/` directory.

**Goal:** Translate all `en.md` lesson documents in `phases/06-speech-and-audio` and `phases/07-transformers-deep-dive` into Chinese (`zh.md`).

**Architecture:** One subagent per file. Each subagent reads the source `en.md`, translates prose to fluent Chinese while preserving Markdown structure (headings, tables, code blocks, inline code, links, image references), and writes a sibling `zh.md`. No changes to `en.md`.

**Tech Stack:** Markdown; translation performed by LLM subagents.

---

## Translation Rules

- Preserve all Markdown syntax exactly: `#` headings, `**bold**`, `` `inline code` ``, code fences, tables, lists, blockquotes, links, image references.
- Translate prose into natural, professional Chinese suitable for AI engineering learners.
- Keep technical terms clear: common terms like "transformer", "attention", "embedding", "token", "loss", "gradient" may be kept in English or translated with English parenthetical on first use (e.g., “注意力（attention）”).
- Keep code comments inside code blocks in English unless they are explanatory prose; variable names and code identifiers must stay unchanged.
- Keep file paths, command snippets, and URL slugs unchanged.
- Output filename is always `zh.md` in the same directory as the source `en.md`.

---

## File Inventory

### Phase 06 — Speech and Audio

| Source en.md | Target zh.md |
|---|---|
| `phases/06-speech-and-audio/01-audio-fundamentals/docs/en.md` | `phases/06-speech-and-audio/01-audio-fundamentals/docs/zh.md` |
| `phases/06-speech-and-audio/02-spectrograms-mel-features/docs/en.md` | `phases/06-speech-and-audio/02-spectrograms-mel-features/docs/zh.md` |
| `phases/06-speech-and-audio/03-audio-classification/docs/en.md` | `phases/06-speech-and-audio/03-audio-classification/docs/zh.md` |
| `phases/06-speech-and-audio/04-speech-recognition-asr/docs/en.md` | `phases/06-speech-and-audio/04-speech-recognition-asr/docs/zh.md` |
| `phases/06-speech-and-audio/05-whisper-architecture-finetuning/docs/en.md` | `phases/06-speech-and-audio/05-whisper-architecture-finetuning/docs/zh.md` |
| `phases/06-speech-and-audio/06-speaker-recognition-verification/docs/en.md` | `phases/06-speech-and-audio/06-speaker-recognition-verification/docs/zh.md` |
| `phases/06-speech-and-audio/07-text-to-speech/docs/en.md` | `phases/06-speech-and-audio/07-text-to-speech/docs/zh.md` |
| `phases/06-speech-and-audio/08-voice-cloning-conversion/docs/en.md` | `phases/06-speech-and-audio/08-voice-cloning-conversion/docs/zh.md` |
| `phases/06-speech-and-audio/09-music-generation/docs/en.md` | `phases/06-speech-and-audio/09-music-generation/docs/zh.md` |
| `phases/06-speech-and-audio/10-audio-language-models/docs/en.md` | `phases/06-speech-and-audio/10-audio-language-models/docs/zh.md` |
| `phases/06-speech-and-audio/11-real-time-audio-processing/docs/en.md` | `phases/06-speech-and-audio/11-real-time-audio-processing/docs/zh.md` |
| `phases/06-speech-and-audio/12-voice-assistant-pipeline/docs/en.md` | `phases/06-speech-and-audio/12-voice-assistant-pipeline/docs/zh.md` |
| `phases/06-speech-and-audio/13-neural-audio-codecs/docs/en.md` | `phases/06-speech-and-audio/13-neural-audio-codecs/docs/zh.md` |
| `phases/06-speech-and-audio/14-voice-activity-detection-turn-taking/docs/en.md` | `phases/06-speech-and-audio/14-voice-activity-detection-turn-taking/docs/zh.md` |
| `phases/06-speech-and-audio/15-streaming-speech-to-speech-moshi-hibiki/docs/en.md` | `phases/06-speech-and-audio/15-streaming-speech-to-speech-moshi-hibiki/docs/zh.md` |
| `phases/06-speech-and-audio/16-anti-spoofing-audio-watermarking/docs/en.md` | `phases/06-speech-and-audio/16-anti-spoofing-audio-watermarking/docs/zh.md` |
| `phases/06-speech-and-audio/17-audio-evaluation-metrics/docs/en.md` | `phases/06-speech-and-audio/17-audio-evaluation-metrics/docs/zh.md` |

### Phase 07 — Transformers Deep Dive

| Source en.md | Target zh.md |
|---|---|
| `phases/07-transformers-deep-dive/01-why-transformers/docs/en.md` | `phases/07-transformers-deep-dive/01-why-transformers/docs/zh.md` |
| `phases/07-transformers-deep-dive/02-self-attention-from-scratch/docs/en.md` | `phases/07-transformers-deep-dive/02-self-attention-from-scratch/docs/zh.md` |
| `phases/07-transformers-deep-dive/03-multi-head-attention/docs/en.md` | `phases/07-transformers-deep-dive/03-multi-head-attention/docs/zh.md` |
| `phases/07-transformers-deep-dive/04-positional-encoding/docs/en.md` | `phases/07-transformers-deep-dive/04-positional-encoding/docs/zh.md` |
| `phases/07-transformers-deep-dive/05-full-transformer/docs/en.md` | `phases/07-transformers-deep-dive/05-full-transformer/docs/zh.md` |
| `phases/07-transformers-deep-dive/06-bert-masked-language-modeling/docs/en.md` | `phases/07-transformers-deep-dive/06-bert-masked-language-modeling/docs/zh.md` |
| `phases/07-transformers-deep-dive/07-gpt-causal-language-modeling/docs/en.md` | `phases/07-transformers-deep-dive/07-gpt-causal-language-modeling/docs/zh.md` |
| `phases/07-transformers-deep-dive/08-t5-bart-encoder-decoder/docs/en.md` | `phases/07-transformers-deep-dive/08-t5-bart-encoder-decoder/docs/zh.md` |
| `phases/07-transformers-deep-dive/09-vision-transformers/docs/en.md` | `phases/07-transformers-deep-dive/09-vision-transformers/docs/zh.md` |
| `phases/07-transformers-deep-dive/10-audio-transformers-whisper/docs/en.md` | `phases/07-transformers-deep-dive/10-audio-transformers-whisper/docs/zh.md` |
| `phases/07-transformers-deep-dive/11-mixture-of-experts/docs/en.md` | `phases/07-transformers-deep-dive/11-mixture-of-experts/docs/zh.md` |
| `phases/07-transformers-deep-dive/12-kv-cache-flash-attention/docs/en.md` | `phases/07-transformers-deep-dive/12-kv-cache-flash-attention/docs/zh.md` |
| `phases/07-transformers-deep-dive/13-scaling-laws/docs/en.md` | `phases/07-transformers-deep-dive/13-scaling-laws/docs/zh.md` |
| `phases/07-transformers-deep-dive/14-build-a-transformer-capstone/docs/en.md` | `phases/07-transformers-deep-dive/14-build-a-transformer-capstone/docs/zh.md` |
| `phases/07-transformers-deep-dive/15-attention-variants/docs/en.md` | `phases/07-transformers-deep-dive/15-attention-variants/docs/zh.md` |
| `phases/07-transformers-deep-dive/16-speculative-decoding/docs/en.md` | `phases/07-transformers-deep-dive/16-speculative-decoding/docs/zh.md` |

---

## Generic Task: Translate One en.md to zh.md

**Files:**
- Read source: `phases/<phase>/<lesson>/docs/en.md`
- Create target: `phases/<phase>/<lesson>/docs/zh.md`

- [ ] **Step 1: Read the source file**
  Read the entire `en.md` to understand structure and content.

- [ ] **Step 2: Translate while preserving Markdown**
  Produce a Chinese version following the rules above. Do not modify `en.md`.

- [ ] **Step 3: Write zh.md**
  Write the translated content to the sibling `zh.md` with UTF-8 encoding.

- [ ] **Step 4: Verify structure**
  Confirm the number of top-level headings, code blocks, and tables matches the source. Spot-check that no English prose was left untranslated except technical identifiers and code.

---

## Self-Review

1. **Spec coverage:** Every `en.md` in phases 06 and 07 is listed with a corresponding `zh.md` target.
2. **Placeholder scan:** No TODO/TBD; exact paths provided.
3. **Type consistency:** Output naming is consistently `zh.md` in the same directory as the source.
