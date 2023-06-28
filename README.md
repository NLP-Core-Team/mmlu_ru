# MMLU in Russian (Massive Multitask Language Understanding)


## Quickstart

```bash
python3.9 mmlu_ru.py --hf_model_id "huggyllama/llama-7b" --k_shot 5 --lang "ru" --output_dir "results"
```

Possible parameters:
- `lang`: "ru" or "en".
- `k_shot`: 0 to 5.
- `hf_model_id`: huggyllama-LLaMA and IlyaGusev-Saiga series.

It will produce JSONL files with actual prompts and MMLU choices scores ("A"/"B"/"C"/"D"), and CSV files with accuracy scores (with total/category/subcategory granularity).

CPU-only setup not tested.


## Other models

To use with other models, revisit:
- `conversation.py` for suitable Conversation class
- `mmlu_ru.get_prompt_from_dataframes` for any custom (should be ok for both foundation/instruct-tuned LLMs)
- `mmlu_ru.load_model_components` for loading customs settings


## MMLU Dataset

Dataset used: https://huggingface.co/datasets/NLPCoreTeam/mmlu_ru (translated into Russian via Yandex.Translate API).

MMLU dataset covers 57 different tasks. Each task requires to choose the right answer out of four options for a given question. Totally ~14K test samples.


## Evals

Please note the scores may slightly vary (vs other evals), but inter-model comparison should be stable.


## Additional Resources

- Original https://github.com/hendrycks/test
- Helpful https://github.com/EleutherAI/lm-evaluation-harness/pull/497
- Also some evals https://github.com/declare-lab/instruct-eval/blob/cc25984f8529db9d5627cec52ed2fba7081d521a/mmlu.py#L219


## Contributions

Dataset translated and code adopted by NLP core team RnD [Telegram channel](https://t.me/nlpcoreteam)
