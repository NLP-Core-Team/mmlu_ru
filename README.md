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

|  model      | paper, MMLU EN | MMLU EN, k=5, ctx=2048 | MMLU RU, k=5, ctx=2048 |
| :---------- | -------------: | ---------------------: | ---------------------: |
| Llama 1 7B  | 35.1	         | 36.18	                | 31.65                  |
| Llama 1 13B	| 46.9	         | 48.81	                | 38.03                  |
| Llama 1 33B	| 57.8	         | 59.63	                | 49.06                  |
| Llama 1 65B	| 63.4	         | 65.21	                | 53.96                  |
| Llama 2 7B  | 45.3		       | 47.87		              | 37.86                  |
| Llama 2 13B	| 54.8		       | 56.96		              | 45.29                  |
| Llama 2 34B	| 62.6 	         | unk	                  | unk                    |
| Llama 2 70B	| 68.9           | 71.16		              | 62.86                  |

Please note the scores may slightly vary (vs other evals), but inter-model comparison should be stable.


## Additional Resources

- Original https://github.com/hendrycks/test
- Helpful https://github.com/EleutherAI/lm-evaluation-harness/pull/497
- Also some evals https://github.com/declare-lab/instruct-eval/blob/cc25984f8529db9d5627cec52ed2fba7081d521a/mmlu.py#L219
- Investigation of several evals https://huggingface.co/blog/evaluating-mmlu-leaderboard

## Contributions

Dataset translated and code adopted by NLP core team RnD [Telegram channel](https://t.me/nlpcoreteam)
