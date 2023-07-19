# Modified https://github.com/hendrycks/test
# Helpful https://github.com/EleutherAI/lm-evaluation-harness/pull/497
# Also some evals https://github.com/declare-lab/instruct-eval/blob/cc25984f8529db9d5627cec52ed2fba7081d521a/mmlu.py#L219

import argparse
import json
import logging
import os
import pathlib
import typing as tp

import pandas as pd
import datasets
import peft
import transformers
import torch
from tqdm.auto import tqdm

import categories
import conversation
import stats


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

LANGUAGE_CONFIG: tp.Dict[str, tp.Dict[str, str]] = {
    "en": {
        "headline_prefix": "The following are multiple choice questions (with answers) about",
        "answer_prefix": "Answer:",
    },
    "ru": {
        "headline_prefix": "Ниже приведены вопросы с множественным выбором (с ответами) по",
        "answer_prefix": "Ответ:",
    },
}

def get_df_in_hendrycks_format(subject: str, split: str, lang: str) -> pd.DataFrame:
    dataset = datasets.load_dataset("NLPCoreTeam/mmlu_ru", name=subject, split=split, use_auth_token=True)
    wanted_cols = {
        "en": ["question_en", "choices_en", "answer"],
        "ru": ["question_ru", "choices_ru", "answer"],
    }[lang]
    df = dataset.to_pandas()[wanted_cols]
    int2str = dataset.features["answer"].int2str
    df[df.columns[2]] = df[df.columns[2]].apply(lambda x: int2str(x))
    df = pd.concat([
        df[[df.columns[0]]],
        pd.DataFrame(df[df.columns[1]].tolist()),
        df[[df.columns[2]]],
    ], axis=1)
    df.columns = range(len(df.columns))
    return df

def format_subject(subject: str) -> str:
    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s.strip()

def get_pretty_subject(subject: str, lang: str) -> str:
    return format_subject({
        "en": subject,
        "ru": categories.subcategories_en2ru[subject],  # predefined map
    }[lang])

def get_prompt_from_dataframes(dev_df: pd.DataFrame, test_df: pd.DataFrame,
                               k: int, test_iloc_idx: int, lang: str, subject: str, conversation_type: str):
    assert 0 <= k <= 5
    headline_prefix = LANGUAGE_CONFIG[lang]["headline_prefix"]
    headline_postfix = get_pretty_subject(subject=subject, lang=lang)
    headline = f"{headline_prefix} {headline_postfix}.\n\n"

    answer_prefix = LANGUAGE_CONFIG[lang]["answer_prefix"]
    
    conv = conversation.conversation_classes[conversation_type]()
    
    is_already_taken_headline = False
    for row_idx, row in dev_df.head(k).iterrows():
        q = row[0]
        options = row[1:5].tolist()
        lettered_options = [f"{x}. {y}" for x, y in zip(["A", "B", "C", "D"], options)]
        q_with_lettered_options = "\n".join([q] + lettered_options)
        if row_idx == 0:
            q_with_lettered_options = headline + q_with_lettered_options
            is_already_taken_headline = True
        conv.append_message(conv.roles[0], q_with_lettered_options)
        a = row[5]
        
        # if is not instruct, needed to be manually separated for mmlu examples
        if conv.roles == ("", ""):
            conv.append_message(conv.roles[1], f"\n{answer_prefix} {a}\n\n")
        else:
            conv.append_message(conv.roles[1], f"{answer_prefix} {a}")

    row = test_df.iloc[test_iloc_idx]
    q = row[0]
    options = row[1:5].tolist()
    lettered_options = [f"{x}. {y}" for x, y in zip(["A", "B", "C", "D"], options)]
    q_with_lettered_options = "\n".join([q] + lettered_options)
    if not is_already_taken_headline:
        q_with_lettered_options = headline + q_with_lettered_options
        is_already_taken_headline = True
    conv.append_message(conv.roles[0], q_with_lettered_options)
    a = row[5]
    conv.append_message(conv.roles[1], None)
    
    prompt = f"{conv.get_prompt()}\n{answer_prefix}"
    return prompt

def load_llama_model(model_id: str) -> tp.Tuple:
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_id,
        use_fast=False,
    )
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        load_in_8bit=False,
        device_map="auto",
    )
    model.eval()
    logger.info(f"Model id: {model_id}, params: {model.num_parameters()}, dtype: {model.dtype}")
    return (tokenizer, model, 2048, "llama")

def load_saiga_model(model_id: str) -> tp.Tuple:
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_id,
        use_fast=False,
    )
    config = peft.PeftConfig.from_pretrained(model_id)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        config.base_model_name_or_path,
        load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model = peft.PeftModel.from_pretrained(
        model,
        model_id,
        torch_dtype=torch.float16
    )
    model.eval()
    logger.info(f"Model id: {model_id}, params: {model.num_parameters()}, dtype: {model.dtype}")
    return (tokenizer, model, 2000, "saiga")  # Saiga was trained with 2000


def load_model_components(model_id: str) -> tp.Tuple:
    llama_models = ["huggyllama/llama-7b", "huggyllama/llama-13b", "huggyllama/llama-30b", "huggyllama/llama-65b",
                    "TheBloke/Llama-2-7B-fp16", "TheBloke/Llama-2-13B-fp16", "TheBloke/Llama-2-70B-fp16"]
    saiga_models = ["IlyaGusev/saiga_7b_lora", "IlyaGusev/saiga_13b_lora", "IlyaGusev/saiga_30b_lora", "IlyaGusev/saiga_65b_lora"]

    if model_id in llama_models:
        return load_llama_model(model_id)
    elif model_id in saiga_models:
        return load_saiga_model(model_id)
    else:
        raise Exception(f"Probably not supported: {model_id}.")

def calculate_token_interest_probs(
    input_prompt: str,
    tokenizer: transformers.PreTrainedTokenizerBase,
    model: tp.Union[transformers.PreTrainedModel, peft.peft_model.PeftModelForCausalLM],
) -> tp.Dict[str, float]:
    assert isinstance(input_prompt, str)
    input_ids = tokenizer(input_prompt, return_tensors="pt").input_ids.to(model.device)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids)
    logits = outputs.logits  # shape (batch_size, sequence_length, vocab_size)
    next_token_logits = logits[:, -1, :]  # shape (batch_size, vocab_size)

    next_token_logits = next_token_logits.flatten()
    assert next_token_logits.shape == torch.Size((model.config.vocab_size, ))

    next_token_probs = torch.nn.functional.softmax(next_token_logits, dim=-1).cpu()  # all probs over vocab
    assert torch.isclose(next_token_probs.sum(), torch.tensor(1.0).to(next_token_probs.dtype), atol=1e-03)  # dtype for half/nothalf, -03 for float16
    
    tokens_of_interest = [
        tokenizer("A", add_special_tokens=False).input_ids[-1],
        tokenizer("B", add_special_tokens=False).input_ids[-1],
        tokenizer("C", add_special_tokens=False).input_ids[-1],
        tokenizer("D", add_special_tokens=False).input_ids[-1],
    ]
    
    probs = next_token_probs[tokens_of_interest].tolist()
    res = dict(zip(["A", "B", "C", "D"], probs))
    return res

def append_to_jsonl(data: list, filename: str) -> None:
    with open(filename, "a") as f:
        f.write(json.dumps(data) + "\n")

def evaluate_subject(
    subject: str,
    lang: str,
    k_shot: int,
    jsonl_filepath: str,
    maxlen: int,
    convtype: str,
    tokenizer: transformers.PreTrainedTokenizerBase,
    model: tp.Union[transformers.PreTrainedModel, peft.peft_model.PeftModelForCausalLM],
) -> None:

    dev_df = get_df_in_hendrycks_format(subject=subject, split="dev", lang=lang)
    test_df = get_df_in_hendrycks_format(subject=subject, split="test", lang=lang)

    for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc=subject):

        current_k_shot = k_shot
        skip_too_lengthy = False
        while True:
            input_prompt = get_prompt_from_dataframes(
                dev_df=dev_df,
                test_df=test_df,
                k=current_k_shot,
                test_iloc_idx=idx,
                lang=lang,
                subject=subject,
                conversation_type=convtype,
            )
            input_ids = tokenizer(input_prompt, return_tensors="pt").input_ids.to(model.device)
            if input_ids.shape[-1] > maxlen and current_k_shot >= 0:
                logger.info("Takes smaller current_k_shot since maxlen.")
                current_k_shot -= 1
            elif current_k_shot < 0:
                logger.info("Skip too lengthy.")
                skip_too_lengthy = True
            else:
                break
        if skip_too_lengthy:
            continue

        label = row[5]

        preds = calculate_token_interest_probs(
            input_prompt=input_prompt,
            tokenizer=tokenizer,
            model=model,
        )

        append_to_jsonl(data=[input_prompt, label, preds], filename=jsonl_filepath)


def evaluate_all_subjects(lang: str, k_shot: int, output_dir: str,
                          maxlen: int, convtype: str, tokenizer: transformers.PreTrainedTokenizerBase,
                          model: tp.Union[transformers.PreTrainedModel, peft.peft_model.PeftModelForCausalLM]):
    subjects = list(categories.subcategories.keys())
    for each_subject in subjects:
        jsonl_filepath = str(pathlib.Path(output_dir) / f"{each_subject}.jsonl")
        pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
        logger.info(f"Filepath JSONL: {jsonl_filepath}")
        if pathlib.Path(jsonl_filepath).exists():
            logger.info(f"File already exists! Please manually verify that it wasn't partially interrupted.")
            continue
        evaluate_subject(
            subject=each_subject,
            lang=lang,
            k_shot=k_shot,
            jsonl_filepath=jsonl_filepath,
            maxlen=maxlen, convtype=convtype,
            tokenizer=tokenizer,
            model=model,
        )

def store_results(output_dir: str):
    (subcategories_df, categories_df, total_df) = stats.calculate_accuracy_from_directory(dirpath=output_dir)
    
    subcategories_df.to_csv(str(pathlib.Path(output_dir) / "accuracy_subcategories.csv"), index=False)
    categories_df.to_csv(str(pathlib.Path(output_dir) / "accuracy_categories.csv"), index=False)
    total_df.to_csv(str(pathlib.Path(output_dir) / "accuracy_total.csv"), index=False)
    
    with open(str(pathlib.Path(output_dir) / "args.json"), "w") as f:
        json.dump(vars(args), f)
    
    logger.info(f"Results stored at: {args.output_dir}")


def main(model_id: str, k_shot: int, lang: str, output_dir: str):
    (tokenizer, model, maxlen, convtype) = load_model_components(model_id)
    evaluate_all_subjects(lang, k_shot, output_dir, maxlen, convtype, tokenizer, model)
    store_results(output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_model_id", type=str)
    parser.add_argument("--k_shot", type=int)
    parser.add_argument("--lang", type=str)
    parser.add_argument("--output_dir", type=str)
    args = parser.parse_args()
    main(args.hf_model_id, args.k_shot, args.lang, args.output_dir)
    logger.info("Done!")
