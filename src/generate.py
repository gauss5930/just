from prompts import prompts
from models import litellm_models, load_model
from litellm import batch_completion
from tqdm.auto import tqdm
import torch


def safe_parse_litellm(text):
    try:
        return text.choices[0].message.content
    except:
        return None


def safe_parse_vllm(text):
    try:
        return text.outputs[0].text
    except:
        return None


def generate_queries(df, model_name, tokenizer, prompt_id, reasoning):
    qrys = []
    
    for _,row in df.iterrows():
        question = row.original if prompt_id == "en" else row.question
        msg = " ".join([question, prompts[prompt_id]]).strip()

        if model_name not in litellm_models:
            qry = tokenizer.apply_chat_template([{"role": "user", "content": msg}], tokenize=False, add_generation_prompt=True, enable_thinking=reasoning)
        else:
            qry = [{"role": "user", "content": msg}]

        qrys.append(qry)

    return qrys


def generate_solution(prompt_id, model_name, reasoning, temperature, p, max_tokens, dfs):
    if model_name not in litellm_models:
        model, tokenizer, params = load_model(model_name, temperature, p, max_tokens)

    df_results = {}
    for k, df in tqdm(dfs.items(), total=len(dfs)):
        prompts = generate_queries(df, model_name, tokenizer, prompt_id, reasoning)
        if model_name in litellm_models:
            responses = batch_completion(model=model_name, messages=prompts, temperature=temperature, top_p=p, max_tokens=max_tokens)
            outputs = [safe_parse_litellm(resp) for resp in responses]
        else:
            responses = model.generate(prompts, params)
            outputs = [safe_parse_vllm(output) for output in responses]

        df["solution"] = outputs
            
        df_results[k] = df

    del model
    del tokenizer
    torch.cuda.empty_cache()

    return df_results