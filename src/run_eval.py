import os
import argparse
from score import scoring_func
from generate import generate_solution
from datasets import load_dataset
from typing import List
import pandas as pd
import yaml
import json

os.makedirs('results', exist_ok=True)
os.makedirs('score_results', exist_ok=True)


def load_config(
        path: str
):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main(
        subsets: List[str],
        model_list: List[str],
        prompt_id: List[str],
        reasoning: bool,
        score_type: List[str],
        temperature: float,
        p: float,
        max_tokens: int
):
    dfs = {subset: pd.DataFrame(load_dataset('HAERAE-HUB/HRM8K', subset)['test']) for subset in subsets}

    for pi in prompt_id:
        os.makedirs(f"results/{pi}", exist_ok=True)
        os.makedirs(f"score_results/{pi}", exist_ok=True)

        for model_name in model_list:
            model_path = model_name.replace('/', '_')
            os.makedirs(f"results/{pi}/{model_path}", exist_ok=True)
            os.makedirs(f"score_results/{pi}/{model_path}", exist_ok=True)
            print(f"{model_name} - {prompt_id} Evaluation is starting..")

            results = generate_solution(pi, model_name, reasoning, temperature, p, max_tokens, dfs)
            for k in results.keys():
                results[k].to_csv(f"results/{pi}/{model_path}/{k}.csv", index=False)
                
            scores = scoring_func(score_type, pi, f"results/{pi}/{model_path}")
            print("----------------------- Score Board -----------------------")
            for key in scores.keys():
                print(f"{key}: {scores[key]}")
            print("-----------------------------------------------------------")
            
            with open(f"score_results/{pi}/{model_path}.json", "w") as f:
                json.dump(scores, f, indent=4)


if __name__ == "__main__":
    args = load_config("eval_config.yaml")
    main(args["subsets"], args["models"], args["prompt_id"], args["reasoning"], args["score_type"], args["temperature"], args["top_p"], args["max_tokens"])