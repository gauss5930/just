import pandas as pd
import os
import argparse
from score import parse_boxed_value, answer_in_last_sentence, parse_mcqa_value, parse_ksm_value
import json
import yaml
from typing import List

os.makedirs("check_results", exist_ok=True)


def load_config(
        path: str
):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def check_func(checking):
    if checking:
        return "O"
    else:
        return "X"
    

def main(
        subsets: List[str],
        model_list: List[str],
        prompt_id: List[str],
):
    scores = {}

    for pi in prompt_id:
        scores[pi] = {}

        os.makedirs(f"check_results/{pi}", exist_ok=True)

        for model_name in model_list:
            model_path = model_name.replace('/', '_')
            os.makedirs(f"check_results/{pi}/{model_path}", exist_ok=True)
            for subset in subsets:
                df_result = pd.read_csv(os.path.join("results", model_path, pi, f"{subset}.csv"))
                checks = []
                if subset in ["GSM8K", "MATH", "OMNI_MATH"]:
                    score = sum([1 for _,row in df_result.iterrows() if any([answer_in_last_sentence(row.solution,row.answer), parse_boxed_value(row.solution,row.answer)])]) / len(df_result) * 100
                    for _,row in df_result.iterrows():
                        checks.append(check_func(any([answer_in_last_sentence(row.solution, row.answer), parse_boxed_value(row.solution, row.answer)])))
                elif subset == "MMMLU":
                    score = sum([1 for _,row in df_result.iterrows() if any([parse_mcqa_value(row.question,row.solution,row.answer)])]) / len(df_result) * 100
                    for _,row in df_result.iterrows():
                        checks.append(check_func(parse_mcqa_value(row.question, row.solution, row.answer)))
                elif subset == "KSM":
                    score = sum([1 for _,row in df_result.iterrows() if parse_ksm_value(row.question,row.solution,row.answer)]) / len(df_result) * 100
                    for _,row in df_result.iterrows():
                        checks.append(check_func(parse_ksm_value(row.question, row.solution, row.answer)))
                scores[pi][model_name][subset] = score
                df_result["check"] = checks
                df_result.to_csv(f"check_results/{pi}/{model_path}/{subset}_check.csv", index=False)

    with open(f"check_score.json", "w") as f:
        json.dump(scores, f, indent=4)

    print(f'########### {model_name} ###########')
    for k,v in scores[model_name].items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    args = load_config("eval_config.yaml")
    main(args["subsets"], args["models"], args["prompt_id"])