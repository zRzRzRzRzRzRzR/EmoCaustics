import argparse
import os
import re
import json
import asyncio
import numpy as np
from tqdm import tqdm
from openai import OpenAI
from utils import load_yaml_config, load_json, call_embedding, cosine_similarity

similarity_judge_prompt = """
你是一个负责对比两个文本的评测器，我会给你两个文本A和B，你需要判断它们语义是否相似，或者在描述一个相似的态度和事件。
如果语义基本一致或高度近似(意思相或者目标相同)，请输出1，否则输出0。只需要输出一个数字0或1，不要输出其他解释。
"""


async def judge_similarity_with_llm(text_a, text_b, judge_config):
    """
    Use the LLM specified by judge_config to determine whether text_a and text_b are similar, returning 0 or 1.
    judge_config: { “model”: “…”, “base_url”: “…”, “api_key”: “…” }
    """
    messages = [
        {"role": "system", "content": similarity_judge_prompt},
        {
            "role": "user",
            "content": f"文本A: {text_a}\n文本B: {text_b}\n请判断是否相似(0或1):",
        },
    ]

    client = OpenAI(
        api_key=judge_config["api_key"],
        base_url=judge_config["base_url"],
    )
    try:
        response = await asyncio.to_thread(
            client.chat.completions.create, model=judge_config["model"], messages=messages, temperature=0.0
        )
        content = response.choices[0].message.content.strip()
        val = int(content)
        return 1 if val == 1 else 0
    except:
        return 0


async def evaluate_single_prediction(
    gt_target,
    gt_aspect,
    gt_opinion,
    gt_sentiment,
    gt_rationale,
    pred_target,
    pred_aspect,
    pred_opinion,
    pred_sentiment,
    pred_rationale,
    embed_config,
):
    """
    计算scheme1(向量相似度)分数 + sentiment是否匹配
    """
    texts_to_embed = [
        gt_target,
        pred_target,
        gt_aspect,
        pred_aspect,
        gt_opinion,
        pred_opinion,
        gt_rationale,
        pred_rationale,
    ]
    texts_to_embed = [t if t else " " for t in texts_to_embed]
    embeddings = call_embedding(texts_to_embed, **embed_config)
    gt_target_emb = embeddings[0]
    pred_target_emb = embeddings[1]
    gt_aspect_emb = embeddings[2]
    pred_aspect_emb = embeddings[3]
    gt_opinion_emb = embeddings[4]
    pred_opinion_emb = embeddings[5]
    gt_rationale_emb = embeddings[6]
    pred_rationale_emb = embeddings[7]

    target_sim_1 = cosine_similarity(gt_target_emb, pred_target_emb)
    aspect_sim_1 = cosine_similarity(gt_aspect_emb, pred_aspect_emb)
    opinion_sim_1 = cosine_similarity(gt_opinion_emb, pred_opinion_emb)
    rationale_sim_1 = cosine_similarity(gt_rationale_emb, pred_rationale_emb)
    sentiment_correct_1 = 1 if gt_sentiment == pred_sentiment else 0

    return {
        "target_similarity": target_sim_1,
        "aspect_similarity": aspect_sim_1,
        "opinion_similarity": opinion_sim_1,
        "rationale_similarity": rationale_sim_1,
        "sentiment_correct": sentiment_correct_1,
    }


def select_best_prediction(all_preds_scores):
    """
    Select the one with the highest overall score from multiple candidate quintuplets.
    """
    if not all_preds_scores:
        return {
            "target_similarity": 0.0,
            "aspect_similarity": 0.0,
            "opinion_similarity": 0.0,
            "rationale_similarity": 0.0,
            "sentiment_correct": 0,
            "pred_target": "",
            "pred_aspect": "",
            "pred_opinion": "",
            "pred_sentiment": "",
            "pred_rationale": "",
        }
    best = sorted(
        all_preds_scores,
        key=lambda x: (
            x["target_similarity"],
            x["aspect_similarity"],
            x["sentiment_correct"],
            x["opinion_similarity"],
            x["rationale_similarity"],
        ),
        reverse=True,
    )[0]
    return best


async def evaluate_line(pred_data, gt_data, i, embed_config, judge_config):
    """
    - Step 1: Cosine similarity + sentiment correctness
        - Step 2: LLM similarity determination (0/1) + sentiment correctness
    """
    pred_item = pred_data[i]
    gt_item = gt_data[i]

    gt_target = gt_item.get("Target", "")
    gt_aspect = gt_item.get("Aspect", "")
    gt_opinion = gt_item.get("Opinion", "")
    gt_sentiment = gt_item.get("Sentiment", "")
    gt_rationale = gt_item.get("Rationale", "")
    pred_resp = pred_item.get("final_model_response", {})

    if isinstance(pred_resp, dict):
        pred_resps_list = [pred_resp]
    elif isinstance(pred_resp, list):
        pred_resps_list = pred_resp
    else:
        pred_resps_list = []

    if not pred_resps_list:
        pred_resps_list = [
            {
                "target": "",
                "aspect": "",
                "opinions": "",
                "sentiment": "",
                "rationale": "",
            }
        ]

    candidates = []
    for p in pred_resps_list:
        pred_target = p.get("target", p.get("Target", ""))
        pred_aspect = p.get("aspect", p.get("Aspect", ""))
        pred_opinion = p.get("opinion", p.get("opinions", ""))
        pred_sentiment = p.get("sentiment", p.get("Sentiment", ""))
        pred_rationale = p.get("rationale", p.get("Rationale", ""))
        scheme1_scores = await evaluate_single_prediction(
            gt_target,
            gt_aspect,
            gt_opinion,
            gt_sentiment,
            gt_rationale,
            pred_target,
            pred_aspect,
            pred_opinion,
            pred_sentiment,
            pred_rationale,
            embed_config,
        )
        candidate = {
            **scheme1_scores,
            "pred_target": pred_target,
            "pred_aspect": pred_aspect,
            "pred_opinion": pred_opinion,
            "pred_sentiment": pred_sentiment,
            "pred_rationale": pred_rationale,
        }
        candidates.append(candidate)

    best_pred = select_best_prediction(candidates)

    target_sim_2 = await judge_similarity_with_llm(gt_target, best_pred["pred_target"], judge_config)
    aspect_sim_2 = await judge_similarity_with_llm(gt_aspect, best_pred["pred_aspect"], judge_config)
    opinion_sim_2 = await judge_similarity_with_llm(gt_opinion, best_pred["pred_opinion"], judge_config)
    rationale_sim_2 = await judge_similarity_with_llm(gt_rationale, best_pred["pred_rationale"], judge_config)
    sentiment_correct_2 = 1 if gt_sentiment == best_pred["pred_sentiment"] else 0

    result_item = {
        "file": os.path.basename("dummy"),  # 以后可再补充真实文件名
        "index": i,
        "chosen_prediction": {
            "target": best_pred["pred_target"],
            "aspect": best_pred["pred_aspect"],
            "opinion": best_pred["pred_opinion"],
            "sentiment": best_pred["pred_sentiment"],
            "rationale": best_pred["pred_rationale"],
        },
        "scheme1": {
            "target_similarity": best_pred["target_similarity"],
            "aspect_similarity": best_pred["aspect_similarity"],
            "opinion_similarity": best_pred["opinion_similarity"],
            "rationale_similarity": best_pred["rationale_similarity"],
            "sentiment_correct": best_pred["sentiment_correct"],
        },
        "scheme2": {
            "target_similarity": target_sim_2,
            "aspect_similarity": aspect_sim_2,
            "opinion_similarity": opinion_sim_2,
            "rationale_similarity": rationale_sim_2,
            "sentiment_correct": sentiment_correct_2,
        },
    }
    return result_item


async def check_and_fix_suspicious_lines(out_file_path, pred_file_path, gt_file_path, embed_config, judge_config):
    """
    读取评估文件，检测可疑行（全部0分），重新计算并更新保存
    """
    if not os.path.exists(out_file_path):
        return

    data = load_json(out_file_path)
    details = data.get("details", [])
    suspicious_indices = []
    for idx, d in enumerate(details):
        s1 = d.get("scheme1", {})
        if (
            s1.get("target_similarity", 0.0) == 0.0
            and s1.get("aspect_similarity", 0.0) == 0.0
            and s1.get("opinion_similarity", 0.0) == 0.0
            and s1.get("rationale_similarity", 0.0) == 0.0
            and s1.get("sentiment_correct", 0) == 0
        ):
            suspicious_indices.append(idx)

    if not suspicious_indices:
        return

    pred_data = load_json(pred_file_path)
    gt_data = load_json(gt_file_path)

    for idx in suspicious_indices:
        line_idx = details[idx]["index"]
        new_result_item = await evaluate_line(pred_data, gt_data, line_idx, embed_config, judge_config)
        # 保留原先的 file 字段
        new_result_item["file"] = details[idx]["file"]
        details[idx] = new_result_item

    # 重新计算 overall averages
    scheme1_target_sims = []
    scheme1_aspect_sims = []
    scheme1_opinion_sims = []
    scheme1_rationale_sims = []
    scheme1_sentiment_correct = []

    scheme2_target_sims = []
    scheme2_aspect_sims = []
    scheme2_opinion_sims = []
    scheme2_rationale_sims = []
    scheme2_sentiment_correct = []

    for d in details:
        s1 = d["scheme1"]
        s2 = d["scheme2"]
        scheme1_target_sims.append(s1["target_similarity"])
        scheme1_aspect_sims.append(s1["aspect_similarity"])
        scheme1_opinion_sims.append(s1["opinion_similarity"])
        scheme1_rationale_sims.append(s1["rationale_similarity"])
        scheme1_sentiment_correct.append(s1["sentiment_correct"])

        scheme2_target_sims.append(s2["target_similarity"])
        scheme2_aspect_sims.append(s2["aspect_similarity"])
        scheme2_opinion_sims.append(s2["opinion_similarity"])
        scheme2_rationale_sims.append(s2["rationale_similarity"])
        scheme2_sentiment_correct.append(s2["sentiment_correct"])

    avg_res = {
        "scheme1": {
            "avg_target_similarity": float(np.mean(scheme1_target_sims)) if scheme1_target_sims else 0.0,
            "avg_aspect_similarity": float(np.mean(scheme1_aspect_sims)) if scheme1_aspect_sims else 0.0,
            "avg_opinion_similarity": float(np.mean(scheme1_opinion_sims)) if scheme1_opinion_sims else 0.0,
            "avg_rationale_similarity": float(np.mean(scheme1_rationale_sims)) if scheme1_rationale_sims else 0.0,
            "avg_sentiment_correct": float(np.mean(scheme1_sentiment_correct)) if scheme1_sentiment_correct else 0.0,
        },
        "scheme2": {
            "avg_target_similarity": float(np.mean(scheme2_target_sims)) if scheme2_target_sims else 0.0,
            "avg_aspect_similarity": float(np.mean(scheme2_aspect_sims)) if scheme2_aspect_sims else 0.0,
            "avg_opinion_similarity": float(np.mean(scheme2_opinion_sims)) if scheme2_opinion_sims else 0.0,
            "avg_rationale_similarity": float(np.mean(scheme2_rationale_sims)) if scheme2_rationale_sims else 0.0,
            "avg_sentiment_correct": float(np.mean(scheme2_sentiment_correct)) if scheme2_sentiment_correct else 0.0,
        },
    }

    data["details"] = details
    data["averages"] = avg_res

    with open(out_file_path, "w", encoding="utf-8") as fo:
        json.dump(data, fo, ensure_ascii=False, indent=2)


async def evaluate_file(pred_file, gt_file, embed_config, judge_config):
    """
    对 pred_file 与 gt_file 做对齐行的评估
    """
    pred_data = load_json(pred_file)
    gt_data = load_json(gt_file)

    if len(pred_data) != len(gt_data):
        print(f"Warning: {pred_file} & {gt_file} 行数不同: pred={len(pred_data)} gt={len(gt_data)}")

    results = []
    scheme1_target_sims = []
    scheme1_aspect_sims = []
    scheme1_opinion_sims = []
    scheme1_rationale_sims = []
    scheme1_sentiment_correct = []

    scheme2_target_sims = []
    scheme2_aspect_sims = []
    scheme2_opinion_sims = []
    scheme2_rationale_sims = []
    scheme2_sentiment_correct = []

    min_len = min(len(pred_data), len(gt_data))
    for i in tqdm(range(min_len), desc=f"Processing {os.path.basename(pred_file)}", leave=False):
        # 同行信息
        line_eval = await evaluate_line(pred_data, gt_data, i, embed_config, judge_config)
        line_eval["file"] = os.path.basename(pred_file)  # 覆盖 dummy
        results.append(line_eval)

        s1 = line_eval["scheme1"]
        s2 = line_eval["scheme2"]

        scheme1_target_sims.append(s1["target_similarity"])
        scheme1_aspect_sims.append(s1["aspect_similarity"])
        scheme1_opinion_sims.append(s1["opinion_similarity"])
        scheme1_rationale_sims.append(s1["rationale_similarity"])
        scheme1_sentiment_correct.append(s1["sentiment_correct"])

        scheme2_target_sims.append(s2["target_similarity"])
        scheme2_aspect_sims.append(s2["aspect_similarity"])
        scheme2_opinion_sims.append(s2["opinion_similarity"])
        scheme2_rationale_sims.append(s2["rationale_similarity"])
        scheme2_sentiment_correct.append(s2["sentiment_correct"])

    avg_res = {
        "scheme1": {
            "avg_target_similarity": float(np.mean(scheme1_target_sims)) if scheme1_target_sims else 0.0,
            "avg_aspect_similarity": float(np.mean(scheme1_aspect_sims)) if scheme1_aspect_sims else 0.0,
            "avg_opinion_similarity": float(np.mean(scheme1_opinion_sims)) if scheme1_opinion_sims else 0.0,
            "avg_rationale_similarity": float(np.mean(scheme1_rationale_sims)) if scheme1_rationale_sims else 0.0,
            "avg_sentiment_correct": float(np.mean(scheme1_sentiment_correct)) if scheme1_sentiment_correct else 0.0,
        },
        "scheme2": {
            "avg_target_similarity": float(np.mean(scheme2_target_sims)) if scheme2_target_sims else 0.0,
            "avg_aspect_similarity": float(np.mean(scheme2_aspect_sims)) if scheme2_aspect_sims else 0.0,
            "avg_opinion_similarity": float(np.mean(scheme2_opinion_sims)) if scheme2_opinion_sims else 0.0,
            "avg_rationale_similarity": float(np.mean(scheme2_rationale_sims)) if scheme2_rationale_sims else 0.0,
            "avg_sentiment_correct": float(np.mean(scheme2_sentiment_correct)) if scheme2_sentiment_correct else 0.0,
        },
    }

    final_out = {"details": results, "averages": avg_res}
    return final_out


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pred_dir",
        type=str,
        required=True,
        help="Path to the prediction results directory",
    )
    parser.add_argument("--gt_dir", type=str, required=True, help="Path to the ground truth directory")
    parser.add_argument("--config_path", type=str, default="config.yaml", help="Path to the YAML configuration file")
    parser.add_argument(
        "--embedding_model",
        type=str,
        default="zhipu",
        help="Name of the embedding model config in config.yaml",
    )
    parser.add_argument(
        "--llm_model",
        type=str,
        default="zhipu",
        help="Name of the LLM model config in config.yaml for similarity judgement",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="evaluation_results",
        help="Directory to store output files",
    )
    parser.add_argument("--batch", type=int, default=4, help="Max number of concurrent tasks")
    args = parser.parse_args()
    embed_config = load_yaml_config(args.config_path, args.embedding_model, config_type="embed_config")
    judge_config = load_yaml_config(args.config_path, args.llm_model, config_type="llm_config")
    os.makedirs(args.output_dir, exist_ok=True)
    all_files = []
    for root, _, files in os.walk(args.pred_dir):
        for f in files:
            if re.match(r"output_chat_(\d+)\.json", f):
                number = re.match(r"output_chat_(\d+)\.json", f).group(1)
                pred_file_path = os.path.join(root, f)
                gt_name = f"chat_{number}.json"
                gt_file_path = os.path.join(args.gt_dir, gt_name)
                if os.path.exists(gt_file_path):
                    all_files.append((pred_file_path, gt_file_path))
                else:
                    print(f"GT file not found for {pred_file_path}")

    sem = asyncio.Semaphore(args.batch)

    async def sem_evaluate_file(pred, gt):
        async with sem:
            return (pred, gt, await evaluate_file(pred, gt, embed_config, judge_config))

    filtered_files = []
    for pred_file_path, gt_file_path in all_files:
        base_name = os.path.splitext(os.path.basename(gt_file_path))[0]  # chat_006
        out_file_path = os.path.join(args.output_dir, f"evaluation_results_{base_name}.json")
        if os.path.exists(out_file_path):
            print(f"Skipping {pred_file_path} because {out_file_path} already exists.")
            await check_and_fix_suspicious_lines(
                out_file_path, pred_file_path, gt_file_path, embed_config, judge_config
            )
        else:
            filtered_files.append((pred_file_path, gt_file_path))
    tasks = [asyncio.create_task(sem_evaluate_file(p, g)) for p, g in filtered_files]

    all_results = []
    for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Evaluating all files"):
        pred_file_path, gt_file_path, res = await coro
        all_results.append((pred_file_path, gt_file_path, res))

        base_name = os.path.splitext(os.path.basename(gt_file_path))[0]
        out_file_path = os.path.join(args.output_dir, f"evaluation_results_{base_name}.json")
        with open(out_file_path, "w", encoding="utf-8") as fo:
            json.dump(res, fo, ensure_ascii=False, indent=2)
        await check_and_fix_suspicious_lines(out_file_path, pred_file_path, gt_file_path, embed_config, judge_config)
    final_files = [
        f
        for f in os.listdir(args.output_dir)
        if f.startswith("evaluation_results_") and f.endswith(".json") and f != "evaluation_results_summary.json"
    ]

    scheme1_target_all = []
    scheme1_aspect_all = []
    scheme1_opinion_all = []
    scheme1_rationale_all = []
    scheme1_sentiment_all = []

    scheme2_target_all = []
    scheme2_aspect_all = []
    scheme2_opinion_all = []
    scheme2_rationale_all = []
    scheme2_sentiment_all = []

    files_info = []

    for ff in final_files:
        if ff == "evaluation_results_summary.json":
            continue
        path = os.path.join(args.output_dir, ff)
        data = load_json(path)
        averages = data["averages"]
        scheme1 = averages["scheme1"]
        scheme2 = averages["scheme2"]

        scheme1_target_all.append(scheme1["avg_target_similarity"])
        scheme1_aspect_all.append(scheme1["avg_aspect_similarity"])
        scheme1_opinion_all.append(scheme1["avg_opinion_similarity"])
        scheme1_rationale_all.append(scheme1["avg_rationale_similarity"])
        scheme1_sentiment_all.append(scheme1["avg_sentiment_correct"])

        scheme2_target_all.append(scheme2["avg_target_similarity"])
        scheme2_aspect_all.append(scheme2["avg_aspect_similarity"])
        scheme2_opinion_all.append(scheme2["avg_opinion_similarity"])
        scheme2_rationale_all.append(scheme2["avg_rationale_similarity"])
        scheme2_sentiment_all.append(scheme2["avg_sentiment_correct"])

        base_name = ff.replace("evaluation_results_", "").replace(".json", "")
        matched = [x for x in all_files if os.path.splitext(os.path.basename(x[1]))[0] == base_name]
        if matched:
            p, g = matched[0]
            files_info.append({"pred_file": p, "gt_file": g, "result": data})
        else:
            files_info.append({"pred_file": None, "gt_file": None, "result": data})

    summary = {
        "global_averages": {
            "scheme1": {
                "avg_target_similarity": float(np.mean(scheme1_target_all)) if scheme1_target_all else 0.0,
                "avg_aspect_similarity": float(np.mean(scheme1_aspect_all)) if scheme1_aspect_all else 0.0,
                "avg_opinion_similarity": float(np.mean(scheme1_opinion_all)) if scheme1_opinion_all else 0.0,
                "avg_rationale_similarity": float(np.mean(scheme1_rationale_all)) if scheme1_rationale_all else 0.0,
                "avg_sentiment_correct": float(np.mean(scheme1_sentiment_all)) if scheme1_sentiment_all else 0.0,
            },
            "scheme2": {
                "avg_target_similarity": float(np.mean(scheme2_target_all)) if scheme2_target_all else 0.0,
                "avg_aspect_similarity": float(np.mean(scheme2_aspect_all)) if scheme2_aspect_all else 0.0,
                "avg_opinion_similarity": float(np.mean(scheme2_opinion_all)) if scheme2_opinion_all else 0.0,
                "avg_rationale_similarity": float(np.mean(scheme2_rationale_all)) if scheme2_rationale_all else 0.0,
                "avg_sentiment_correct": float(np.mean(scheme2_sentiment_all)) if scheme2_sentiment_all else 0.0,
            },
        },
        "files": files_info,
    }

    summary_path = os.path.join(args.output_dir, "evaluation_results_summary.json")
    with open(summary_path, "w", encoding="utf-8") as fo:
        json.dump(summary, fo, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    asyncio.run(main())
