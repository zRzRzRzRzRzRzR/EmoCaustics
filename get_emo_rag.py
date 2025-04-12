import os
import json
import re
import argparse
from tqdm import tqdm
import concurrent.futures
from utils import call_large_model, parse_json_response, load_yaml_config, merge_similar_emotions_with_llm


def format_chat_history_for_llm(chat_data):
    formatted_chat = []
    for item in chat_data:
        holder = item.get("holder", "")
        sentence = item.get("input_sentence", "")
        if holder and sentence:
            formatted_chat.append(f'"{holder}": "{sentence}"')

    return "[\n" + ",\n".join(formatted_chat) + "\n]"


def segment_events_by_topic(dialogues, api_key, base_url, model_name):
    formatted_chat = format_chat_history_for_llm(dialogues)

    system_prompt_template = r"""
    你是一名情绪事件分析助手。给定一个角色的全部对话，以及与之相关的情绪五元组信息，放置在 [相关历史记录和情绪五元组] 后。
    你需要根据对话和五元组，推断出该角色最重要、最核心的情绪事件。

    ### 数据特点 
    1. 这是一段完整的对话历史，包含多个说话人，语言非常口语化。
    2. 可能存在某个角色连续多次发言的情况。
    3. 可能有多个不同的说话人参与对话。
    4. 五元组信息与历史对话顺序相同，可用于辅助识别角色的情绪波动。
    5. 当前人说话可能没有任何情绪或参与到任何事件，比如主持人活着旁白，这种时候，events允许是空的。
    
    ### 输出格式要求
    1. JSON 格式，外层结构应只包含当前角色的 ID，例如 `"1": { "events": [...] }`。
    2. `events` 为一个列表，每个事件包含：
       - "event"（事件名称，通常是一个很大的事件或者话题，每个角色的两个话题差异通常是很大的）
       - "emotions"（角色在该事件中的情绪列表和每个阶段导致这个情绪的原因）

    3. `emotions` 结构：
       - 只记录**关键的情绪变化**，避免相同情绪重复
       - `state` 选项：`["positive", "negative", "neutral", "ambiguous", "doubt"]`
       - `reason` 需要描述角色情绪变化的依据或原因。这通常是这个事件的进展。

    ### 注意
    - 同一事件中若情绪未发生变化，不应重复记录。
    - 一个事件的情绪变化一般不会超过 2 次。你应该给出尽量少的event事件，具体的进展和情绪变化都放在emotions的state和reason字段中，event只是一个简单的大事件概述，不超过10个字。
    - 必须根据事件主题，合并情绪变化的原因。
    - 一个事件通常很大，小的原因和事件进展都应该归类在 reason。
    - 不需要带有任何解释,只能返回要求内的内容。
    - 输出格式必须完全和 ### 输出格式要求的一样
    
    ### 输出格式要求
    ```json
    {{
        "角色": {{
            "events":
                [
                    {{
                        "event": "事件名称",
                        "emotions": [
                            {{
                                "state": "positive/negative/neutral",
                                "reason": "该事件产生该情绪的原因"
                            }}
                        ]
                    }}
                ]
        }}
    }}
    好的，我们开始吧:
    """.strip()
    user_prompt_template = f"""
    [相关历史记录和情绪五元组]
    {formatted_chat}
    
    - 输出格式必须完全和 ### 输出格式要求的一样，必须是个JSON
    """
    response = call_large_model(
        messages=[
            {"role": "system", "content": system_prompt_template},
            {"role": "user", "content": user_prompt_template},
        ],
        api_key=api_key,
        base_url=base_url,
        model=model_name,
    )
    return parse_json_response(response)


def process_single_file(
    fname,
    input_dir,
    output_dir,
    llm_api_key,
    llm_base_url,
    llm_model_name,
):
    file_id = re.search(r"chat_(\d+)\.json", fname)
    file_id = file_id.group(1) if file_id else fname.replace(".json", "")
    out_fname = f"output_emotions_{file_id}.json"
    out_path = os.path.join(output_dir, out_fname)
    if os.path.exists(out_path):
        print(f"⏩ Skipping {fname}, output already exists: {out_path}")
        return

    file_path = os.path.join(input_dir, fname)
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    holders_this_file = set(item["holder"] for item in data if "holder" in item)
    file_result = {}

    for holder_id in sorted(holders_this_file):
        holder_sentences = [item for item in data if item.get("holder") == holder_id]
        events = segment_events_by_topic(holder_sentences, llm_api_key, llm_base_url, llm_model_name)
        file_result[holder_id] = {"events": []}
        for holder_id, holder_data in events.items():
            file_result[holder_id] = {"events": []}
            for event in holder_data["events"]:
                event["emotions"] = merge_similar_emotions_with_llm(
                    event["emotions"], llm_api_key, llm_base_url, llm_model_name
                )
                file_result[holder_id]["events"].append(event)
    with open(out_path, "w", encoding="utf-8") as outf:
        json.dump(file_result, outf, ensure_ascii=False, indent=2)

    print(f"✅ Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--config_path", type=str, default="config.yaml")
    parser.add_argument("--llm_model", type=str, required=True)
    parser.add_argument("--batch", type=int, default=8)
    args = parser.parse_args()

    llm_cfg = load_yaml_config(args.config_path, args.llm_model, "llm_config")
    os.makedirs(args.output_dir, exist_ok=True)

    # **1️⃣ 解析所有 JSON 文件，并按数字顺序排序**
    def extract_number(file_name):
        match = re.search(r"chat_(\d+)\.json", file_name)
        return int(match.group(1)) if match else float("inf")  # **确保 chat_1 在 chat_10 之前**

    all_files = sorted(
        [f for f in os.listdir(args.input_dir) if f.endswith(".json")],
        key=extract_number,  # **基于数字排序**
    )

    # **2️⃣ 按 batch 依次提交任务**
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.batch) as executor:
        with tqdm(total=len(all_files), desc="Processing files") as pbar:
            for i in range(0, len(all_files), args.batch):  # **按 batch 大小分批**
                batch_files = all_files[i : i + args.batch]  # **取出当前 batch**

                futures = {
                    executor.submit(
                        process_single_file,
                        fname,
                        args.input_dir,
                        args.output_dir,
                        llm_cfg["api_key"],
                        llm_cfg["base_url"],
                        llm_cfg["model"],
                    ): fname
                    for fname in batch_files
                }

                for future in concurrent.futures.as_completed(futures):
                    future.result()
                    pbar.update(1)


if __name__ == "__main__":
    main()
