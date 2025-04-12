import asyncio
import json
import os
import argparse
from datetime import datetime
from tqdm import tqdm
from utils import call_large_model, parse_json_response, load_yaml_config

get_quadruples_with_history_prompt = """
你应该根据输入的当前句子、history和RAG_history，输出当前句子的完整情感五元组。
输出的五元组格式和对应字段的解释是:
[
    {
        "target": 说话人提到的目标对象,可以是人，行为，物品等。
        "aspect": 说话人提到的目标对象的属性，可以是性格，行为，物品的某一个特性等，但是必须是说话人评论的对象。
        "opinion": 说话人对目标对象的评论，可以是说话人的态度，情感，看法等，请注意，这一定是一个态度。
        "sentiment": 情感态度,从positive、negative、neutral、ambiguous、doubt中选择一个。
        "rationale": 说话人这么说的原因，可以是说话人的行为，经历，观察等。
    }
]

当前句子和历史记录按照这个格式提供的:
[F
    {"holder": "说话人", "sentence": "说话内容"}
]

历史记录部分，我将会为你提供整段对话的完整信息，这是这段对话的完整历史记录，我会用一个Json的列表来进行记录:
[
    {"holder": "说话人1", "sentence": "说话内容1"}
    {"holder": "说话人2", "sentence": "说话内容2"}
    {"holder": "说话人1", "sentence": "说话内容3"}
    ...（多条记录）
]
其中，1，2 代表说话人，这是你分析五元组的重要提示。

请注意:
1. target不能为空，也不可用"None"。
2. RAG的信息不一定有用，它只是帮助理解代词和上下文的辅助信息。
3. 直接输出最后的结果，必须是个JSON，不要输出任何你的分析。

现在，我们开始吧:
"""

get_quadruples_with_history_prompt_user = """
全局历史记录:
{history}
当前句子:
"""


async def process_single_file(input_path, output_path, llm_config):
    """
    处理单个 JSON 文件:
    - 将整个 JSON 文件作为全局历史
    - 针对每个句子，调用大模型生成情感五元组
    - 保存结果到 output_path
    """
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    try:
        global_history = [{"holder": entry.get("Holder", ""), "sentence": entry.get("sentence", "")} for entry in data]
    except:
        breakpoint()
    global_history_str = json.dumps(global_history, ensure_ascii=False, indent=2)

    results = []
    for i in tqdm(range(len(data)), desc=f"Processing {os.path.basename(input_path)}"):
        current = data[i]
        current_sentence = current.get("sentence", "")
        current_holder = current.get("Holder", "")

        current_json_str = json.dumps({"holder": current_holder, "sentence": current_sentence}, ensure_ascii=False)
        user_content = f"全局历史记录:\n{global_history_str}\n\n当前句子:\n{current_json_str}\n"

        messages = [
            {"role": "system", "content": get_quadruples_with_history_prompt},
            {"role": "user", "content": user_content},
        ]
        response_str = await asyncio.to_thread(call_large_model, messages, **llm_config)
        final_parsed = parse_json_response(response_str)
        if not final_parsed:
            final_parsed = {"raw_response": response_str}

        results.append(
            {"input_sentence": current_sentence, "holder": current_holder, "final_model_response": final_parsed}
        )

    # 保存到输出文件
    with open(output_path, "w", encoding="utf-8") as fo:
        json.dump(results, fo, ensure_ascii=False, indent=2)


async def process_dataset(dataset_dir, input_dir, output_dir, llm_config, batch):
    if dataset_dir != "chat":
        return
    cur_input_path = os.path.join(input_dir, dataset_dir)
    json_files = [f for f in os.listdir(cur_input_path) if f.endswith(".json") and f.startswith("chat_")]
    if not json_files:
        return

    cur_output_path = os.path.join(output_dir, dataset_dir)
    os.makedirs(cur_output_path, exist_ok=True)

    semaphore = asyncio.Semaphore(batch)

    async def sem_task(input_path, output_path):
        async with semaphore:
            await process_single_file(input_path=input_path, output_path=output_path, llm_config=llm_config)

    tasks = []
    for jf in json_files:
        input_path = os.path.join(cur_input_path, jf)
        output_filename = "output_" + jf
        output_path = os.path.join(cur_output_path, output_filename)

        if os.path.exists(output_path):
            print(f"Skipping {output_path} as it already exists.")
            continue

        tasks.append(asyncio.create_task(sem_task(input_path, output_path)))

    await asyncio.gather(*tasks)


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        type=str,
        default="../dataset",
        help="Path to the input directory containing multiple JSON files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs_0105",
        help="Directory to store output files.",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="config.yaml",
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--llm_model",
        type=str,
        default="zhipu",
        help="LLM model name to use (defined in config.yaml under llm_config).",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=1,
        help="Max concurrent files to process.",
    )
    args = parser.parse_args()

    llm_config = load_yaml_config(args.config_path, args.llm_model, config_type="llm_config")

    subdirs = [d for d in os.listdir(args.input_dir) if os.path.isdir(os.path.join(args.input_dir, d))]
    output_dir = os.path.join(
        args.output_dir, os.path.basename(llm_config["model"]) + "_" + datetime.now().strftime("%m%d")
    )
    os.makedirs(output_dir, exist_ok=True)

    semaphore = asyncio.Semaphore(args.batch)

    async def sem_task(dir_name):
        async with semaphore:
            await process_dataset(
                dataset_dir=dir_name,
                input_dir=args.input_dir,
                output_dir=output_dir,
                llm_config=llm_config,
                batch=args.batch,
            )

    tasks = [asyncio.create_task(sem_task(d)) for d in subdirs]
    await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(main())
