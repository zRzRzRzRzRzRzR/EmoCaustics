import asyncio
import json
import os
import argparse
from datetime import datetime
from tqdm import tqdm
from utils import call_large_model, remove_spaces, load_yaml_config, parse_json_response

event_summary_prompt = """
你是一名高级情绪事件分析助手。你的任务是：

1. 根据给定的对话历史记录和已有事件，判断当前对话中的句子是否属于已有事件，如果是，请更新对应事件；如果不是，请新建一个事件。

2. 每个事件包含以下信息：
   - "event": 事件的最新名称或主题（字符串，可能经过更新）
   - "original_event": 原始事件的名称或主题（必须与 [已有事件] 中提供的 `event` 的完全一致，如果当前句子不属于已有事件，则返回 null）
   - "description": 事件的具体经过描述（需要注重细节，每加入一个新的句子，描述可能会更新）
   - "holders": 所有参与到这个事件中的说话人（数组格式）
   - "sentence_ids": 参与到这个事件的句子的编号（绝对编号，数组格式）
3. 请根据以下数据生成更新后的事件总结，并按照如下 JSON 格式返回结果：

```json
[
    {
      "event": "事件最新名称",
      "original_event": "原始事件名称",
      "description": "事件的描述",
      "holders": ["holder1", "holder2"],
      "sentence_ids": [1, 2, 3]
    }
]
```

4. 我会为你提供过去的历史记录和已经有的事件桶，他们的结构如下:
[已有事件]
[
    {"evnet": "事件概括1", "holders": ["说话人1", "说话人2"], "description": 事件描述, "last_sentences": ["最近说话内容1", "最近说话内容2", "最近说话内容3"]}
    {"evnet": "事件概括2", "holders": ["说话人1", "说话人2"], "description": 事件描述, "last_sentences": ["最近说话内容1", "最近说话内容2", "最近说话内容3"]}
    ...
]
这代表了已经有的一个事件，以及涉及到的说话人id，事件描述，以及这个事件最新的K个句子的原话。

[历史记录]
[
    {'sentence_id': 1, 'sentence': '句子1', 'holder': '说话人id'}, 
    {'sentence_id': 2, 'sentence': '句子2', 'holder': '说话人id'}
    ...
]

### 注意

- 如果当前事件和已有事件属于同一个事件，`original_event` 返回的时候必须和 [已有事件] 中的 `event` 完全一致。这非常重要。
- `description` 字段描述应该详细，特别是人物和说话的细节，以便更好地理解事件。
- 如果当前句子不属于已有事件，则 `original_event` 返回 null。

"""

get_quadruples_system_prompt = """
你应该根据输入的当前句子和历史记录，以及它们所在事件的详细信息，输出当前句子的完整情感五元组。
输出的五元组格式以及各字段的含义如下：
```json
[
    {
      "target": "说话人提到的目标对象，可以是人、一个具体的行为或物品，或者一个事件，比如“XXX的行为”。",
      "aspect": "说话人描述目标对象的某个属性（如性格、行为或特性），必须是评论的对象，具体在例子中查看。",
      "opinion": "说话人对目标对象所表达的态度、情感或看法，必须为明确态度而不是一个事件",
      "sentiment": "情感态度，从positive、negative、neutral、ambiguous、doubt中选择一个",
      "rationale": "说话人发表上述评论的原因，如其行为、经历或观察等。"
    }
]
```

如果当前句子不包含任何情感，请返回：
```json
[]
```

### 输入数据

我会为你提供如下数据：

[当前句子]
[
    {"holder": "说话人", "sentence": "说话内容"}
]

[历史记录格式]
[
    {
      "sentence": "...",
      "holder": "...",
      "target": "...",
      "aspect": "...",
      "opinion": "...",
      "sentiment": "...",
      "rationale": "..."
    }
]

[相关事件和历史记录]
```json
[
    {
        "event": "事件名称",
        "holders": [...],
        "description": "事件的详细描述",
        "past_dialogues": [
            {"holder": "说话人", "sentence": "对话内容1" },
            {"holder": "说话人", "sentence": "对话内容2" }
        ],
        "past_quadruples": [
            {
                包含上述历史对话的情绪五元组信息，用于参考，这是最重要的部分。
            }
        ]
    }
]
```

其中，[相关事件和历史记录] 包含了:

+ 当前句子所属事件的整体描述，请注意，这个信息是为了让你更好的参考，特别是target。
+ 前句子之前该事件中出现的明文对话（past_dialogues）和对应的情感五元组（past_quadruples）, 请注意，这些对话是连续的。

如果没有提供，则直接提供[历史记录格式]，这是该句子之前的连续数段连续对话的内容。

### 示例

#### 输入数据

[历史记录]
[
    {
        "sentence_id": 0,
        "sentence":  党和人民就没有赋予我们这样的权利，"
        "holder": 1
    }
]

[已有事件]
[]

### 输出示例

```json
[
    {
      "target": "权利",
      "aspect": "权利的赋予",
      "opinion": "没有赋予",
      "sentiment": "neutral",
      "rationale": "说话人1在陈述一个事实，即党和人民没有赋予某种权利。"
    }
]
```

### 注意事项

- 请你基于上述数据，结合同一事件的整体描述、历史明文对话与对应的五元组，仅使用当前句子之前的信息来推断本句的情感五元组；
- 不允许利用当前句子之后的信息来进行情感推断。
- 请你严格按照上述 ```json 格式返回结果，不要包含任何额外信息或中间推理过程。示例中的 aspect 和 target的关系可以作为参考。

"""


async def process_single_file(input_path, output_path, llm_config, history_num, window_size, last_sentences_count):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    data = remove_spaces(data)
    total_sentences = len(data)
    id_to_sentence = {}
    for idx, item in enumerate(data):
        abs_id = item.get("sentence_id") if item.get("sentence_id") is not None else idx + 1
        sentence_text = (
            item.get("input_sentence") if item.get("input_sentence") is not None else item.get("sentence", "")
        )
        id_to_sentence[abs_id] = sentence_text
    events_bucket = []
    processed_results = []

    step = history_num if history_num > 0 else 1

    for window_start in range(0, total_sentences, step):
        window_end = min(total_sentences, window_start + window_size)
        window_data = data[window_start:window_end]
        filtered_window_data = [
            {
                "sentence_id": idx + 1,
                "sentence": (
                    item.get("input_sentence") if item.get("input_sentence") is not None else item.get("sentence", "")
                ),
                "holder": (item.get("holder") if item.get("holder") is not None else item.get("Holder", "")),
            }
            for idx, item in enumerate(window_data)
        ]
        formatted_events = []
        for event in events_bucket:
            sorted_ids = sorted(event.get("sentence_ids", []))
            last_ids = sorted_ids[-last_sentences_count:] if len(sorted_ids) >= last_sentences_count else sorted_ids
            last_sentences = [id_to_sentence[sid] for sid in last_ids if sid in id_to_sentence]
            formatted_events.append(
                {
                    "event": event.get("event"),
                    "holders": event.get("holders"),
                    "description": event.get("description", ""),
                    "last_sentences": last_sentences,
                }
            )
        user_prompt = "[历史记录]\n" + json.dumps(filtered_window_data, ensure_ascii=False, indent=2) + "\n"
        user_prompt += "[已有事件]\n" + json.dumps(formatted_events, ensure_ascii=False, indent=2) + "\n"
        messages = [{"role": "system", "content": event_summary_prompt}, {"role": "user", "content": user_prompt}]
        response_text = call_large_model(
            messages, api_key=llm_config["api_key"], base_url=llm_config["base_url"], model=llm_config["model"]
        )
        new_events = parse_json_response(response_text)
        if not new_events:
            print("警告：模型返回内容解析为空。")
            continue

        for new_event in new_events:
            rel_ids = new_event.get("sentence_ids", [])
            abs_ids = [window_start + rid for rid in rel_ids]
            new_event["sentence_ids"] = abs_ids

            def normalize_event_name(name):
                return name.strip().lower() if name else ""

            matched = False
            new_orig = new_event.get("original_event")
            if new_orig is not None:
                for event in events_bucket:
                    if normalize_event_name(event.get("event")) == normalize_event_name(new_orig):
                        event["event"] = new_event.get("event")
                        event["description"] = new_event.get("description")
                        event["holders"] = list(set(event.get("holders", []) + new_event.get("holders", [])))
                        event["sentence_ids"] = list(
                            set(event.get("sentence_ids", []) + new_event.get("sentence_ids", []))
                        )
                        matched = True
                        break
            if not matched:
                events_bucket.append(new_event)
    for event in events_bucket:
        event.pop("original_event", None)

    for idx, item in enumerate(data):
        abs_id = item.get("sentence_id") if item.get("sentence_id") is not None else idx + 1
        assigned_event_data = None
        for ev in events_bucket:
            if abs_id in ev.get("sentence_ids", []):
                assigned_event_data = ev
                break
        prompt_six = (
            "[当前句子]\n"
            + json.dumps(
                [
                    {
                        "holder": item.get("holder") if item.get("holder") is not None else item.get("Holder", ""),
                        "sentence": item.get("input_sentence")
                        if item.get("input_sentence") is not None
                        else item.get("sentence", ""),
                    }
                ],
                ensure_ascii=False,
                indent=2,
            )
            + "\n"
        )
        if assigned_event_data is not None:
            relevant_ids = [sid for sid in sorted(assigned_event_data.get("sentence_ids", [])) if sid < abs_id]
            relevant_ids = relevant_ids[-last_sentences_count:]
            past_dialogues = []
            for sid in relevant_ids:
                if sid in id_to_sentence:
                    holder_str = data[sid - 1].get("holder") or data[sid - 1].get("Holder", "")
                    past_dialogues.append({"holder": holder_str, "sentence": id_to_sentence[sid]})
            past_quadruples = []
            for pr in processed_results:
                sid = pr["sentence_id"]
                if sid in relevant_ids and isinstance(pr["final_model_response"], list):
                    for quadruple in pr["final_model_response"]:
                        entry = {
                            "holder": pr["holder"],
                            "target": quadruple.get("target", ""),
                            "aspect": quadruple.get("aspect", ""),
                            "opinion": quadruple.get("opinion", ""),
                            "sentiment": quadruple.get("sentiment", ""),
                            "rationale": quadruple.get("rationale", ""),
                        }
                        past_quadruples.append(entry)
            event_obj = {
                "event": assigned_event_data["event"],
                "holders": assigned_event_data.get("holders", []),
                "description": assigned_event_data.get("description", ""),
                "past_dialogues": past_dialogues,
                "past_quadruples": past_quadruples,
            }
            prompt_six += "[相关事件和历史记录]\n" + json.dumps([event_obj], ensure_ascii=False, indent=2) + "\n"
        else:
            fallback_count = min(last_sentences_count, abs_id - 1)
            start_id = max(1, abs_id - fallback_count)
            fallback_dialogues = []
            for sid in range(start_id, abs_id):
                holder_str = data[sid - 1].get("holder") or data[sid - 1].get("Holder", "")
                fallback_dialogues.append({"holder": holder_str, "sentence": id_to_sentence[sid]})
            fallback_obj = {"past_dialogues": fallback_dialogues}
            prompt_six += "[相关事件和历史记录]\n" + json.dumps([fallback_obj], ensure_ascii=False, indent=2) + "\n"
        messages = [
            {"role": "system", "content": get_quadruples_system_prompt},
            {"role": "user", "content": prompt_six},
        ]
        response_text = call_large_model(
            messages, api_key=llm_config["api_key"], base_url=llm_config["base_url"], model=llm_config["model"]
        )
        try:
            six_tuple = parse_json_response(response_text)
        except Exception as e:
            print("模型返回内容无法正常解析为JSON:", e)
            six_tuple = []
        result_record = {
            "input_sentence": item.get("input_sentence")
            if item.get("input_sentence") is not None
            else item.get("sentence", ""),
            "holder": item.get("holder") if item.get("holder") is not None else item.get("Holder", ""),
            "sentence_id": abs_id,
            "event": assigned_event_data["event"] if assigned_event_data else None,
            "final_model_response": six_tuple,
        }
        processed_results.append(result_record)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(processed_results, f, ensure_ascii=False, indent=2)


async def process_dataset(
    dataset_dir, input_dir, output_dir, llm_config, history_num, batch, window_size, last_sentences_count
):
    cur_input_path = os.path.join(input_dir, dataset_dir)
    json_files = [f for f in os.listdir(cur_input_path) if f.endswith(".json") and f.startswith("chat_")]
    if not json_files:
        return
    cur_output_path = os.path.join(output_dir, dataset_dir)
    os.makedirs(cur_output_path, exist_ok=True)
    semaphore = asyncio.Semaphore(batch)

    async def sem_task(input_path, output_path):
        async with semaphore:
            await process_single_file(
                input_path, output_path, llm_config, history_num, window_size, last_sentences_count
            )

    tasks = []
    for jf in json_files:
        input_path = os.path.join(cur_input_path, jf)
        output_filename = "output_" + jf
        output_path = os.path.join(cur_output_path, output_filename)
        if os.path.exists(output_path):
            print(f"Skipping {output_path} as it already exists.")
            continue
        tasks.append(asyncio.create_task(sem_task(input_path, output_path)))
    total_tasks = len(tasks)
    with tqdm(total=total_tasks, desc="Processing JSON files") as global_pbar:
        for fut in asyncio.as_completed(tasks):
            await fut
            global_pbar.update(1)


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir", type=str, default="../dataset", help="Path to input directory containing multiple JSON files."
    )
    parser.add_argument("--output_dir", type=str, default="outputs", help="Directory to store output files.")
    parser.add_argument("--config_path", type=str, default="config.yaml", help="Path to the YAML configuration file.")
    parser.add_argument("--llm_model", type=str, default="zhipu", help="Name of the LLM model.")
    parser.add_argument(
        "--history_num",
        type=int,
        default=5,
        help="Number of sentences used for historical reference in the sliding window.",
    )
    parser.add_argument("--batch", type=int, default=1, help="Number of JSON files to process concurrently.")
    parser.add_argument(
        "--window_size", type=int, default=20, help="Size of the sliding window (number of sentences per window)."
    )
    parser.add_argument(
        "--last_sentences_count",
        type=int,
        default=3,
        help="Number of most recent plain sentences to take when passing existing events (default is 3 sentences).",
    )
    args = parser.parse_args()

    llm_config = load_yaml_config(args.config_path, args.llm_model, config_type="llm_config")
    subdirs = ["chat"]
    output_dir = os.path.join(
        args.output_dir, os.path.basename(llm_config["model"]) + "_" + datetime.now().strftime("%m%d")
    )
    os.makedirs(output_dir, exist_ok=True)
    tasks = []
    for d in subdirs:
        tasks.append(
            asyncio.create_task(
                process_dataset(
                    dataset_dir=d,
                    input_dir=args.input_dir,
                    output_dir=output_dir,
                    llm_config=llm_config,
                    history_num=args.history_num,
                    batch=args.batch,
                    window_size=args.window_size,
                    last_sentences_count=args.last_sentences_count,
                )
            )
        )
    await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(main())
