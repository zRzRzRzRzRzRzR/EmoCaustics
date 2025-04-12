import os
import json
import argparse
from tqdm import tqdm
from utils import call_large_model, parse_json_response, load_yaml_config
import concurrent.futures

system_prompt = r"""
你是一名情绪事件分析助手。给定一个角色的全部对话，以及与之相关的情绪五元组信息，需要你根据对话和五元组，推断出该角色最重要、最核心的情绪事件。

历史记录有以下特点：
1. 一段对话中完整的历史，包含说话当事人和其他人，这些都是口语对话，因此语言非常口语化。
2，一个人可能多多段连续的说话，这在日常对话中是很正常的。
3. 对话中很有可能不止两个人。
4. 五元组的顺序和历史对话顺序相同，情绪五元组能很好的帮助你分析说话人的情绪。通常来说，这个五元组都是对的。可以用来敏锐的捕捉情绪波动。

输出格式要求:

1. 按照 JSON 格式输出，并且输出的最外层结构只包含当前角色一个 key，例如 `"1": { "events": [...] }`。
2. `"events` 为一个列表，每个元素包含 `"event"`（事件名称）和 `"emotions"`（角色针对该事件的情绪列表）。
3. `emotions` 是一个列表，每个元素至少包含：`{"state": "...", "reason": "..."}`。
   - `state` 必须在 `["positive", "negative", "neutral", "ambiguous", "doubt"]` 中选择。
   - `reason` 描述角色产生此情绪的原因或依据。
4. 一个人可能多有多个事件，如果这个事件是独立的，则必须单独展开。

具体格式如下：
```json
{
    "角色ID_1": {
        "events": [
            {
                "event": 事件名称,
                "sentence_ids": 跟这个事件相关的句子编号，是一个列表，返回()中的数字，比如第一句，第二句，第五句，返回[1,2,5]
                "emotions": [
                    {
                        "source_id": 角色ID(只有一个, 一个数字)，如果这个情感变化是由于某一个角色（一定是角色ID）导致，则输出对应角色ID。如果没有指定，或者由于上下文理解为自己的情感变化，则为自己的角色ID。这个句子一定出现在所有的holders内。
                        "state": "positive/negative/neutral/ambiguous/doubt",
                        "reason": 该事件产生该情绪的原因
                    }
                ] 
            }
        ]
    },
    "角色ID_2": {
        ... 相同的结构
    }
}
```

我会给你提供一个例子
[示例，我的输入会和这个例子相同]
[所有历史记录内容]
[
"1": "这口服液他坚决不能喝，整个一套它就是一欺诈行为，",
"2": "我给孩子们弄点课程，让孩子们喝点口服液，我就成欺诈行为了。孩子们的事你就会在那和稀泥，他们犯了错你就可以跟他讲大道理，你这是啥行为？虚伪，",
"1": "我为孩子好我还虚伪了."
"2": "这满嘴大道理不讲一点实际用处的行为就是虚伪，他们考不上大学咋办？一个个没出息去。"
"1": "采油，怎么？油田怎么了？"
"1": "这些年我这俩孩子都是这油田养的，你别把自己办不成的事加到孩子身上行不行？"
"1": "干嘛非逼着俩孩子去大城市，咱们是什么情况，咱们就过什么日子好不好？"
"1": "孩子们有他们自己的命运对吧？你别非得他们要怎么样行不行？"
"1": "我的意见。"
]
五元组：
[
  {
    "target": "口服液",
    "aspect": "行为",
    "opinion": "骗人",
    "sentiment": "negative",
    "rationale": "说话人对口服液的质疑，认为没效果"
  }
]

此时如果要分析「说话人1」的情绪事件，可输出：

[示例输出]
{
  "1": {
    "events": [
      {
        "event": "质疑口服液",
        "emotions": [
          {
            "source_id": 2
            "state": "negative",
            "reason": "认为口服液是骗人的，没有效果"
          }
        ]
      }
    ]
  }
}

### 注意
- 同一事件中若情绪未发生变化，不应重复记录。
- 一个事件的情绪变化一般不会超过 2 次。你应该给出尽量少的event事件，具体的进展和情绪变化都放在emotions的state和reason字段中，event只是一个简单的大事件概述，不超过10个字。
 - 必须根据事件主题，合并情绪变化的原因。
- 一个事件通常很大，小的原因和事件进展都应该归类在 reason。

""".strip()

user_prompt_template = r"""
我希望你根据下面给你的 [所有历史记录内容] 和 [所有情绪元组内容]，只分析说话人 {holder_id} 的情绪事件。

[所有历史记录内容]
{chat_history}

[所有情绪五元组内容]
{tuple_info}

请忽略其他说话人，不要输出与 {holder_id} 无关的事件。按照系统提示的格式要求输出。
""".strip()


def process_single_file(
    fname, input_dir, output_dir, system_prompt, user_prompt_template, api_key, base_url, model_name
):
    out_fname = fname.replace("chat_", "emotions_")
    out_path = os.path.join(output_dir, out_fname)
    if os.path.exists(out_path):
        return
    file_path = os.path.join(input_dir, fname)
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    lines = []
    holders_this_file = set()
    all_tuples = []
    for item in data:
        sentence = item.get("input_sentence", "")
        holder = item.get("holder", "")
        if holder and sentence:
            line_str = f'"{holder}": "{sentence}"'
            lines.append(line_str)
            holders_this_file.add(holder)
        fmr = item.get("final_model_response", [])
        all_tuples.extend(fmr)
    chat_history_str = "[\n" + ",\n".join(lines) + "\n]"
    tuple_info_str = json.dumps(all_tuples, ensure_ascii=False, indent=2)
    file_result = {}
    for holder_id in sorted(holders_this_file):
        user_prompt = user_prompt_template.format(
            holder_id=holder_id, chat_history=chat_history_str, tuple_info=tuple_info_str
        )
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        response_text = call_large_model(messages, api_key=api_key, base_url=base_url, model=model_name)
        parsed_json = parse_json_response(response_text)
        print(response_text)
        if isinstance(parsed_json, dict) and holder_id in parsed_json:
            file_result[holder_id] = parsed_json[holder_id]
        else:
            file_result[holder_id] = {"events": []}
    with open(out_path, "w", encoding="utf-8") as outf:
        json.dump(file_result, outf, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help="Path to input JSON dir (e.g., ../data)")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Path to save final JSON")
    parser.add_argument("--config_path", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--llm_model", type=str, required=True, help="LLM model name in config")
    parser.add_argument("--batch", type=int, default=20)
    args = parser.parse_args()
    llm_cfg = load_yaml_config(args.config_path, args.llm_model, "llm_config")
    api_key = llm_cfg["api_key"]
    base_url = llm_cfg["base_url"]
    model_name = llm_cfg["model"]
    os.makedirs(args.output_dir, exist_ok=True)
    all_files = sorted(f for f in os.listdir(args.input_dir) if f.endswith(".json"))
    futures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.batch) as executor:
        for fname in all_files:
            futures.append(
                executor.submit(
                    process_single_file,
                    fname,
                    args.input_dir,
                    args.output_dir,
                    system_prompt,
                    user_prompt_template,
                    api_key,
                    base_url,
                    model_name,
                )
            )
        for _ in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing files"):
            pass


if __name__ == "__main__":
    main()
