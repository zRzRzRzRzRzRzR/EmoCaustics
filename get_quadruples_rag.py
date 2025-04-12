import asyncio
import json
import os
import argparse
from datetime import datetime
from tqdm import tqdm
import numpy as np
from utils import (
    call_large_model,
    parse_json_response,
    remove_spaces,
    build_faiss_index,
    call_embedding,
    load_yaml_config,
    build_history_triples,
)

extract_target_prompt = """
你现在的任务是根据给定的当前句子和近期的对话历史来识别当前句子中提到的target对象。无论是当前句子，还是历史，都是按照这个格式提供的:
[
    {"holder": "说话人", "sentence": "说话内容"}
]
当前句子中的holder就是目前说话的角色，你需要根据他的和历史中其他角色的对话进行判断。

你要返回的一元组的内容如下:
{
  "target": 说话人提到的目标对象,可以是人，行为，物品等。但是不能是代词，比如“这一位”，“这个”都是不行的。
}

请注意:
1. 说话人的角色非常重要，target也可能是另一个说话人，如果是另一个人说话人作为target，请你将其识别出来，并输出对应的数字编号。 
2. 你可以参考历史对话来确定代词所指代的具体target。不能直接返回代词。
3. target不能为空。
只需要输出识别的target对象字符串。无需输出其他信息。

请以JSON格式输出结果，例如:
{
  "target": "…"
}

如果无法确定target，请输出空字符串。

下面是一个示例：

当前句子:
{"holder": "1", "sentence": "一听这名就知道这人青春期加痴呆整。哈哈哈，没经验。"}

history:
[
"1": "否则怎么着？否则还能停咱家楼艇，哈哈哈，要不停阳台上去？哎，还有一个刚进的地方，就是姐姐的写字桌，去去去，哈哈哈啊。",
"1": "嗯，那么有个性的网友什么时候给我们引荐引荐啊。他的网名叫什么呀？哈哈哈，好了。",
"2": "歪喉。"
]

输出:
{
  "target": "歪喉网名"
}
"""

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

当前句子按照这个格式提供的:
[
    {"holder": "说话人", "sentence": "说话内容"}
]

历史记录部分，则还会提供一段推断的关于之前对应的历史句子中的五元组供你参考，这个history如下格式:
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

关于RAG的部分，RAG匹配的之前与本人相关可能的对话,通常是一段五个句子的对话，对话不带有五元组。
如果有多个RAG模块，每个RAG模块都是一套这样的对话数组形式，存放在RAG_history的数组中。
即:
RAG_history:
[
  [
    {"holder": "X", "sentence": "..."},
    {"holder": "Y", "sentence": "..."},
    ...
  ],
  [
    ...
  ]
]

请注意:
1. target不能为空，也不可用"None"。
2. RAG的信息不一定有用，它只是帮助理解代词和上下文的辅助信息。
3. 直接输出最后的结果，必须是个JSON，不要输出任何你的分析。

我们开始吧
"""

rag_relevance_judge_prompt = """
你的任务是判断给定的RAG候选句子是否与当前句子相关，以及输出RAG句子的五元组。

给定输入：
- 当前句子（含holder和内容）
- RAG候选句子

如果RAG候选句子与当前句子语义相关（讨论相同人物、同一话题或者为当前目标对象的相关上下文），则输出"相关"，否则输出"不相关"。

输出格式:
{
  "relevance": "相关"或"不相关",
  "quadruple": [
    {
      "target": "...",
      "aspect": "...",
      "ppinion": "...",
      "sentiment": "...",
      "rationale": "..."
    }
  ]
}

如果"不相关"则quadruple留空数组即可。

示例1:
当前句子:
"1": "小雪今天又回来得很晚，衣服很脏。"
RAG候选句子:
"1": "昨天小雪回来的时候也是很晚，身上有泥渍。"

输出:

{
  "relevance": "相关",
  "quadruple": [
    {
      "target": "小雪",
      "aspect": "卫生状况",
      "ppinion": "衣服脏",
      "sentiment": "negative",
      "rationale": "观察到小雪衣服脏污"
    }
  ]
}

示例2:
当前句子:
"1": "小雪今天又回来得很晚，衣服很脏。"
RAG候选句子:
"2": "今天的天气真好，有阳光。"

输出:
{
  "relevance": "不相关",
  "quadruple": []
}

## 注意

- 输出的是一个JSON，而不是一个列表
"""


async def process_single_file(
    input_path, output_path, llm_config, embed_config, history_num, top_k, holder_indices, holder_sentences
):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    data = remove_spaces(data)

    results = []
    sentences = data

    in_window_sentences = {}

    TARGET_SIM_THRESHOLD = 0.5
    SENTENCE_SIM_THRESHOLD = 0.5

    for i in tqdm(range(len(sentences)), desc=f"Processing {input_path}"):
        current = sentences[i]
        current_sentence = current.get("sentence", "")
        current_holder = current.get("Holder", "")
        start_idx = max(0, i - history_num)
        history_data = sentences[start_idx:i]
        formatted_history = [f'"{h.get("Holder", "")}": "{h.get("sentence", "")}"' for h in history_data]

        current_json_str = json.dumps({"holder": current_holder, "sentence": current_sentence}, ensure_ascii=False)
        history_json_str = "[\n" + ",\n".join(formatted_history) + "\n]"
        messages = [
            {"role": "system", "content": extract_target_prompt},
            {
                "role": "user",
                "content": (f"当前句子:\n{current_json_str}\n\nhistory:\n{history_json_str}\n\n请输出target。"),
            },
        ]
        response_str_target = await asyncio.to_thread(call_large_model, messages, **llm_config)
        target_parsed = parse_json_response(response_str_target)
        identified_target = target_parsed.get("target", "").strip() if target_parsed else ""

        rag_modules = []
        if identified_target:
            target_embedding = (await asyncio.to_thread(call_embedding, [identified_target], **embed_config))[0]
            query_embedding = (await asyncio.to_thread(call_embedding, [current_sentence], **embed_config))[0]

            if current_holder in holder_indices and holder_indices[current_holder].ntotal > 0:
                holder_faiss_index = holder_indices[current_holder]
                holder_list = holder_sentences[current_holder]

                target_query = np.array([target_embedding], dtype=np.float32)
                target_distances, target_indices = holder_faiss_index.search(target_query, top_k)

                candidate_ids = [
                    rid
                    for idx_d, rid in enumerate(target_indices[0])
                    if rid >= 0 and rid < len(holder_list) and target_distances[0][idx_d] >= TARGET_SIM_THRESHOLD
                ]

                if candidate_ids:
                    query_query = np.array([query_embedding], dtype=np.float32)
                    query_distances, query_all_indices = holder_faiss_index.search(query_query, len(holder_list))
                    index_to_dist = {
                        query_all_indices[0][j]: query_distances[0][j] for j in range(len(query_all_indices[0]))
                    }

                    final_candidates = [
                        rid
                        for rid in candidate_ids
                        if rid in index_to_dist and index_to_dist[rid] >= SENTENCE_SIM_THRESHOLD
                    ]

                    count = 0
                    for rid in final_candidates:
                        if count >= top_k:
                            break
                        rag_sentence_data = holder_list[rid]
                        rag_holder = rag_sentence_data["holder"]
                        rag_sentence = rag_sentence_data["sentence"]

                        start_idx_rag = max(0, rid - 2)
                        end_idx_rag = min(len(holder_list), rid + 3)
                        rag_context = holder_list[start_idx_rag:end_idx_rag]

                        rag_context_list = [{"holder": x["holder"], "sentence": x["sentence"]} for x in rag_context]

                        judge_messages = [
                            {"role": "system", "content": rag_relevance_judge_prompt},
                            {
                                "role": "user",
                                "content": (
                                    f'当前句子:\n"{current_holder}": "{current_sentence}"\n\n'
                                    f'RAG候选句子:\n"{rag_holder}": "{rag_sentence}"\n\n'
                                    "请根据指令判断相关性并输出五元组。"
                                ),
                            },
                        ]
                        judge_response = await asyncio.to_thread(call_large_model, judge_messages, **llm_config)
                        judge_parsed = parse_json_response(judge_response)
                        if type(judge_parsed) == dict and judge_parsed.get("relevance") == "相关":
                            rag_modules.append(
                                {
                                    "rag_context_list": rag_context_list,
                                    "rag_main_sentence": rag_sentence,
                                    "rag_quadruple": judge_parsed.get("quadruple", []),
                                }
                            )
                            count += 1
        final_history_list = build_history_triples(results, i, history_num)
        rag_history_str = (
            json.dumps([m["rag_context_list"] for m in rag_modules], ensure_ascii=False) if rag_modules else "[]"
        )

        final_messages = [
            {"role": "system", "content": get_quadruples_with_history_prompt},
            {
                "role": "user",
                "content": (
                    f"当前句子:\n{json.dumps({'holder': current_holder, 'sentence': current_sentence}, ensure_ascii=False)}\n\n"
                    f"history:\n{json.dumps(final_history_list, ensure_ascii=False, indent=2)}\n\n"
                    f"RAG_history:\n{rag_history_str}"
                ),
            },
        ]

        final_response = await asyncio.to_thread(call_large_model, final_messages, **llm_config)
        final_parsed = parse_json_response(final_response) or {"raw_response": final_response}

        results.append(
            {
                "input_sentence": current_sentence,
                "holder": current_holder,
                "identified_target": identified_target,
                "rag_modules": rag_modules,
                "final_model_response": final_parsed,
            }
        )

        if identified_target:
            current_embedding = (await asyncio.to_thread(call_embedding, [current_sentence], **embed_config))[0]
            in_window_sentences[i] = {
                "holder": current_holder,
                "sentence": current_sentence,
                "embedding": current_embedding,
            }

        old_index = i - history_num
        if old_index >= 0 and old_index in in_window_sentences:
            try:
                old_data = in_window_sentences[old_index]
                old_holder = sentences[old_index]["Holder"]
                if old_holder not in holder_indices:
                    dim = old_data["embedding"].shape[0]
                    holder_indices[old_holder] = build_faiss_index(dimension=dim)
                if old_holder not in holder_sentences:
                    holder_sentences[old_holder] = []
                holder_sentences[old_holder].append(old_data)
                holder_indices[old_holder].add(np.array([old_data["embedding"]], dtype=np.float32))
                del in_window_sentences[old_index]
            except Exception as e:
                print(f"Error adding old index {old_index}: {e}")

    with open(output_path, "w", encoding="utf-8") as fo:
        json.dump(results, fo, ensure_ascii=False, indent=2)


async def process_dataset(dataset_dir, input_dir, output_dir, llm_config, embed_config, history_num, top_k, batch):
    # 构建输入和输出路径
    cur_input_path = os.path.join(input_dir, dataset_dir)
    json_files = [f for f in os.listdir(cur_input_path) if f.endswith(".json") and f.startswith("chat_")]
    if not json_files:
        return

    cur_output_path = os.path.join(output_dir, dataset_dir)
    os.makedirs(cur_output_path, exist_ok=True)
    holder_indices = {}
    holder_sentences = {}

    semaphore = asyncio.Semaphore(batch)

    async def sem_task(input_path, output_path):
        async with semaphore:
            await process_single_file(
                input_path=input_path,
                output_path=output_path,
                llm_config=llm_config,
                embed_config=embed_config,
                history_num=history_num,
                top_k=top_k,
                holder_indices=holder_indices,
                holder_sentences=holder_sentences,
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
    await asyncio.gather(*tasks)


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        type=str,
        default="../dataset",
        help="Path to input directory containing multiple JSON files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
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
        help="Model name.",
    )
    parser.add_argument(
        "--embedding_model",
        type=str,
        default="zhipu",
        help="Embedding model name.",
    )
    parser.add_argument(
        "--history_num",
        type=int,
        default=3,
        help="Number of history sentences for sliding window.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=3,
        help="Number of top similar targets to retrieve for RAG.",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=1,
        help="Number of datasets to process concurrently.",
    )
    args = parser.parse_args()

    # 加载配置
    llm_config = load_yaml_config(args.config_path, args.llm_model, config_type="llm_config")
    embed_config = load_yaml_config(args.config_path, args.embedding_model, config_type="embed_config")

    # 准备数据目录
    # subdirs = [
    #     d for d in os.listdir(args.input_dir) if os.path.isdir(os.path.join(args.input_dir, d))
    # ]
    subdirs = ["chat"]
    # 输出目录
    output_dir = os.path.join(
        args.output_dir, os.path.basename(llm_config["model"]) + "_" + datetime.now().strftime("%m%d")
    )
    semaphore = asyncio.Semaphore(args.batch)

    async def sem_task(dir_name):
        async with semaphore:
            await process_dataset(
                dataset_dir=dir_name,
                input_dir=args.input_dir,
                output_dir=output_dir,
                llm_config=llm_config,
                embed_config=embed_config,
                history_num=args.history_num,
                top_k=args.top_k,
                batch=args.batch,
            )

    # 创建任务
    tasks = [asyncio.create_task(sem_task(d)) for d in subdirs]
    await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(main())
