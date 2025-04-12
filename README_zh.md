# EmoCaustics

# 数据原理 🌀

## 数据集背景

数据集为从网络中搜集到的现实世界中真实发生的长对话数据，主要内容可分为四个大场景：吵架，辩论，研讨会以及除此之外的日常沟通。
获得其中的长对话文本内容后进行情绪六元组的标注和事件链的标注。

## 人工标注

在获得其中的文本内容后，使用大模型辅助进行情绪六元组的标注。在辅助完成后对其进行人工逐条审核并进行修改，使其符合事件的原始内容。
情感(sentiment)则使用大模型进行多次分析评分，在最后辅以人工评分进行情感的确定。
事件链中对事件的内容，事件中的情感变化，以及引起情感变化的原因进行人工标注。

### 事件区分

在区分事件中同样使用大模型进行辅助标注，在完成后进行人工分析，区分不同的事件进行修改。其中事件(event)
为对话中的主要事件，而子事件则放入细粒度分析的情绪变化原因(reason)中。
其中导致某一对话以及引起说话人情感变化的人被标记为source_id，其为holder的id且其通常不是说话人自身。

### 数据集格式解释

数据集分为两部分，第一部分为对话的情绪六元组，第二部分为事件链。
第一部分：

```json
[
  {
    "sentence": "对话的内容",
    "Holder": "说话人序号",
    "Target": "对话的主要内容",
    "Aspect": "对话的主题",
    "Opinion": "说话人对对话中主题的意见",
    "Sentiment": "说话人在该句对话中体现的情感",
    "Rationale": "出现情感的原因"
  }
]
```

第二部分:

```json
{
  "holder": {
    "events": [
      {
        "event": "holder在对话中参与的主要事件",
        "emotions": [
          {
            "state": "holder在事件中的情感",
            "reason": "情感变化的原因",
            "source_id": "导致情感变化的事情(其他说话人且通常不是自己)"
          }
        ]
      }
    ]
  }
}
```


## 论文中的数据

+ 我们使用了`100`条常规数据集和`20`条长数据集和有难度的数据集来进行测试。他们的ID是`chat_01` - `chat_100`。
+ 其中，`20` 条数据集用来比较不同滑动窗口和步长。这 20 条数据长度超过 100 轮对话，包含多个任务或者复杂的话题，情绪变化较大，较多。他们的编号是：
`chat_207`, `chat_218`, `chat_223`, `chat_235`, `chat_251`, `chat_276`, `chat_313`, `chat_318`, `chat_330`, `chat_338`, `chat_365`, `chat_366`, `chat_367`, `chat_376`, `chat_381`, `chat_387`, `chat_389`, `chat_392`, `chat_394`, `chat_395`
+ 数据中已经提供了 ground truth。但是在运行过程中，我们不会使用GT，而是使用大模型来进行推理。并最后和GT进行对比。
+ 在本代码仓库中，你可以在 `datasets` 下找到完整的`1000`条数据。

# 算法原理 🛎️

## 获取五元组

### 滑动窗口和事件桶

由于输入的历史数据太长，大模型通常无法一次性对那么多的数据(超过8K长度的数据)进行理解。同时，直接调用大模型导致了大模型在总结句子关系上遗漏多个关键信息。
如果我们讲一段非常场的对话分成多个小段，然后对每个小段进行情绪五元组的提取，那么我们就可以得到更多的信息。我们将这个过程称为滑动窗口。

但是，这还不够，滑动窗口的长度较小，假设我们将滑动窗口长度设为`K`，而每次移动`S`的步长。 那么，在第`i`个滑动窗口下滑动到第
`i + 1`个滑动窗口中，位于 `[k *(i-1) + S， K * (i) + S]` 区间的数据是重复的，会总结出多个重复的事件。因此，我们需要引入`事件桶`
的概念。
当新出现的句子如果和事件桶中的句子相似度超过阈值，或者用大模型判断他们处于同一个事件下，那么我们认为这两个句子是同一个事件，我们将这个句子放入事件桶中。否则，我们将这个句子作为一个新的事件。

### 事件桶更新

事件桶应该是一个列表的形式存在，包括四个字段：

```json
[
  {
    "event": "事件名称1",
    "holders": "所有参与到这个事件中的holders",
    "description": "事件的描述 (不是总结，而是记流水账的方式，不然新的句子难以插入)",
    "sentence_ids": "跟这个事件相关的句子编号，比如第一句，第二句，第五句，返回[1,2,5]"
  }
]
```

`description` 字段只描述了当前某一个句子条件下，已经发现的事件的描述。在后续的处理中，我们会对这个字段进行更新。

在每一个滑动窗口中，我们都能获得`S`个新的句子传入，并对这`K`长度的滑动窗口来进行更新，这时候，更新后的事件桶其实会记录上
`K * i + S`的句子的总结，所以每执行一个代码，就有`S`个长度的句子被成功加入到事件桶。

这时候有两种情况：

+ 当 `事件名称` 和已经在`events`列表中相同时，则调用一次描述函数，更新`description`字段。
+ 当 `事件名称` 和已经在`events`列表中不相同时，则将这个事件加入到`events`列表中。并对这些句子进行描述。

### 获取target

由于滑动窗口的存在，这一步，我们还可以做一个事情，获取`target`。 六元组中数据集已经给我们提供了`holder`。而通常来说，人说话的时候一般的
`target`是面向自己的上一句，或者某一个角色的前一句。否则，在说话中会直接指明`targets`。在这第一轮遍历中，我们就能直接获得
`targets`。

### 总结六元组

现在，我们拥有了每个句子所在的事件和`target`，我们可以对每个句子进行情绪五元组的提取。我们将这个过程称为`总结六元组`。
同时，判断每个句子的六元组之前，我们会提供这个事件的完整信息，这主要是提升 `rationale` 和 `aspect` 。

第二次遍历的时候，我们使用一个句子一个句子的方式遍历,而不是使用滑动窗口来完成遍历。当第二次便利完数据集的时候，就完成了所有的六元组的处理。
最后输出的每个事件是:

```json
{
  "input_sentence": "干嘛非逼着俩孩子去大城市，咱们是什么情况，咱们就过什么日子好不好？",
  "holder": "1",
  "event": "争论孩子是否应该去大城市读书",
  "final_model_response": [
    {
      "target": "俩孩子",
      "aspect": "去大城市",
      "opinion": "不应该被逼着去",
      "sentiment": "negative",
      "rationale": "说话人认为应该根据实际情况过生活，不应该强迫孩子们去大城市"
    }
  ]
}
```

将每个事件拼起来，就能实现了所有的六元组的提取。

## 获取情绪事件

直接使用 LLM 来获得所有句子的情绪事件，同样面临长文本状态下信息丢失和超过大模型长度上限的问题。我们能在完成五元组的提取的情况下，使用五元组的信息作为额外的condition，进一步完成情绪事件的提取。

情绪事件的格式为：

```json
{
  "1": {
    "events": [
      {
        "event": "医生身份争议",
        "emotions": [
          {
            "state": "neutral",
            "reason": "以专业态度开始治疗对话",
            "source_id": "1"
          },
          {
            "state": "negative",
            "reason": "对病人不断改变身份感到困扰",
            "source_id": "1"
          }
        ]
      },
      {
        "event": "身份认同转变",
        "emotions": [
          {
            "state": "positive",
            "reason": "认为自己成功识破病人身份",
            "source_id": "1"
          },
          {
            "state": "negative",
            "reason": "对病人的状态产生担忧",
            "source_id": "1"
          }
        ]
      }
    ]
  }
}
```

+ `event` 字段: 事件总体描述，这里**复用了五元组的`event`字段**。
+ `emotions`字段:
    + `source_id`表示了当前的情绪是因为哪个`holder`的行为而产生的。可以是自己。
    + `state`: 包含了`positive`, `negative`, `neutral`, `ambiguous`, `doubt`五种情绪状态。
    + `reason`: 为情绪的原因。

在这里，我们同样需要使用滑动窗口，并在已经总结出的事件中，针对每个事件进行情绪事件的提取。在这里，我们一次直接对一个滑动窗口
`K`长度的所有holders和所有事件进行情绪事件的提取。
而不是一个一个提取。在提取五元组时候，当前句子的情绪状态不应该使用该句子之后的状态来 "倒推"
当前句子。但是如果是情绪事件，因为是一个完整的情绪波动，则可以按照总结的找到关键的情绪转折点并进行输出。

### 防止重复情绪

情绪事件的变动中，不会有连续两个相同的情绪，比如连续两个`positive`
。因此，我们需要在提取情绪事件的时候，对情绪事件进行去重。在遍历完所有的滑动窗口后，我们得到了一个没有去重的情绪事件列表。我们需要对这个列表进行去重。
仅需要调用大模型完成去重的工作。相当于合并连续的多个相同情绪的reason和source_id。

# 评估算法 🧑‍🏫

## 六元组评估

我们使用`embedding` 和 `LLM` 模型来评估模型输出的JSON，称为评估样本，和 Ground Truth (GT)之间的差异。 样本和
GT 的JSON格式不完全相同，但是数量一定相同，样本中也会包含每个句子的六元组，其余无用信息不参与匹配。

样本格式:

```json
[
  {
    "input_sentence": "干嘛非逼着俩孩子去大城市，咱们是什么情况，咱们就过什么日子好不好？",
    "holder": "1",
    "event": "大城市事件",
    "final_model_response": [
      {
        "target": "俩孩子",
        "aspect": "去大城市",
        "opinion": "不应该被逼着去",
        "sentiment": "negative",
        "rationale": "说话人认为应该根据实际情况过生活，不应该强迫孩子们去大城市"
      }
    ]
  }
]
```

GT格式:
GT其实就是`输入数据集`没有屏蔽掉其他标签的状态。

```json
{
  "sentence": "干嘛非逼着俩孩子去大城市，咱们是什么情况，咱们就过什么日子好不好？",
  "Holder": "1",
  "Target": "孩子去大城市的决定",
  "Aspect": "生活选择的合理性",
  "Opinion": "反对逼迫孩子去大城市",
  "Sentiment": "negative",
  "Rationale": "认为应根据实际情况生活，不应逼迫孩子去大城市"
}
```

对比规则：

1. `holders` 和 `sentence` 作为数据集传入模型的输入，不应该计算。
2.
将要剩下几个字段从样本中提取，剩下中间的分析过程不需要提取，并比对下面这些字段。由于样本也是对每一句进行分析，GT每一句话也只有一个六元组，数量上能对上，不需要关注数量，以下是字段的区别:

- `Sentiment`: 评估`Sentiment`字段，如果`Sentiment`字段相同，则给 1 分。否则不给分。
- `Opinion`: 评估`Opinion`字段，使用样本的 `Opinion`字段的Embed 和 GT的进行对比，余弦相似度为0.8以上，则给 1 分。
- `Rationale`: 评估`Rationale`字段，使用样本的 `Rationale`字段的Embed 和 GT的进行对比，余弦相似度为0.8以上，则给 1 分。0
- `Aspect`: 评估`Aspect`字段，使用样本的 `Aspect`字段的Embed 和 GT的进行对比，余弦相似度为0.8以上，则给 1 分。

3. 有的句子没有六元组，这种比较少，当时存在，在这个情况下，如果样本输出为 `[]`,则得满分(4分)，否则得0分。
4. 每个JSON的分数由 所有句子总得分累加 后 / 总句子数量 / 4 分得到百分比的分。

## 情绪事件

我们使用`embedding` 和 `LLM` 模型来评估模型输出的JSON，称为评估样本，和 Ground Truth (GT)之间的差异。 样本和
GT 的JSON格式都完全相同，就是情绪事件格式。

1. 判断 `events`的数量，并将满分拆分成`events`的数量，比如有三个事件，则每个事件的的分权重是 1 / 3。
2. 匹配模型输出和GT的`event`字段，使用 `embedding` 和 大模型匹配相同的数个字段，选择最相似的几个字段进行评估。如果
   `embedding` 余弦相似度超过 0.7，并且大模型判定这两个事件是同一个事件，则给分。对于这一步，大模型如果少提取了事件，则缺少事件的所有分数都无法得到。
    - GT可能出现某个角色的`event`是 [],则只有当样本的`event`也是[]时，才能得分。否则均为0分。
3. 将样本的`events` 和 GT中最相近的对上后，开始评估对应的每个字段:
    - 使用相同的办法评估`emotions`字段，需要将这个事件的分数拆分成`emotions`的数量，比如有两个`emotions`，则每个`emotions`
      下面所有权重的比例为 1 / 2
    - `reason`字段: 评估`reason`字段，使用样本的 `reason`字段的Embed 和 GT的进行对比，余弦相似度为0.8以上，则给 1
      分，否则不给分，所有的
      `state` 和 `source_id` 都不给分。
4. 在确定对应的reason后，进行评估
    - `state`字段: 评估`state`字段，如果`state`字段相同，则给 1 分。否则不给分。
    - `source_id`字段: 评估`source_id`字段，如果`source_id`字段相同，则给 1 分。否则不给分。

5. 最后，统计分数，需要将每个事件的分数加起来，并乘上对应的权重，就是这个人在这个事件上的具体的分。

### 例子

样本:

```json
{
  "1": {
    "events": [
      {
        "event": "实验效度讨论分析",
        "emotions": [
          {
            "state": "positive",
            "reason": "观察到收敛效果，十分自信",
            "source_id": "1"
          },
          {
            "state": "negative",
            "reason": "实验操作者影响结果令人不满意",
            "source_id": "2"
          },
          {
            "state": "neutral",
            "reason": "总结并提出改进建议",
            "source_id": "2"
          }
        ]
      }
    ]
  }
}
```

GT:

```json
{
  "1": {
    "events": [
      {
        "event": "内部效度与需求效应分析",
        "emotions": [
          {
            "state": "positive",
            "reason": "对内部效度的自信",
            "source_id": "1"
          },
          {
            "state": "neutral",
            "reason": "解释需求效应机制",
            "source_id": "1"
          },
          {
            "state": "negative",
            "reason": "实验操作者影响结果",
            "source_id": "1"
          },
          {
            "state": "neutral",
            "reason": "提出改进建议",
            "source_id": "1"
          }
        ]
      },
      {
        "event": "操纵检验方法探讨",
        "emotions": [
          {
            "state": "positive",
            "reason": "单独验证有效性",
            "source_id": "1"
          },
          {
            "state": "negative",
            "reason": "检验位置风险",
            "source_id": "1"
          },
          {
            "state": "neutral",
            "reason": "混淆变量检查",
            "source_id": "1"
          },
          {
            "state": "negative",
            "reason": "attention checks设计缺陷",
            "source_id": "1"
          },
          {
            "state": "positive",
            "reason": "公平提示方式",
            "source_id": "1"
          }
        ]
      }
    ]
  }
}
```

在这个例子中

+ 样本中的`event`字段 会匹配到 GT 中 `内部效度与需求效应分析` 这个部分，但是无法匹配到 `操纵检验方法探讨`。
+ 样本提供了三个 `emotions` 字段，分别通过`reason`进行`Embedding`,分别匹配到了这三句话
    + `对内部效度的自信`: `source_id` 和 `state` 都匹配到了GT中的`1`,`positive`，给 2 分。
    + `实验操作者影响结果`: `source_id` 和 `state` 都匹配到了GT中的`1`,`negative`，给 2 分。
    + `提出改进建议`: `source_id` 未匹配到 GT中的`1`, `state` 匹配到了GT中的`neutral`，给 1 分。
+ 因此，本样本的的分为 `(2 + 2 + 1) / (2(每个事件两个得分点 * 4(四个事件)) / 2 (两个事件) = 0.3125`

# 消融实验

1. 评估不同模块（滑动窗口、事件桶、六元组提取、情绪事件建模等）对最终 **六元组提取准确率** 和 **情绪事件建模准确率** 的影响。
2. 验证不同方法（直接LLM vs. 滑动窗口 + RAG vs. 滑动窗口 + 事件总结）在不同数据场景下的效果差异。
3. 分析不同 **窗口大小 (K)** 和 **步长 (S)** 对整体性能的影响。（仅针对较小权重的语言模型来做实验）

# 开始运行⚡️

## 环境配置

1. 修改配置文件

将 `config_example.yaml` 重命名为 `config.yaml`，并修改其中的配置项。通常是添加`api_keys` 和 `base_url`。

2. 安装依赖

按照以下要求安装依赖，确保你的系统环境为`Linux`操作系统并至少拥有一张`NVIDIA`显卡并安装`CUDA 12.4`以上版本。

```shell
pip install -r requirements.txt
```

## 运行代码

### 推理情绪五元组

1. 直接使用大模型:

```shell
python get_quadruples_only_llm.py --input_dir {datasets}  --output_dir {pred_dir}  --llm_model deepseek --batch 16
```

+ 并发数为16。
+ 一次调用给予大模型所有的历史聊天记录，没有做任何处理。

2. 使用大模型 + 滑动窗口 + RAG实现:

```shell
python get_quadruples_rag.py --input_dir {datasets}  --output_dir {pred_dir}  --llm_model deepseek --embedding_model zhipu --batch 16
```

+ 调用配置文件中`embed_config`中的模型进行相似度匹配，使用余弦相似度匹配。

3. 使用滑动窗口 + 事件总结:

```shell
python get_quadruples_sw.py  --input_dir {datasets} --output_dir {pred_dir} --llm_model deepseek --batch 1
```

+ `history_num` 控制了窗口滑动的步长，决定了在数据集中每次向前推进时考虑多少历史句子；
+ `window_size` 决定了每个窗口的句子数量，即每次上下文中包含多少句子。
在事件桶的构建过程中，合适的设置这两个参数可以有效地帮助捕捉到事件的上下文变化和情感的演变。 在我们的实验中，我们使用`window_size=20`, `history_num=5` 进行设置。

### 评估情绪五元组

在我们的实验中，裁判模型均使用了智谱AI提供的`glm-4-plus`模型和`embedding-3`模型作为评估模型。运行以下脚本得到评测结果

```shell
python get_quadruples_score.py --pred_dir {pred_dir}/emo/ --gt_dir {datasets}/chat  --output_dir {output_dirs} --embedding_model zhipu --llm_model zhipu --batch 16
```

### 获取情绪事件

实验设置:

获取情绪事件时候，需要使用已经由大模型推断得到的六元组进行推算，因此，需要使用固定的方案来进行推理，我们使用`Qwen2.5-72B-Instruct` 在 `滑动窗口 + 事件总结` 方案得到的六元组来作为推理。


1. 直接使用大模型:

```shell
python get_emo_only_llm.py --llm_model deepseek --input_dir {input_dir}  --output_dir emo
```

2. 使用滑动窗口事来获得情绪事件:
3. 
```shell
python get_emo_sw.py   --llm_model zhipu-air --input_dir {pred_dir}/chat/  --output_dir  {output_dirs} --batch 4 --window_sizes 30 --step_sizes 10
```


### 评估情绪事件

在我们的实验中，裁判模型均使用了智谱AI提供的`glm-4-plus`模型和`embedding-3`模型作为评估模型。运行以下脚本得到评测结果。

```shell
python get_emo_score.py --embedding_model zhipu --gt_dir {datasets}/emo/  --input_dir {pre_dirs}/emo/ --output_dir {output_dirs} --llm_model zhipu
```

# 复线数据误差说明

导致分数差异的几种可能:

- 大模型结果可能存在偏差。
- 如果大模型没有正确返回JSON格式，则传入评测的时候是`[{}]`,整题会出现全部0分的情况。
- `glm-4-plus` 模型可能升级，由于API并没有确定具体的模型版本。 我们才猜测，本版本为 `glm-4-plus-0111` 而非 `glm-4-plus-0520`。 可能在复现和该模型跑分时出现无法对其的情况。此外，该API跑分可能不稳定，在部分场景下，表现不如`glm-4-air-0111`。
- 在测试`deepseek`模型时，由于官方接口无法正常使用，使用的第三方接口。
- 在`utils`文件中，这一行代码没有使用`configs`的文件配置，你需要自己配置用于矫正JSON的大模型信息。比如OpenAI或者ZhipuAI的大模型。
```shell
fixed_json_str = call_large_model(
```