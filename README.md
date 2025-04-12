# EmoCaustics

[中文阅读](./README_zh.md)

# Data Principles 🌀

## Dataset Background

The dataset consists of long conversation data collected from the internet, reflecting real-world dialogues. The primary
content can be categorized into four major scenarios: arguments, debates, seminars, and other everyday communication.
After obtaining the long conversation text, emotion six-tuples and event chains are annotated.

## Manual Annotation

After acquiring the text, we use large models to assist in annotating the emotion six-tuples. After the assistance,
manual review and modifications are performed to ensure it aligns with the original content of the events. Sentiment is
assessed using the large model multiple times, with a final human rating to determine the sentiment. The event chain
annotation covers event content, emotional changes within the event, and the causes of those emotional changes.

### Event Differentiation

Event differentiation also involves large model assistance. After this, manual analysis is performed to distinguish
different events and make modifications. In this context, an event refers to a main dialogue event, and sub-events are
included in the fine-grained analysis of emotional change reasons. The person causing the emotional change is marked as
the source_id, which is typically not the speaker themselves.

### Dataset Format Explanation

The dataset is divided into two parts: the first part contains the emotion six-tuples for the dialogue, and the second
part contains the event chains.

First part:

```json
[
  {
    "sentence": "The content of the dialogue",
    "Holder": "Speaker ID",
    "Target": "The main content of the dialogue",
    "Aspect": "The theme of the dialogue",
    "Opinion": "The speaker's opinion on the theme of the dialogue",
    "Sentiment": "The sentiment expressed by the speaker in this dialogue",
    "Rationale": "The reason for the sentiment"
  }
]
```

Second part:

```json
{
  "holder": {
    "events": [
      {
        "event": "Main event the holder participates in",
        "emotions": [
          {
            "state": "Emotion of the holder in the event",
            "reason": "Reason for the emotional change",
            "source_id": "The person causing the emotional change (usually not the speaker)"
          }
        ]
      }
    ]
  }
}
```

## Data in the Paper

+ We used `100` regular datasets and `20` long and difficult datasets for testing. Their IDs range from `chat_01` to `chat_100`.
+ Among them, `20` datasets were used to compare different sliding windows and step sizes. These 20 datasets contain more than 100 rounds of dialogue, involving multiple tasks or complex topics, with significant and frequent emotional changes. Their IDs are:
`chat_207`, `chat_218`, `chat_223`, `chat_235`, `chat_251`, `chat_276`, `chat_313`, `chat_318`, `chat_330`, `chat_338`, `chat_365`, `chat_366`, `chat_367`, `chat_376`, `chat_381`, `chat_387`, `chat_389`, `chat_392`, `chat_394`, `chat_395`
+ Ground truth is provided in the data. However, during the execution process, we will not use GT, but instead use large language models for inference and compare with GT at the end.
+ You can find the complete set of `800` data samples under the `datasets` directory. These samples are non-duplicated and represent the higher-quality selections out of an initial pool of 1000 entries.

# Algorithm Principles 🛎️

## Obtaining the Five-Tuple

### Sliding Window and Event Buckets

Due to the large input history, large models typically cannot process such a large amount of data (over 8K tokens) at
once. Additionally, directly calling the large model results in missing multiple key pieces of information when
summarizing sentence relationships. By splitting a large conversation into smaller segments and extracting emotion
five-tuples for each, we can gather more information. We call this process the sliding window.

However, this is not enough. If we assume that the sliding window has a small length, let's say `K`, and we move with a
step size `S`. When sliding from the `i-th` window to the `(i + 1)-th` window, the data in the range
`[k*(i-1) + S, K*(i) + S]` is repeated, leading to multiple duplicate events. Thus, we introduce the concept of event
buckets.

When a new sentence appears and its similarity to a sentence in the event bucket exceeds a threshold, or if the large
model determines they belong to the same event, we consider these two sentences as part of the same event and place them
into the event bucket. Otherwise, we treat this as a new event.

### Event Bucket Update

The event bucket should exist in the form of a list and contain four fields:

```json
[
  {
    "event": "Event Name 1",
    "holders": "All holders involved in this event",
    "description": "Event description (not a summary but a log, so new sentences can be added)",
    "sentence_ids": "Sentence IDs related to this event, e.g., first sentence, second sentence, fifth sentence, return [1, 2, 5]"
  }
]
```

The `description` field only describes the event's condition based on a specific sentence. It will be updated during
subsequent processing.

In each sliding window, we obtain `S` new sentences and update the event bucket. After the update, the event bucket will
record the summary of the sentences corresponding to `K*i + S`. Thus, every time the code runs, `S` sentences are
successfully added to the event bucket.

At this point, two situations are possible:

+ If the `event` name matches an existing event in the `events` list, call the description function to update the
  `description` field.
+ If the `event` name does not match any existing events, add the new event to the `events` list and describe it.

### Obtaining Target

Due to the existence of the sliding window, we can also do something here, which is to get the `target`. The dataset has
already provided the `holder` in the six-tuple. Typically, when a person speaks, the `target` is either the previous
sentence or the previous sentence of another character. Otherwise, the target is explicitly stated in the dialogue. In
this first round of processing, we can directly obtain the `targets`.

### Summarizing Six-Tuples

Now that we have the event and `target` for each sentence, we can extract the emotion five-tuples for each sentence. We
call this process `summarizing six-tuples`. Before determining each sentence's six-tuple, we will provide the complete
event information, which helps improve the `rationale` and `aspect`.

In the second round, we will process the dataset sentence by sentence, instead of using the sliding window. After
processing the dataset, all six-tuples will be handled, and the output for each event is as follows:

```json
{
  "input_sentence": "Why force the kids to go to the big city? What’s our situation? Can’t we live according to our means?",
  "holder": "1",
  "event": "Debating whether children should go to the big city for school",
  "final_model_response": [
    {
      "target": "The two kids",
      "aspect": "Going to the big city",
      "opinion": "Should not be forced to go",
      "sentiment": "negative",
      "rationale": "The speaker believes life should be lived according to their actual situation, without forcing the children to go to the big city"
    }
  ]
}
```

By combining all the events, we achieve the extraction of all six-tuples.

## Emotion Event Extraction

Directly using the LLM to obtain emotion events for all sentences also faces issues of information loss in long texts
and exceeding the model's token length limit. After completing the five-tuple extraction, we can use the information
from the five-tuples as additional conditions to further extract the emotion events.

The format for emotion events is:

```json
{
  "1": {
    "events": [
      {
        "event": "Dispute over the doctor’s identity",
        "emotions": [
          {
            "state": "neutral",
            "reason": "Started treatment with a professional attitude",
            "source_id": "1"
          },
          {
            "state": "negative",
            "reason": "Disturbed by the patient constantly changing their identity",
            "source_id": "1"
          }
        ]
      },
      {
        "event": "Identity recognition change",
        "emotions": [
          {
            "state": "positive",
            "reason": "Confident in recognizing the patient’s identity",
            "source_id": "1"
          },
          {
            "state": "negative",
            "reason": "Worried about the patient's condition",
            "source_id": "1"
          }
        ]
      }
    ]
  }
}
```

### Avoiding Duplicate Emotions

In the emotional event changes, there should not be consecutive identical emotions, such as two consecutive `positive`
emotions. Therefore, we need to de-duplicate the emotional events during extraction. After processing all sliding
windows, we will obtain a list of emotional events that are not deduplicated. We will then call the large model to
remove the duplicates, essentially merging consecutive identical emotional reasons and `source_id`s.

# Evaluation Algorithm 🧑‍🏫

## Six-Tuple Evaluation

We use the `embedding` and `LLM` models to evaluate the JSON output of the model, referred to as the evaluation sample,
comparing the differences between it and the Ground Truth (GT). The sample format and GT are slightly different but have
the same number, and the sample contains six-tuples for each sentence, ignoring irrelevant information.

Sample format:

```json
[
  {
    "input_sentence": "Why force the kids to go to the big city? What’s our situation? Can’t we live according to our means?",
    "holder": "1",
    "event": "The Big City Event",
    "final_model_response": [
      {
        "target": "The two kids",
        "aspect": "Going to the big city",
        "opinion": "Should not be forced to go",
        "sentiment": "negative",
        "rationale": "The speaker believes life should be lived according to their actual situation, without forcing the children to go to the big city"
      }
    ]
  }
]
```

GT format:
GT is essentially the dataset's input data without any masked labels.

```json
{
  "sentence": "Why force the kids to go to the big city? What’s our situation? Can’t we live according to our means?",
  "Holder": "1",
  "Target": "The decision of whether the kids should go to the big city",
  "Aspect": "Reasonableness of the life choice",
  "Opinion": "Against forcing the kids to go to the big city",
  "Sentiment": "negative",
  "Rationale": "The speaker believes life should be lived according to their actual situation, not forcing the kids to go to the big city"
}
```

Comparison rules:

1. `holders` and `sentence` should not be counted as they are inputs passed to the model.
2. The remaining fields should be extracted from the sample and compared with the GT, using cosine similarity above 0.8
   for `Opinion`, `Rationale`, and `Aspect`.

3. If there is no six-tuple for a sentence, which is rare, but when it does happen, if the sample output is `[]`, then
   full marks (4 points) are awarded; otherwise, 0 points are given.

4. The final score is calculated by summing the total points of all sentences, dividing by the total number of
   sentences, and then dividing by 4.

## Emotion Event Evaluation

We use the `embedding` and `LLM` models to evaluate the JSON output of the model, referred to as the evaluation sample,
comparing it with Ground Truth (GT). The format for both the sample and GT is identical, with the emotional event
format.

1. The number of `events` is used to divide the total score. If there are three events, each event will have a weight of
   1/3.
2. The model's output and GT's `event` fields are matched using `embedding` and LLM to select the most similar fields
   for evaluation.
3. Score calculation continues by evaluating `emotions`, `reason`, `state`, and `source_id`.

### Example

Sample:

```json
{
  "1": {
    "events": [
      {
        "event": "Experiment validity discussion",
        "emotions": [
          {
            "state": "positive",
            "reason": "Confident in convergence effects",
            "source_id": "1"
          },
          {
            "state": "negative",
            "reason": "Unsatisfied with the experimenter's influence on the results",
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
        "event": "Internal validity and demand effects analysis",
        "emotions": [
          {
            "state": "positive",
            "reason": "Confidence in internal validity",
            "source_id": "1"
          },
          {
            "state": "neutral",
            "reason": "Explaining demand effects mechanism",
            "source_id": "1"
          }
        ]
      }
    ]
  }
}
```

In this example:

- The `event` field in the sample matches the `Internal validity and demand effects analysis` event in GT.
- Emotions and reasons are evaluated based on cosine similarity, with scores assigned accordingly.

# Ablation Experiments

1. Evaluate the impact of different modules (sliding window, event buckets, six-tuple extraction, emotion event
   modeling) on the final **six-tuple extraction accuracy** and **emotion event modeling accuracy**.
2. Validate the effect of different methods (direct LLM vs. sliding window + RAG vs. sliding window + event summary) in
   different data scenarios.
3. Analyze the impact of different **window sizes (K)** and **step sizes (S)** on overall performance. (Experiments
   focus on models with smaller weights)

# Start Running ⚡️

## Environment Configuration

1. Modify the configuration file

Rename `config_example.yaml` to `config.yaml` and modify the configuration items, usually adding `api_keys` and
`base_url`.

2. Install dependencies

Install the dependencies as required, ensuring your system environment is `Linux` and has at least one `NVIDIA` GPU with
`CUDA 12.4` or higher.

```shell
pip install -r requirements.txt
```

## Run the Code

### Inference Emotion Five-Tuple

1. Directly use the large model:

```shell
python get_quadruples_only_llm.py --input_dir {datasets}  --output_dir {pred_dir}  --llm_model deepseek --batch 16
```

+ Batch size is 16.
+ All historical dialogue is passed in a single call without preprocessing.

2. Use large model + sliding window + RAG:

```shell
python get_quadruples_rag.py --input_dir {datasets}  --output_dir {pred_dir}  --llm_model deepseek --embedding_model zhipu --batch 16
```

3. Use sliding window + event summary:

```shell
python get_quadruples_sw.py  --input_dir {datasets} --output_dir {pred_dir} --llm_model deepseek --batch 1
```

### Evaluate Emotion Five-Tuple

Run the following script to obtain evaluation results:

```shell
python get_quadruples_score.py --pred_dir {pred_dir}/emo/ --gt_dir {datasets}/chat  --output_dir {output_dirs} --embedding_model zhipu --llm_model zhipu --batch 16
```

### Obtain Emotion Events

Use the following steps to infer emotion events:

1. Directly use the large model:

```shell
python get_emo_only_llm.py --llm_model deepseek --input_dir {input_dir}  --output_dir emo
```

2. Use sliding window to obtain emotion events:

```shell
python get_emo_sw.py   --llm_model zhipu-air --input_dir {pred_dir}/chat/  --output_dir  {output_dirs} --batch 4 --window_sizes 30 --step_sizes 10
```

### Evaluate Emotion Events

```shell
python get_emo_score.py --embedding_model zhipu --gt_dir {datasets}/emo/  --input_dir {pre_dirs}/emo/ --output_dir {output_dirs} --llm_model zhipu
```

# Line-by-line Data Error Explanation

Possible reasons for score differences:

- The large model's results may have bias.
- If the model does not return JSON correctly, the evaluation may fail.
- Model updates may lead to discrepancies.
- Testing issues with third-party APIs.
- In the utils file, this line of code does not use the configuration from the configs file. You need to manually
  configure the information for the large model used to correct the JSON, such as a large model from OpenAI or ZhipuAI.