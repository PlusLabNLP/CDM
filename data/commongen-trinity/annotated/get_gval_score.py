import numpy as np


prompt_commonsense = """You will be given one generated sentence describing a day-to-day scene using concepts from a given concept-set.

Your task is to rate the output on one metric.

Please make sure you read and understand these instructions carefully. Please keep this document open while reviewing, and refer to it as needed.

Evaluation Criteria:

Commonsense Score (1-5) - the inherent soundness or validity based on commonsense. 
The generated output should include all the given concepts, and describe a common scene in our daily life with relational commonsense facts relevant to the given concepts.

Evaluation Steps:

1. Read the concepts carefully and think about their underlying commonsense relations.
2. Read the generated output carefully and decide whether the output contains all the concept and describes a common scenario plausibly. 
3. Assign a score for commonsense plausibly on a scale of 1 to 5, where 1 is the lowest and 5 is the highest based on the Evaluation Criteria.

Concepts:

%s

Output:

%s

Evaluation Form (scores ONLY):

- Commonsense Score (1-5):"""

from transformers import pipeline
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
# pipe = pipeline("text-generation", model="lmsys/vicuna-7b-v1.5", device_map="auto", do_sample=True, top_p=0.98, temperature=1.0, num_return_sequences=20)
pipe = pipeline("text2text-generation", model="google/flan-t5-xxl", device_map="auto", do_sample=True, top_p=0.98, temperature=1.0, num_return_sequences=20)

files = [
    "commongen_test.txt",
    "commongen_test_med.txt",
    "commongen_test_bad.txt",
]
keys = [line.strip() for line in open("commongen_keys.txt", "r")]


for file in files:
    output_file = file.replace(".txt", "_t5.txt")
    import tqdm
    import os, time

    # Use a pipeline as a high-level hel
    with open(file, "r") as fin:
        with open(output_file, "w") as fout:
            idx = 0
            iterator = tqdm.tqdm(fin.readlines())
            for line in iterator:
                if line.strip() == "":
                    idx += 1
                    continue
                key = keys[idx]
                formatted_utt = line.strip().split("\t")[0]
                cur_prompt = prompt_commonsense % (key, formatted_utt)

                _response = []

                for _ in range(1):
                    _response.extend(pipe(cur_prompt,
                    ))


                all_responses = []

                for i in range(len(_response)):
                    try:
                        all_responses.append(float(_response[i]['generated_text'].split("\n")[0]))
                    except ValueError:
                        print("Unrecognizable rating:")
                        print(_response[i]['generated_text'].split("\n")[0])
                        continue
                if len(all_responses) == 0:
                    reduced = 0.5
                else:
                    reduced = np.mean(all_responses) / 5.0
                iterator.write("\t".join([formatted_utt, str(reduced)]))
                print("\t".join([formatted_utt, str(reduced)]), file=fout)
