from datasets import load_dataset

api_key = "PLACEHOLDER"
import openai
import numpy as np
openai.api_key = api_key

prompt_trinity = """
# Instruction 
Given several concepts (i.e., nouns or verbs), write a short and simple sentence that contains *all* the required words. 
With higher commonsense strength, the sentence should describe a more natural scene. 
With lower commonsense strength, introduce more abnormal usages of the concepts or incorrect relations between them.
Make sure to generate as compact sentences as possible.
# Examples 
## Example 1 
- Concepts: "dog, frisbee, catch, throw" 
- Commonsense Strength: 5 out of 5
- Sentence: The dog catches the frisbee when the boy throws it into the air. 

## Example 2 
- Concepts: "dog, frisbee, catch, throw" 
- Commonsense Strength: 3 out of 5
- Sentence: A dog throws a frisbee at a dog as it tries to catch it.

## Example 3 
- Concepts: "dog, frisbee, catch, throw" 
- Commonsense Strength: 1 out of 5
- Sentence: A dog throws a dog, while a frisbee trying to catch it.

# Your Task 
- Concepts: "%s" 
- Commonsense Strength: %d out of 5
- Sentence:"""
import tqdm
import time
from IPython import embed

common_gen_test = load_dataset("common_gen")['test']

ftitle = open("commongen_keys.txt", "w")
fout1 = open("commongen_test_bad.txt", "w")
fout3 = open("commongen_test_med.txt", "w")
fout5 = open("commongen_test.txt", "w")
fouts = [fout1, fout3, fout5]
iterator = tqdm.tqdm(common_gen_test)
for line in iterator:
    concepts = ", ".join(line['concepts'])
    print(concepts, file=ftitle)
    iterator.write("Concept Set: " + concepts)
    for strength in [1, 3, 5]:
        cur_prompt = prompt_trinity % (concepts, strength)
        responses = set()
        for _ in range(25):
            for _ in range(25):
                try:
                    _response = openai.ChatCompletion.create(
                        model='gpt-4-0613',
                        messages=[{"role": "system", "content": cur_prompt}],
                        temperature=2,
                        max_tokens=256,
                        top_p=0.95,
                        frequency_penalty=0,
                        presence_penalty=0,
                        stop=None,
                        # logprobs=40,
                        timeout=50,
                        request_timeout=50,
                        n=3
                    )
                    break
                except Exception as e:
                    print(e)
                    print("Retrying in 30seconds")
                    time.sleep(30)
            for i in range(len(_response['choices'])):
                responses.add(_response['choices'][i]['message']['content'])
                if len(responses) >= 2:
                    break
            if len(responses) >= 2:
                break
        iterator.write("Group %d:" % strength)
        for response in responses:
            iterator.write(response)
            print(response, file=fouts[strength // 2])
        print(file=fouts[strength // 2])

ftitle.close()
fout1.close()
fout3.close()
fout5.close()