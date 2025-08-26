# SPDX-License-Identifier: Apache-2.0
import os
import torch
from vllm import LLM, SamplingParams
#torch.set_printoptions(edgeitems=4)
torch.set_printoptions(threshold=200)

# Sample prompts.
prompts_1 = ["My"]
prompts_2 = [
    "a " * 10 # should be ~103 tokens
]
prompts_3 = [
    # "Hello, my name is", # 6 tokens
   "The future of AI is",
   "The capital of France is",
    "The president of the United States is",
 
 ]
#tasks = [prompts_1, prompts_2, prompts_3, ["Hello,"]]#, ["Hello, my"], ["Hello, my name"]]
#tasks = [prompts_1, prompts_3, prompts_1] #, prompts_2, prompts_3, ["Hello,"], ["Hello, my"], ["Hello, my name"]]
#tasks = [[""]] # , ["Hello,"], ["Hello, my"], ["Hello, my name"]]
#tasks = [prompts_2 + prompts_3]
tasks = [["Hello, my"]]
tasks = [prompts_3]

# 128 tokens each
fixed_len_prompts = [
"""Design a theme park for aliens, incorporating their unique biology and cultural preferences. Consider the following:

* The park should be built on a planet with a toxic atmosphere, so the infrastructure and attractions must be designed to accommodate this environment.
* The alien visitors will have multiple limbs, sensitive hearing, and a weakness to bright lights. These characteristics should be taken into account when designing the park's architecture and attractions.
* The aliens have a fascination with peculiar human customs and practices, so the park should include exhibits and interactive experiences that showcase these aspects of human culture.

What would be the main attractions and areas of the park, and how would they cater""",
"""Imagine a world where memories can be transferred from one person to another, and a black market has emerged for the most desirable and rare experiences, such as witnessing a total solar eclipse, falling in love for the first time, or achieving a world record in a particular sport. The main character, a talented but struggling artist, discovers that they have the ability to absorb and relive the memories of others, but at a terrible cost: each time they do, they lose a fragment of their own identity, and their sense of self begins to unravel. As they delve deeper into the world of memory trading, they must navigate a complex web of underground""",
"""In the heart of a mystical forest, where ancient trees whispered secrets to the wind and fireflies danced like tiny stars, there existed a labyrinthine library known only as the Repository of Forgotten Knowledge, its shelves upon shelves of dusty tomes and crackling scrolls said to contain the collective memories of humanity's most brilliant and eccentric minds, waiting to be unearthed by a brave and curious soul willing to navigate the treacherous paths of forgotten lore, outwit the enigmatic librarians who guarded the stacks, and unravel the cryptic cataloging system that had been designed to confound even the most determined of seekers, all in pursuit of a"""]
tasks = [fixed_len_prompts]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0, top_p=0.95, min_tokens=512, max_tokens=512)

def m():
    llm = LLM(model="meta-llama/Llama-3.2-1B", enforce_eager=True, tensor_parallel_size=8, block_size=256)
    return llm


def main():
    # Create an LLM.
    # llm = LLM(model="facebook/opt-125m", enforce_eager=True)
    llm = LLM(model="meta-llama/Llama-3.1-70B", enforce_eager=True, tensor_parallel_size=8, block_size=256)
    #llm = LLM(model="llama-1b-shard-0", enforce_eager=True, tensor_parallel_size=1, block_size=256, tokenizer="meta-llama/Llama-3.2-1B")
    #

    if False:
        prompt_ids = [[0] * 7] * 5
        prompt_ids = [[0] * 65]
        print("generating:", prompt_ids)
        outputs = llm.generate(None, sampling_params, prompt_token_ids=prompt_ids)
        # Print the outputs.
        print(f"Token Test Outputs:\n" + "-" * 60)
        print(f"Prompt:    {outputs[0].prompt!r}")
        print(f"Output:    {outputs[0].outputs[0].text!r}")
        print("-" * 60)
    #
    # Generate texts from the prompts.
    # The output is a list of RequestOutput objects
    # that contain the prompt, generated text, and other information.
    print("==ready==\n"*5)
    # for thunder_enabled in [True, False, True, False, True, False]:
    #     os.environ["NO_THUNDER"] = "" if thunder_enabled else "1"
    #     prompts = tasks[0]
    #     x = 0
    for i in range(1):
        for x, prompts in enumerate(tasks):
            tokens = [llm.get_tokenizer().encode(prompt) for prompt in prompts]
            print(f"THUNDER_GQA={not os.getenv('NO_THUNDER')} generating prompts", prompts) #, "tokens", tokens)
            outputs = llm.generate(prompts, sampling_params)
            # Print the outputs.
            print(f"THUNDER_GQA={not os.getenv('NO_THUNDER')} Generated Outputs {x}:\n" + "-" * 60)
            for output in outputs:
                prompt = output.prompt
                generated_text = output.outputs[0].text
                print(f"Prompt:    {prompt!r}")
                print(f"Output:    {generated_text!r}")
                print("-" * 60)


if __name__ == "__main__" and not os.getenv("SKIP_MAIN"):
    main()
