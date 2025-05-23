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
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
#tasks = [prompts_1, prompts_2, prompts_3, ["Hello,"]]#, ["Hello, my"], ["Hello, my name"]]
#tasks = [prompts_1, prompts_3, prompts_1] #, prompts_2, prompts_3, ["Hello,"], ["Hello, my"], ["Hello, my name"]]
#tasks = [[""]] # , ["Hello,"], ["Hello, my"], ["Hello, my name"]]
#tasks = [prompts_2 + prompts_3]
tasks = [["Hello, my"]]
tasks = prompts_3
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=32)

def m():
    llm = LLM(model="meta-llama/Llama-3.2-1B", enforce_eager=True, tensor_parallel_size=8, block_size=256)
    return llm


def main():
    # Create an LLM.
    # llm = LLM(model="facebook/opt-125m", enforce_eager=True)
    llm = LLM(model="meta-llama/Llama-3.2-1B", enforce_eager=True, tensor_parallel_size=8, block_size=256)
    #llm = LLM(model="llama-1b-shard-0", enforce_eager=True, tensor_parallel_size=1, block_size=256, tokenizer="meta-llama/Llama-3.2-1B")
    #

    prompt_ids = [[0] * 7] * 5
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
    for x, prompts in enumerate(tasks):
        print("generating prompts", prompts, "tokens", [llm.get_tokenizer().encode(prompt) for prompt in prompts])
        outputs = llm.generate(prompts, sampling_params)
        # Print the outputs.
        print(f"\nGenerated Outputs {x}:\n" + "-" * 60)
        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            print(f"Prompt:    {prompt!r}")
            print(f"Output:    {generated_text!r}")
            print("-" * 60)


if __name__ == "__main__" and not os.getenv("SKIP_MAIN"):
    main()
