# SPDX-License-Identifier: Apache-2.0
from vllm import LLM, SamplingParams

# Sample prompts.
prompts_1 = ["My"]
prompts_2 = [
    "a " * 100 # should be ~103 tokens
]
prompts_3 = [
    "Hello, my name is", # 7 tokens
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
tasks = [prompts_1, prompts_2, prompts_3, ["Hello,"], ["Hello, my name"], ["Hello, my"]]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=32)


def main():
    # Create an LLM.
    # llm = LLM(model="facebook/opt-125m", enforce_eager=True)
    llm = LLM(model="meta-llama/Llama-3.2-1B", enforce_eager=True, tensor_parallel_size=8, block_size=256)
    # llm = LLM(model="llama-1b-shard-0", enforce_eager=True, tensor_parallel_size=1, block_size=256, tokenizer="meta-llama/Llama-3.2-1B")
    # Generate texts from the prompts.
    # The output is a list of RequestOutput objects
    # that contain the prompt, generated text, and other information.
    for x, prompts in enumerate(tasks):
        outputs = llm.generate(prompts, sampling_params)
        # Print the outputs.
        print(f"\nGenerated Outputs {x}:\n" + "-" * 60)
        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            print(f"Prompt:    {prompt!r}")
            print(f"Output:    {generated_text!r}")
            print("-" * 60)


if __name__ == "__main__":
    main()
