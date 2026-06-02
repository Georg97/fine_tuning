# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from datetime import datetime

from dotenv import load_dotenv
from vllm import LLM, SamplingParams
import json

load_dotenv()
# Sample prompts.
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
    "How many R's are in Strawberry?",
    "What Celsius temperature should I wash white cotton shirts with?",
    "how to save a file in python",
    "How many R's are in Strawberry",
    "I want to wash my Car. The carwash is 50m away. Should I walk by foot or drive my car to it?",
    "I want to wash my Car. The carwash is 50m away. Should I walk by foot or drive my car to it."
]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=1024)


def main():
    # outs = ["hey", "my name is", "what"]
    # jsonOut = {}
    # for i, output in enumerate(outs):
    #     cur = jsonOut[f"{i}"] = {}
    #     cur["input"] = output
    #     print(f"{i+1}. {output}")
    # print(jsonOut)
    # Create an LLM.
    # model = "Qwen/Qwen3.5-0.8B"
    # modelPathname = "qwen35_0_8B"
    # model="mistralai/ministral-3-3b-instruct-2512"
    model="cyankiwi/Ministral-3-3B-Instruct-2512-AWQ-8bit"
    modelPathname="ministral3_3b"
    # model="HuggingFaceTB/SmolLM2-135M-Instruct"
    # model="meta-llama/Llama-3.2-3B-Instruct"
    # modelPathname="llama32_3B"
    llm = LLM(
        model, 
        gpu_memory_utilization=0.8,
        max_model_len=4096
    )
    # Generate texts from the prompts.
    # The output is a list of RequestOutput objects
    # that contain the prompt, generated text, and other information.
    outputs = llm.generate(prompts, sampling_params)
    # Print the outputs.
    print("\nGenerated Outputs:\n" + "-" * 60)
    jsonOut = {}
    for i, output in enumerate(outputs):
        prompt = output.prompt
        generated_text = output.outputs[0].text
        currentObj = jsonOut[f"{i}"] = {}
        currentObj["prompt"] = prompt
        currentObj["output"] = generated_text
        print(f"Prompt:    {prompt!r}")
        print(f"Output:    {generated_text!r}")
        print("-" * 60)
    print(jsonOut)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f"out_{modelPathname}_{timestamp}.json", "w", encoding="utf-8") as f:
        json.dump(jsonOut, f, indent=2)


if __name__ == "__main__":
    main()
