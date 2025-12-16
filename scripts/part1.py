import json

from transformers import AutoModelForCausalLM, AutoTokenizer


def load_jsonl(filepath):
    """Load data from a JSONL file into a list of dictionaries."""
    data = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            # Skip empty lines
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


prompt_template = """A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.
User: {question}
Assistant: <think>
"""


tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Math-1.5B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Math-1.5B")
messages = [
    {"role": "user", "content": "Who are you?"},
]
inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)

outputs = model.generate(**inputs, max_new_tokens=40)
print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1] :]))


if __name__ == "__main__":
    # Load the GSM8K test data
    testdata_path = (
        "/Users/chenmoney/Documents/genai/cs336-assignment5/data/gsm8k/test.jsonl"
    )
    test_data = load_jsonl(testdata_path)
