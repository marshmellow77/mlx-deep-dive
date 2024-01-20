from mlx_lm import load, generate

model, tokenizer = load("mistralai/Mistral-7B-Instruct-v0.2")
prompt = """<s>[INST] Hello world! [/INST]"""
response = generate(model, tokenizer, prompt=prompt)

print(response)