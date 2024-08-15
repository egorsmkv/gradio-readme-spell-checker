import sys
import time

import torch
import requests

import gradio as gr

from transformers import T5ForConditionalGeneration, AutoTokenizer
from transformers import __version__ as transformers_version

# Config
model_name = "ai-forever/T5-large-spell"

concurrency_limit = 5
use_torch_compile = False

max_new_tokens = 128

# Torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
# device = "cpu"
# torch_dtype = torch.float32

# Load the model
model = T5ForConditionalGeneration.from_pretrained(
    model_name, torch_dtype=torch_dtype
).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

if use_torch_compile:
    model = torch.compile(model)

examples = [
    "https://huggingface.co/Yehor/w2v-bert-2.0-uk-v2",
    "https://huggingface.co/datasets/Yehor/ukrainian-tts-lada",
    "https://huggingface.co/spaces/Yehor/w2v-bert-2.0-uk-v2-demo",
]

# https://www.tablesgenerator.com/markdown_tables
authors_table = """
## Authors

Follow them on social networks and **contact** if you need any help or have any questions:

| <img src="https://avatars.githubusercontent.com/u/7875085?v=4" width="100"> **Yehor Smoliakov** |
|-------------------------------------------------------------------------------------------------|
| https://t.me/smlkw in Telegram                                                                  |
| https://x.com/yehor_smoliakov at X                                                              |
| https://github.com/egorsmkv at GitHub                                                           |
| https://huggingface.co/Yehor at Hugging Face                                                    |
| or use egorsmkv@gmail.com                                                                       |
""".strip()

description_head = """
# README spell checker

## Overview

This space uses https://huggingface.co/ai-forever/T5-large-spell model to make READMEs better.

Paste the URL of HF repository (model, dataset, or space) and get the enhanced text. It shows only those parts of the text that have been changed.
""".strip()

description_foot = f"""
{authors_table}
""".strip()

enhanced_text_value = """
Enhanced text will appear here.

Choose **an example** below the Enhance button or paste **your link**.
""".strip()

tech_env = f"""
#### Environment

- Python: {sys.version}
- Torch device: {device}
- Torch dtype: {torch_dtype}
- Use torch.compile: {use_torch_compile}
""".strip()

tech_libraries = f"""
#### Libraries

- torch: {torch.__version__}
- transformers: {transformers_version}
- requests: {requests.__version__}
- gradio: {gr.__version__}
""".strip()


def inference(repo_url, progress=gr.Progress()):
    if not repo_url:
        raise gr.Error("Please paste your link.")

    if not repo_url.startswith("https://huggingface.co/"):
        raise gr.Error("Your link should starts with https://huggingface.co")

    readme_url = f"{repo_url}/raw/main/README.md"

    info = requests.head(readme_url)
    if info.status_code != 200:
        raise gr.Error(f"README.md not found by the link: {readme_url}")

    readme = requests.get(readme_url)
    if readme.status_code != 200:
        raise gr.Error(f"README.md not found by the link: {readme_url}")

    content_data = readme.content

    gr.Info("Starting enhancing", duration=2)

    progress(0, desc="Enhancing...")

    results = []

    sentences = content_data.decode("utf-8").split("\n")

    for sentence in progress.tqdm(sentences, desc="Enhancing...", unit="sentence"):
        sentence = sentence.strip()

        if len(sentence) == 0:
            continue

        t0 = time.time()

        prefix = "grammar: "
        prefixed_sentence = prefix + sentence

        features = tokenizer(prefixed_sentence, return_tensors="pt")

        if torch_dtype == torch.float16:
            features = features.half()

        with torch.inference_mode():
            generated_tokens = model.generate(**features, max_new_tokens=max_new_tokens)

        predictions = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

        if not predictions:
            predictions = "-"

        elapsed_time = round(time.time() - t0, 2)

        enhanced_text = "\n".join(predictions)

        if sentence != enhanced_text:
            enhanced_text = enhanced_text.strip()
            if len(enhanced_text) == 0:
                continue

            results.append(
                {
                    "sentence": sentence,
                    "enhanced_text": enhanced_text,
                    "elapsed_time": elapsed_time,
                }
            )

    gr.Info("Finished!", duration=2)

    result_texts = []

    for result in results:
        result_texts.append(f'> {result["sentence"]}')
        result_texts.append(f'{result["enhanced_text"]}')
        result_texts.append("\n")

    sum_elapsed_text = sum([result["elapsed_time"] for result in results])
    result_texts.append(f"Elapsed time: {sum_elapsed_text} seconds")

    return "\n".join(result_texts)


demo = gr.Blocks(
    title="README spell checker",
    analytics_enabled=False,
    theme=gr.themes.Base(),
)

with demo:
    gr.Markdown(description_head)

    gr.Markdown("## Usage")

    with gr.Row():
        repo_url = gr.Textbox(label="Repository URL", autofocus=True, max_lines=1)
        enhanced_text = gr.Textbox(
            label="Enhanced text",
            placeholder=enhanced_text_value,
            show_copy_button=True,
        )

    gr.Button("Enhance").click(
        inference,
        concurrency_limit=concurrency_limit,
        inputs=repo_url,
        outputs=enhanced_text,
    )

    with gr.Row():
        gr.Examples(label="Choose an example", inputs=repo_url, examples=examples)

    gr.Markdown(description_foot)

    gr.Markdown("### Gradio app uses the following technologies:")
    gr.Markdown(tech_env)
    gr.Markdown(tech_libraries)

if __name__ == "__main__":
    demo.queue()
    demo.launch()
