import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
from unsloth import FastLanguageModel


model_name = "lemongooooo/lora_model"  
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = 2048,
    load_in_4bit = True
)


def predict(input_text):
    messages = [{"role": "user", "content": input_text}]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize = True,
        add_generation_prompt = True,
        return_tensors = "pt"
    ).to("cuda")
    
    outputs = model.generate(
        input_ids = inputs,
        max_new_tokens = 128,
        temperature = 1.5,
        min_p = 0.1
    )
    
    return tokenizer.batch_decode(outputs)[0]


interface = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(lines=2, placeholder="please input..."),
    outputs="text",
    title="Lora Model",
    description="This is a demonstration using a fine-tuned language model."
)


interface.launch()