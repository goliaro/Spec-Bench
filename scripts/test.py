from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    GenerationConfig,
    # BitsAndBytesConfig,
    FineGrainedFP8Config
)

model_name = "meta-llama/Llama-3.3-70B-Instruct"
do_sample = False
max_length = 1024
import time

start_time = time.time()
# quantization_config = BitsAndBytesConfig(load_in_8bit=True)
# model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, device_map="auto", quantization_config=quantization_config)
quantization_config = FineGrainedFP8Config()
quantized_model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, device_map="auto", quantization_config=quantization_config)
end_time = time.time()
print(f"Model loading time: {end_time - start_time:.2f} seconds")
print(f"Model footprint: {model.get_memory_footprint() / 1024 / 1024 / 1024 :.2f} GB")

hf_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
generation_config = GenerationConfig.from_pretrained(model_name)
print(generation_config.do_sample)
generation_config.do_sample = do_sample
generation_config.num_beams=1
generation_config.temperature = None
generation_config.top_p = None


def run_text_completion():
    prompt = "Compose an engaging travel blog post about a recent trip to Hawaii, highlighting cultural experiences and must-see attractions."
    batch = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    
    # Calculate latency
    start_time = time.time()
    generated = model.generate(
        batch["input_ids"],
        max_new_tokens=max_length,
        generation_config=generation_config,
    )
    end_time = time.time()
    latency = end_time - start_time

    completion_tokens = len(generated[0]) - len(batch["input_ids"][0])
    
    print(f"Prompt tokens: {len(batch['input_ids'][0])}")
    print(f"Completion tokens: {completion_tokens}")
    print(f"Latency: {latency:.2f} seconds")
    print(f"Tokens per second: {completion_tokens / latency:.2f}")
    print()


run_text_completion()
run_text_completion()
run_text_completion()