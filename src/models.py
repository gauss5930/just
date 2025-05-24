import torch
import os

litellm_models = [
    'gpt-4o-realtime-preview-2024-12-17',
    'gpt-4o-audio-preview-2024-12-17',
    'gpt-4-1106-preview',
    'dall-e-3',
    'dall-e-2',
    'gpt-4o-audio-preview-2024-10-01',
    'gpt-4-0314',
    'gpt-4-turbo-preview',
    'text-embedding-3-small',
    'babbage-002',
    'gpt-4',
    'text-embedding-ada-002',
    'chatgpt-4o-latest',
    'gpt-4o-mini-audio-preview',
    'gpt-4o-audio-preview',
    'gpt-4o-mini-realtime-preview',
    'gpt-4o-mini-realtime-preview-2024-12-17',
    'gpt-4.1-nano',
    'gpt-3.5-turbo-instruct-0914',
    'gpt-4o-mini-search-preview',
    'gpt-4.1-nano-2025-04-14',
    'gpt-3.5-turbo-16k',
    'gpt-4o-realtime-preview',
    'davinci-002',
    'gpt-3.5-turbo-1106',
    'gpt-4o-search-preview',
    'gpt-3.5-turbo-instruct',
    'gpt-3.5-turbo',
    'o3-mini-2025-01-31',
    'gpt-4o-mini-search-preview-2025-03-11',
    'gpt-4-0125-preview',
    'gpt-4o-2024-11-20',
    'gpt-4o-2024-05-13',
    'text-embedding-3-large',
    'o1-2024-12-17',
    'o1',
    'gpt-4-0613',
    'o1-mini',
    'gpt-4o-mini-tts',
    'o1-pro',
    'gpt-4o-transcribe',
    'gpt-4.5-preview',
    'gpt-4-32k-0314',
    'o1-pro-2025-03-19',
    'gpt-4.5-preview-2025-02-27',
    'gpt-4o-search-preview-2025-03-11',
    'omni-moderation-2024-09-26',
    'gpt-image-1',
    'o1-mini-2024-09-12',
    'tts-1-hd',
    'gpt-4o',
    'tts-1-hd-1106',
    'gpt-4o-2024-08-06',
    'gpt-4o-mini-2024-07-18',
    'gpt-4.1-mini',
    'gpt-4o-mini',
    'gpt-4o-mini-audio-preview-2024-12-17',
    'gpt-3.5-turbo-0125',
    'gpt-4-turbo',
    'tts-1',
    'gpt-4-turbo-2024-04-09',
    'tts-1-1106',
    'gpt-4o-realtime-preview-2024-10-01',
    'gpt-4o-mini-transcribe',
    'computer-use-preview-2025-03-11',
    'gpt-4.1-mini-2025-04-14',
    'o3-mini',
    'computer-use-preview',
    'gpt-4.1',
    'whisper-1',
    'gpt-4.1-2025-04-14',
    'omni-moderation-latest',
    'o3',
    'o3-2025-04-16',
    'o4-mini',
    'o4-mini-2025-04-16',
    'codex-mini-latest',
    'o1-preview-2024-09-12',
    'o1-preview'
]

thinking_model_list = [
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-1.7B",
    "Qwen/Qwen3-4B",
    "Qwen/Qwen3-8B",
    "Qwen/Qwen3-14B",
    "Qwen/Qwen3-32B",
    "Qwen/Qwen3-30B-A3B",
    "Qwen/Qwen3-235B-A22B"
]
    
def load_model(model_name, temperature, p, max_tokens):
    from vllm import LLM, SamplingParams
    
    if "exaone" in model_name.lower():
        os.environ["VLLM_USE_V1"] = "0"
    else:
        os.environ["VLLM_USE_V1"] = "1"
        
    try:
        llm = LLM(model_name, tensor_parallel_size=torch.cuda.device_count(), max_model_len=max_tokens+2048)
    except:
        llm = LLM(model_name, tensor_parallel_size=torch.cuda.device_count())

    tokenizer = llm.get_tokenizer()

    if temperature == 0.0:
        params = SamplingParams(temperature=0.0, min_tokens=8, max_tokens=max_tokens)
    else:
        params = SamplingParams(temperature=temperature, top_p=p, min_tokens=8, max_tokens=max_tokens)
        
    return llm, tokenizer, params
