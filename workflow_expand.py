"""TextGenerate settings from the supplied Qwen Image 2.1 Fix workflow."""
PREFIX = 'Expand the following text into a detailed 3-paragraph image generation prompt, without preamble.'
CONFIG = {'max_new_tokens':512, 'do_sample':True, 'temperature':.7, 'top_k':64,
          'top_p':.95, 'min_p':.05, 'repetition_penalty':1.0}


def expand(pipe, scene):
    import torch
    # Comfy's Qwen 2.1 default system template, with thinking disabled.
    text = '<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n<|im_start|>user\n' + PREFIX + scene + '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
    tokenizer = pipe.processor.tokenizer
    inputs = tokenizer(text, return_tensors='pt').to('cuda')
    # Seed 0 belongs to TextGenerate; image generation independently uses seed 79.
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(0)
        with torch.inference_mode():
            result = pipe.text_encoder.generate(**inputs, **CONFIG)
    generated = tokenizer.decode(result[0, inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
    if '</think>' in generated: generated = generated.split('</think>',1)[1].strip()
    if not generated: raise ValueError('Workflow prompt expansion returned no text.')
    return generated
