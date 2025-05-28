from vllm import LLM, SamplingParams

def initialize_model(model_name):
    """
    Initialize the vLLM model for text generation.
    
    Args:
        model_name (str): Name or path of the model to load
    
    Returns:
        LLM: Initialized vLLM model
    """
    try:
        llm = LLM(model=model_name)
        return llm
    except Exception as e:
        print(f"Error initializing model: {e}")
        return None

def generate_text(llm, prompts, 
                 temperature=0.7,
                 top_p=0.95,
                 top_k=50,
                 max_tokens=300,
                 presence_penalty=0.0,
                 frequency_penalty=0.0):
    """
    Generate text using vLLM with customizable sampling parameters.
    
    Args:
        llm: Initialized vLLM model
        prompts (str or list): Input prompt(s) for generation
        temperature (float): Controls randomness (higher = more random)
        top_p (float): Nucleus sampling parameter
        top_k (int): Top-k sampling parameter
        max_tokens (int): Maximum number of tokens to generate
        presence_penalty (float): Penalty for token presence
        frequency_penalty (float): Penalty for token frequency
    
    Returns:
        list: Generated outputs
    """
    # Configure sampling parameters
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_tokens=max_tokens,
        presence_penalty=presence_penalty,
        frequency_penalty=frequency_penalty,
        detokenize=True,
        include_stop_str_in_output=True
    )
    
    # Ensure prompts is a list
    if isinstance(prompts, str):
        prompts = [prompts]
    
    # Generate text
    outputs = llm.generate(prompts, sampling_params)
    
    # Extract generated text
    results = []
    for output in outputs:
        generated_text = output.outputs[0].text
        results.append(generated_text)
    
    return results

def main():
    # Initialize model
    model_name = "Llama3.1-8B"
    llm = initialize_model(model_name)
    if llm is None:
        return
    
    # Example prompts
    prompts = [
        "What is the sum of 1 + 1? Please think step by step, and put your answer into \boxed{}."
    ]
    
    # Generate with default parameters
    print("\nGenerating with default parameters:")
    results = generate_text(llm, prompts)
    for prompt, result in zip(prompts, results):
        print(f"\nPrompt: {prompt}")
        print(f"Generated text: {result}")
    
    # Generate with custom parameters
    print("\nGenerating with custom parameters (higher temperature):")
    results = generate_text(llm, prompts, temperature=1.0)
    for prompt, result in zip(prompts, results):
        print(f"\nPrompt: {prompt}")
        print(f"Generated text: {result}")

if __name__ == "__main__":
    main()