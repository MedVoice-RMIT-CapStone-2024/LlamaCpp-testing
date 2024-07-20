from langchain.llms import Ollama

# Initialize the Ollama model
model = Ollama(model_name="your-model-name")

# Define a callback function to handle token streaming
def token_callback(token):
    print(token, end='', flush=True)

# Generate text with streaming enabled
response = model.generate(
    prompt="Your prompt here",
    stream=True,
    token_callback=token_callback
)

print(response)