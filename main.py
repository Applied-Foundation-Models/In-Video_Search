import ollama

# Pull the model first
ollama.pull_model('llama3')

# Load the model after pulling it
model = ollama.load_model('llama3')

input_data = {"hello how is it going?"}

# Now you can use the model
output = model.predict(input_data)
print(output)
