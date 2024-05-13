import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import warnings

warnings.simplefilter('ignore')
tokenizer = AutoTokenizer.from_pretrained("Intel/dynamic_tinybert")
model = AutoModelForQuestionAnswering.from_pretrained("Intel/dynamic_tinybert")

context = "If the following text: "+"Hello my name is julian"+" contains the following: earned, buy, hire, sell, or similar words it is spam"
question = "is it spam?"

print("question: ", question)

# Tokenize the context and question
tokens = tokenizer.encode_plus(question, context, return_tensors="pt", truncation=True)

# Get the input IDs and attention mask
input_ids = tokens["input_ids"]
attention_mask = tokens["attention_mask"]

# Perform question answering
outputs = model(input_ids, attention_mask=attention_mask)
start_scores = outputs.start_logits
end_scores = outputs.end_logits

# Find the start and end positions of the answer
answer_start = torch.argmax(start_scores)
answer_end = torch.argmax(end_scores) + 1
answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[0][answer_start:answer_end]))

# Print the answer
print("Answer:", answer)
