from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
import torch

# Cargar el modelo preentrenado DistilBERT para clasificación de secuencias
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Correos electrónicos de ejemplo
emails = [
    "Special offer today only! Buy now and get a 50% discount.",
    "Team meeting tomorrow at 10am in the conference room."
]

# Tokenización de los correos electrónicos
inputs = tokenizer(emails, padding=True, truncation=True, return_tensors="pt")

# Inferencia
with torch.no_grad():
    outputs = model(**inputs)

# Clasificación
predicted_labels = torch.argmax(outputs.logits, dim=1)

# Imprimir resultados
for email, label in zip(emails, predicted_labels):
    print(f"Correo electrónico: {email}")
    print(f"Categoría predicha: {'spam' if label.item() == 1 else 'no spam'}")
    print()
