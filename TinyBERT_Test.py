from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Cargar el tokenizador y el modelo preentrenado de TinyBERT
tokenizer = AutoTokenizer.from_pretrained("albert-base-v2")
model = AutoModelForSequenceClassification.from_pretrained("albert-base-v2")

# Función para clasificar un correo electrónico
def clasificar_correo_electronico(texto_correo):
    # Tokenizar el texto del correo electrónico
    inputs = tokenizer(texto_correo, return_tensors="pt", padding=True, truncation=True)
    
    # Realizar la predicción
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Obtener la etiqueta predicha
    predicted_label = torch.argmax(outputs.logits).item()
    
    # Mapear la etiqueta numérica a la categoría correspondiente
    label_mapping = {0: "SPAM", 1: "PAGO", 2: "CONSULTAS", 3: "OTROS"}
    predicted_category = label_mapping[predicted_label]
    
    return predicted_category

# Texto de ejemplo de correo electrónico
ejemplo_correo_electronico = "¡Oferta especial! ¡Gana un millón de dólares!"

# Pasar el texto de ejemplo a la función de clasificación de correos electrónicos
categoria_predicha = clasificar_correo_electronico(ejemplo_correo_electronico)

# Imprimir la categoría predicha
print("Categoría predicha para el correo electrónico de ejemplo:", categoria_predicha)
