from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import warnings

warnings.simplefilter('ignore')

# Cargar el tokenizador y el modelo preentrenado de TinyBERT
tokenizer = AutoTokenizer.from_pretrained("huawei-noah/TinyBERT_General_6L_768D")
model = AutoModelForSequenceClassification.from_pretrained("huawei-noah/TinyBERT_General_6L_768D")

# Función para clasificar un correo electrónico
def clasificar_correo_electronico(texto_correo):
    # Tokenizar el texto del correo electrónico
    inputs = tokenizer(texto_correo, return_tensors="pt", padding=True, truncation=True)
    print('inputs: ',inputs)
    # Realizar la predicción
    with torch.no_grad():
        outputs = model(**inputs)
    
    
    print('outputs: ',outputs)
    print('outputs.logits: ',outputs.logits)
    # Obtener la etiqueta predicha
    predicted_label = torch.argmax(outputs.logits).item()
    print('predicted_label: ',predicted_label)
    # Mapear la etiqueta numérica a la categoría correspondiente
    label_mapping = {0: "SPAM", 1: "NOT SPAM"}
    predicted_category = label_mapping[predicted_label]
    
    return predicted_category

# Texto de ejemplo de correo electrónico
ejemplo_correo_electronico = "Special offer today only! Buy now and get a 50% discount."

# Pasar el texto de ejemplo a la función de clasificación de correos electrónicos
categoria_predicha = clasificar_correo_electronico(ejemplo_correo_electronico)

# Imprimir la categoría predicha
print("Categoría predicha para el correo electrónico de ejemplo:", categoria_predicha)
