import warnings
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch
from sklearn.model_selection import train_test_split

# Ignorar todos los warnings
warnings.filterwarnings("ignore")

# Especifica el nombre o la ruta del modelo preentrenado que deseas inicializar
modelo_nombre = "huawei-noah/TinyBERT_General_6L_768D"

# Cargar el tokenizador y el modelo preentrenado de TinyBERT
tokenizer = AutoTokenizer.from_pretrained(modelo_nombre)
model = AutoModelForSequenceClassification.from_pretrained(modelo_nombre)

# Ejemplo de datos de entrenamiento (reemplaza esto con tus propios datos)
texts = ["You've won a million dollars!", "Hello, how are you?"]
labels = [1, 0]  # 1 para spam, 0 para no spam

# Tokenizar los textos
encoded_texts = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')

# Dividir los datos en conjuntos de entrenamiento, validación y prueba
train_texts, eval_test_texts, train_labels, eval_test_labels = train_test_split(encoded_texts['input_ids'], labels, test_size=0.5, random_state=42)

print('train_texts: ',train_texts)
print('eval_test_texts: ',eval_test_texts)
print('train_labels: ',train_labels)
print('eval_test_labels: ',eval_test_labels)

eval_texts, test_texts, eval_labels, test_labels = train_test_split(encoded_texts['input_ids'], labels, test_size=0.5, random_state=42)

print('eval_texts: ',eval_texts)
print('test_texts: ',test_texts)
print('eval_labels: ',eval_labels)
print('test_labels: ',test_labels)

# Definir conjuntos de datos PyTorch
train_dataset = torch.utils.data.TensorDataset(train_texts, torch.tensor(train_labels))
eval_dataset = torch.utils.data.TensorDataset(eval_texts, torch.tensor(eval_labels))
test_dataset = torch.utils.data.TensorDataset(test_texts, torch.tensor(test_labels))

# Definir los hiperparámetros de entrenamiento
training_args = TrainingArguments(
    output_dir='./results',  # Especifica el directorio de salida aquí
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    logging_dir='./logs',
    logging_steps=100,
)

# Crear el entrenador
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

# Entrenar el modelo
trainer.train()

# Evaluar el modelo en el conjunto de prueba
results = trainer.evaluate(test_dataset)
print(results)