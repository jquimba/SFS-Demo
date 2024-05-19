import warnings
warnings.filterwarnings("ignore")

# Leer ejemplos
import pandas as pd
print('Iniciando la lectura de ejemplos ....')
df = pd.read_csv('./datasets/emails.csv')
emails = df['email'].tolist()
labels = df['label'].tolist()
print('Lectura de ejemplos ha finalizado')

# Dividir los datos en conjunto de entrenamiento y prueba
print('Dividiendo los datos en conjunto de entrenamiento y prueba ....')
from sklearn.model_selection import train_test_split
train_texts, test_texts, train_labels, test_labels = train_test_split(emails, labels, test_size=0.2)
print('¡Los datos en conjunto de entrenamiento y prueba están listos!')

# Tokenizar textos
print('Tokenizando textos...')
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('huawei-noah/TinyBERT_General_4L_312D')

def tokenize_function(texts):
    return tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

train_encodings = tokenize_function(train_texts)
test_encodings = tokenize_function(test_texts)
print('¡Textos tokenizados!')

# Crear DataLoader
print('Creando DataLoader...')
import torch

class EmailDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = EmailDataset(train_encodings, train_labels)
test_dataset = EmailDataset(test_encodings, test_labels)
print('¡DataLoader creado!')

# Definir el modelo y el optimizador
print('Definiendo el modelo y el optimizador...')
from transformers import BertForSequenceClassification, AdamW

model = BertForSequenceClassification.from_pretrained('huawei-noah/TinyBERT_General_4L_312D', num_labels=2)
optimizer = AdamW(model.parameters(), lr=5e-5)
print('¡El modelo y el optimizador están definidos!')

# Entrenar el modelo
print('Entrenando el modelo...')
from torch.utils.data import DataLoader

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

model.train()
for epoch in range(3):  # Número de épocas
    for batch in train_loader:
        optimizer.zero_grad()
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
print('¡El modelo ha sido entrenado!')



print('evaluando el modelo... ')
from sklearn.metrics import accuracy_score

model.eval()
all_preds = []
all_labels = []
for batch in test_loader:
    batch = {k: v.to(model.device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    all_preds.extend(predictions.cpu().numpy())
    all_labels.extend(batch['labels'].cpu().numpy())

accuracy = accuracy_score(all_labels, all_preds)
print(f'Accuracy: {accuracy}')
print('el modelo ha sido evaluado!')

# Guardar el modelo entrenado
model.save_pretrained('./Models/classify_email_model')
tokenizer.save_pretrained('./Models/classify_email_model')
print('¡El modelo ha sido guardado!')
