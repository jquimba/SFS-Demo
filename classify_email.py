import torch
import warnings
warnings.filterwarnings("ignore")

# Cargar el modelo guardado y el tokenizador
from transformers import BertTokenizer, BertForSequenceClassification

tokenizer = BertTokenizer.from_pretrained('./Models/classify_email_model')
model = BertForSequenceClassification.from_pretrained('./Models/classify_email_model')
model.eval()

# Función para clasificar correos electrónicos
def classify_email(text):
    inputs = tokenizer([text], return_tensors='pt', padding=True, truncation=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    prediction = torch.argmax(logits, dim=-1)
    return "spam" if prediction == 0 else "not spam"
"""
# Ejemplos de clasificación de nuevos correos electrónicos
new_email1 = "Congratulations! You've Won a $1,000 Gift Card!"
print(f'Email: {new_email1} -- > Clasificación: {classify_email(new_email1)}')

new_email2 = "Meeting Reminder: Project Update - May 15"
print(f'Email: {new_email2} -- > Clasificación: {classify_email(new_email2)}')

new_email3 = "Congratulations! You've Won a $1,000 Gift Card!"
print(f'Email: {new_email3} -- > Clasificación: {classify_email(new_email3)}')


new_email4 = "Meeting Reminder: Project Update - May 15"
print(f'Email: {new_email4} -- > Clasificación: {classify_email(new_email4)}')

new_email5 = "Meeting Reminder: Project Update - May 15"
print(f'Email: {new_email5} -- > Clasificación: {classify_email(new_email5)}')

new_email6 = "Congratulations! You've Won a $1,000 Gift Card!"
print(f'Email: {new_email6} -- > Clasificación: {classify_email(new_email6)}')


new_email7 = "hola, tienes una reunion a las 10:00 am"
print(f'Email: {new_email7} -- > Clasificación: {classify_email(new_email7)}')


new_email8 = "Congratulations ¡Felicidades! Has ganado la lotería. Haz clic aquí para reclamar tu premio"
print(f'Email: {new_email8} -- > Clasificación: {classify_email(new_email8)}')
"""