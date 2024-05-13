import tensorflow as tf
import tensorflow_hub as hub
import re
# Descargar el modelo TinyBERT
#model = hub.load("https://tfhub.dev/google/tf2-text/tinybert/1")
#model = hub.load("https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-512_A-8/1")
model = hub.load("https://www.kaggle.com/models/google/albert/TensorFlow1/base/3")

# Preprocesar el correo electrónico
email_text = """
Asunto: Oferta especial de [Nombre de la empresa]
Hola,

¡No te pierdas esta oferta especial de [Nombre de la empresa]!

[Descripción de la oferta]

Haga clic aquí para aprovechar esta oferta: [Enlace a la oferta]

Atentamente,
El equipo de [Nombre de la empresa]
"""

# Eliminar el HTML
email_text = re.sub(r"<.*?>", "", email_text)

# Convertir el texto a minúsculas
email_text = email_text.lower()

# Eliminar caracteres especiales
email_text = re.sub(r"[^\w\s]", "", email_text)

# Tokenizar el texto
tokens = email_text.split()

# Codificar el correo electrónico
encoded_email = model.encode(tokens)

# Clasificar el correo electrónico
predictions = model.predict(encoded_email)
category = tf.argmax(predictions).numpy()

# Imprimir la categoría
if category == 0:
  print("Categoría: Principal")
elif category == 1:
  print("Categoría: Promocional")
else:
  print("Categoría: Spam")
