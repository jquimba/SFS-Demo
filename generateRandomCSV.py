import csv
import random

# Generación aleatoria de correos electrónicos y etiquetas
emails = []
labels = []

for _ in range(500):
    # Generar ejemplos de spam
    spam_email = "Subject: Special Offer!\n\n" + " ".join(["Lorem ipsum" for _ in range(random.randint(5, 20))]) + "\n\nClaim your prize now at: [Claim Now](http://spam-link.example.com)"
    emails.append(spam_email)
    labels.append(0)

for _ in range(500):
    # Generar ejemplos de no spam
    non_spam_email = "Subject: Project Update\n\n" + " ".join(["Lorem ipsum" for _ in range(random.randint(5, 20))]) + "\n\nPlease find attached the latest project update."
    emails.append(non_spam_email)
    labels.append(1)

# Combinar los correos electrónicos y etiquetas en tuplas
email_label_pairs = list(zip(emails, labels))

# Mezclar aleatoriamente los ejemplos
random.shuffle(email_label_pairs)

# Tomar los primeros 1000 ejemplos
email_label_pairs = email_label_pairs[:1000]

# Guardar los datos en un archivo CSV
with open('./datasets/emails.csv', 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['email', 'label'])  # Escribir el encabezado
    writer.writerows(email_label_pairs)  # Escribir los ejemplos de correos electrónicos y etiquetas
