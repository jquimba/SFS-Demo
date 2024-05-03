import os

port = int(os.environ.get('PORT', 5000))
print("El puerto de escucha es:", port)