import os
from flask import Flask

app = Flask(__name__)
@app.route('/')
def hello():
        return '¡Hola, mundo! Este es un servicio web en Python desplegado en Heroku.'
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print("El puerto de escucha es:", port)