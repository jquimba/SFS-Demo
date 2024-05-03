from flask import Flask
import os

app = Flask(__name__)

@app.route('/')
def hello():
    return '¡Hola, mundo! Este es un servicio web en Python desplegado en Heroku.'

if __name__ == '__main__':
    app.run(debug=True)
    port = int(os.environ.get("PORT", 80))
    app.run(host='0.0.0.0', port=port)
