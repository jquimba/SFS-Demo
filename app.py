import re
import os
from flask import Flask, request, jsonify
from datetime import datetime

server = Flask(__name__)
@server.route('/')
def hello():
        return '¡Hola, mundo! Este es un servicio web en Python desplegado en Heroku.'
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print("El puerto de escucha es:", port)
    server.run()

@server.route("/hello/<name>")
def hello_there(name):
    now = datetime.now()
    formatted_now = now.strftime("%A, %d %B, %Y at %X")

    # Filter the name argument to letters only using regular expressions. URL arguments
    # can contain arbitrary text, so we restrict to safe characters only.
    match_object = re.match("[a-zA-Z]+", name)

    if match_object:
        clean_name = match_object.group(0)
    else:
        clean_name = "Friend"

    content = "Hello there, " + clean_name + "! It's " + formatted_now
    return content

import classify_email
@server.route('/ClassifyEmail', methods=['POST'])
def recibir_json():
    # Verifica si la solicitud contiene un JSON
    if request.is_json:
        # Obtiene el JSON de la solicitud
        data = request.get_json()
        # Procesa el JSON (puedes agregar tu lógica aquí)
        response = {
            "Body": data["Body"],
            "CaseId": data["CaseId"],
            "Type": classify_email(data["Body"])
        }

        return jsonify(response), 200
    else:
        return jsonify({"mensaje": "La solicitud no contiene un JSON válido"}), 400

if __name__ == '__main__':
    server.run(debug=True)
