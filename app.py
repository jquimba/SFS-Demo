import re
import os
from flask import Flask
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