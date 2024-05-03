from flask import Flask
import os

app = Flask(__name__)

@app.route("/")
def home():
    port = int(os.environ.get('PORT', 5000))
    print("El puerto de escucha es:", port)
    return "Hello, Flask!"