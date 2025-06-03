from flask import Flask, render_template
from facial import facial_bp
from api import api_bp  # Importamos desde el módulo api

app = Flask(__name__)

# Registrar los blueprints
app.register_blueprint(api_bp, url_prefix='/api')
app.register_blueprint(facial_bp, url_prefix='/facial')

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)