from flask import Blueprint, jsonify, render_template
from .api_reniec import ApisNetPe

# Eliminamos la definición del Blueprint aquí ya que está en __init__.py
from . import api_bp

API_TOKEN = "apis-token-15465.DvgXn4t4WlIwmrMADgodv1NhCjpNgGsZ"
api = ApisNetPe(token=API_TOKEN)

@api_bp.route('/')
def api_index():
    return render_template('api/index.html')

@api_bp.route('/consultar_dni/<dni>')
def consultar_dni(dni):
    persona = api.get_person(dni)
    return jsonify(persona)