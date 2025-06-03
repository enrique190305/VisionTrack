from flask import Blueprint

facial_bp = Blueprint('facial', __name__)

from . import routes