"""
Index
"""
from flask import Blueprint, render_template
bp = Blueprint('bp', __name__,
               template_folder='templates')


@bp.route('/')
def show():
    """
    Blueprint for index page
    :return:
    """
    # read if value in csv fi
    return render_template('index.html')
