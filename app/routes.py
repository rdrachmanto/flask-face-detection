from flask import Blueprint, Response, render_template

from app.nn import face_detection


bp = Blueprint("main", __name__)


@bp.route("/")
def index():
    return render_template('index.html')


@bp.route("/detect")
def detect():
    return Response(
        face_detection.pipeline(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )
