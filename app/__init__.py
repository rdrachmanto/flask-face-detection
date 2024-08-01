from flask import Flask
from app.routes import bp as main


def create_app():
    app = Flask(__name__, instance_relative_config=True)
    app.register_blueprint(main)
    return app
