from os import environ

from flask import Flask
from flask_cors import CORS
from flask_migrate import Migrate
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()
migrate = Migrate()


def create_app():
    app = Flask(__name__)

    # Enable CORS
    CORS(app)

    # Configure the database
    app.config["SQLALCHEMY_DATABASE_URI"] = environ.get(
        "DATABASE_URL", "postgresql://postgres:postgres@db:5432/postgres"
    )

    # Import models to ensure they are registered with SQLAlchemy
    # Import and register Blueprints (routes)
    from routes import main_bp

    db.init_app(app)
    migrate.init_app(app, db)

    app.register_blueprint(main_bp)

    # Apply the database schema
    with app.app_context():
        db.create_all()  # This will create the tables if they do not exist

    return app
