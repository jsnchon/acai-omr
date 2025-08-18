from flask import Flask

def create_app():
    app = Flask(
        __name__,
        template_folder="ui/templates",
        static_folder="ui/static"
    )

    from acai_omr.ui.routes import main
    app.register_blueprint(main)

    return app