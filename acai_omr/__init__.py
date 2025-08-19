from flask import Flask
import logging
import sys

def create_app():
    # configure root logger in this Flask entrypoint. Child loggers will be affected by the config since they'll propogate up
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    stream_handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        "%(module)s -- %(levelname)s: %(message)s"
    )
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    
    app = Flask(
        __name__,
        template_folder="ui/templates",
        static_folder="ui/static"
    )

    from acai_omr.ui.routes import main
    app.register_blueprint(main)

    return app