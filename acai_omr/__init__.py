from flask import Flask
import logging
import sys
from enum import Enum
import json
import pathlib

# inference streaming events. Python back-end can import this Enum while javascript in the front-end can
# use the saved .json for more type safety
class InferenceEvent(Enum):
    ENCODING_START = "encoding_start"
    ENCODING_FINISH = "encoding_finish"
    STEP = "step" # each inference step yields both beam(s) and log prob(s)
    INFERENCE_FINISH = "inference_finish" # include the result from the last step which we treat differently (eg only has one sequence, want to stream its score to the ui)

INFERENCE_EVENTS_JSON_PATH = pathlib.Path("acai_omr/ui/static/inference_events.json")

def create_app():
    inference_events = {e.name: e.value for e in InferenceEvent}
    INFERENCE_EVENTS_JSON_PATH.write_text(json.dumps(inference_events, indent=2))

    # configure root logger in this Flask entrypoint. Child loggers will be affected by the config since they'll propogate up
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    stream_handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        "%(module)s - %(levelname)s: %(message)s"
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
