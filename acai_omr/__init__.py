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
    STEP = "step"
    # INFERENCE_FINISH is sent after a single image is done. ALL_INFERENCE_FINISH is sent when all images that need
    # to be inferred on are done (so one or more INFERENCE_FINISH events have already been sent)
    INFERENCE_FINISH = "inference_finish"
    ALL_INFERENCE_FINISH = "all_inference_finish"

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
