from flask import Blueprint, render_template, request, jsonify
# from acai_omr.inference.vitomr_inference import beam_search

# Create a Blueprint object
main = Blueprint("main", __name__)

@main.route("/")
def index():
    return render_template("index.html")