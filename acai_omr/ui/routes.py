import torch
from flask import Blueprint, render_template, request, Response, send_file
from acai_omr.inference.vitomr_inference import streamed_inference
from acai_omr.config import INFERENCE_VITOMR_PATH
from acai_omr.train.omr_teacher_force_train import set_up_omr_teacher_force_train
from acai_omr.__init__ import InferenceEvent
from acai_omr.utils.utils import stringify_lmx_seq
from olimpic_app.linearization.Delinearizer import direct_delinearize
import logging  
import tempfile
from PIL import Image
from pathlib import Path
import json
import re
import subprocess
import base64
import shutil

main = Blueprint("main", __name__)
logger = logging.getLogger(__name__)

MAX_BATCH_SIZE = 1
CACHE_DTYPE = torch.bfloat16

vitomr, base_img_transform, _, device = set_up_omr_teacher_force_train()
logger.info(f"Enabling caching for decoder with a max batch size of {MAX_BATCH_SIZE} and cache datatype of {CACHE_DTYPE}")
vitomr.decoder = vitomr.decoder.to_cached_version(MAX_BATCH_SIZE, CACHE_DTYPE)

logger.info(f"Loading state dict from {INFERENCE_VITOMR_PATH}")
if device == "cpu":
    vitomr_state_dict = torch.load(INFERENCE_VITOMR_PATH, map_location=torch.device("cpu"))
else:
    vitomr_state_dict = torch.load(INFERENCE_VITOMR_PATH)

vitomr.load_state_dict(vitomr_state_dict)

if device == "cpu":
    flush_interval = 25
else:
    flush_interval = 50 # buffer GPUs more

@main.route("/")
def index():
    return render_template("index.html", weights_path=INFERENCE_VITOMR_PATH)

@main.route("/tmpdir/create", methods=["POST"]) # create temp dir for this whole inference run (includes temp files and other temp dirs)
def create_root_temp_dir():
    root_temp_dir = tempfile.TemporaryDirectory(delete=False)
    return {"path": root_temp_dir.name}

@main.route("/upload", methods=["POST"])
def upload_img():
    f = request.files["img_file"]
    root_temp_dir = request.form["root_temp_dir"]
    disk_f = tempfile.NamedTemporaryFile(dir=root_temp_dir, delete=False)
    f.save(disk_f)
    disk_f.close()
    file_path = str(Path(root_temp_dir) / disk_f.name)
    logger.debug(f"User uploaded image saved to {file_path}")
    return {"path": file_path}

# SSE wrapper that post-processes and then yields events yielded by inference.
# Again, we assume there's only one sequence being passed here at a time
def stream_inference_wrapper(img, vitomr, device, max_inference_len, flush_interval):
    for event in streamed_inference(img, vitomr, device, max_inference_len, flush_interval):
        logger.debug(f"Inference event: {event}")
        if event["type"] == InferenceEvent.STEP.value:
            tokens = event["payload"]["tokens"]
            tokens = tokens[tokens != vitomr.decoder.pad_idx] # last buffer may have leftover pad tokens
            tokens = stringify_lmx_seq(tokens.squeeze(0), vitomr.decoder.idxs_to_tokens)
            event["payload"] = {"tokens": tokens}

        if event["type"] == InferenceEvent.INFERENCE_FINISH.value:
            sequence = event["payload"]["sequence"]
            seq_mask = event["payload"]["mask"]
            sequence = stringify_lmx_seq(sequence.squeeze(0), vitomr.decoder.idxs_to_tokens)
            logger.info("Inference finished")
            seq_log_probs = event["payload"]["log_probs"]
            avg_log_prob = (seq_log_probs.sum() / seq_mask.sum()).item()
            event["payload"] = {"sequence": sequence, "avgLogProb": avg_log_prob}
        
        yield event

"""
Wrapper for multiple images. Takes a string of the path to the directory containing the images to run inference on. This
assumes this directory was created by /inference/setup, meaning images have been labelled according to their order in 
the original whole page. Different events will be yielded for when one image's inference as finished and when all images are finished
"""
def multiple_img_stream_inference_wrapper(img_dir, vitomr, device, max_inference_len, flush_interval):
    img_dir = Path(img_dir)
    # images should be labelled numerically in increasing order. Search for then sort based off of the int in their file name
    for img in sorted(img_dir.iterdir(), key=lambda x: int(re.search(r"\d+", x.name).group(0))):
        logger.debug(f"Running inference on image {img}")
        img = Image.open(img).convert("L")
        img = base_img_transform(img).to(device)
        
        for event in stream_inference_wrapper(img, vitomr, device, max_inference_len, flush_interval):
            yield f"data: {json.dumps(event)}\n\n"

    yield f"data: {json.dumps({"type": InferenceEvent.ALL_INFERENCE_FINISH.value, "payload": None})}\n\n"

# given the path to the original unsplit image and the submitted bounding boxes, crop the image at each bounding box and 
# return a path to a directory containing the split images
@main.route("/inference/setup", methods=["POST"])
def setup_inference():
    data = request.json
    img_path = data["path"]
    bboxes = data["bboxes"]
    root_temp_dir = Path(data["root_temp_dir"])
    logger.debug(f"Received bboxes {bboxes} for img at {img_path}")
    unsplit_img = Image.open(img_path).convert("L")
    tmpdir = tempfile.TemporaryDirectory(dir=root_temp_dir, delete=False)
    splits_tmpdir_path = Path(tmpdir.name)
    # sort from top-most to bottom-most bounding boxes
    bboxes = sorted(bboxes, key=lambda x: x["y0"])

    logger.debug(f"Splitting images and saving to {str(splits_tmpdir_path)}")
    for i, bbox in enumerate(bboxes):
        split_img = unsplit_img.crop((bbox["x0"] * unsplit_img.width, bbox["y0"] * unsplit_img.height, bbox["x1"] * unsplit_img.width, bbox["y1"] * unsplit_img.height))
        split_img.save(splits_tmpdir_path / f"system_{i}.png")

    return {"path": str(splits_tmpdir_path)}

@main.route("/inference/stream")
def stream_inference():
    max_inference_len = int(request.args.get("max_inference_len", 1536))
    img_dir = request.args.get("path")

    logger.info(f"Starting inference with max length {max_inference_len} and streaming from endpoint with a flush interval of {flush_interval}")
    return Response(multiple_img_stream_inference_wrapper(img_dir, vitomr, device, max_inference_len, flush_interval), mimetype="text/event-stream")

"""
Given a path to a musicxml file, converts that to a list of base64 encoded image(s). By default, the musescore
CLI renders images with a transparent background, so we convert its output to a png with a white background using
imagemagick
"""
def musicxml_to_imgs(xml_file_path: Path, root_temp_dir: Path):
    result = []

    logger.info(f"Converting {xml_file_path} to base64 image(s)")
    musescore_out_stem = "musecore_out.png"
    with tempfile.TemporaryDirectory(dir=root_temp_dir) as imgs_temp_dir_name:
        logger.debug(f"Created {imgs_temp_dir_name} temporary directory for musescore CLI outputs")
        subprocess.run(["musescore3", "-o", Path(imgs_temp_dir_name) / musescore_out_stem, xml_file_path])
        musescore_outputs = list(Path(imgs_temp_dir_name).iterdir())
        # in the event musescore outputs multiple files, the stem will have numerical suffixes appended to it corresponding to page numbers
        if len(musescore_outputs) != 1:
            musescore_outputs = sorted(musescore_outputs, key=lambda x: int(re.search(r"\d+", x.name).group(0)))

        for i, musescore_out in enumerate(musescore_outputs):
            logger.debug(f"Converting file {musescore_out} to final png")
            final_img_name = Path(imgs_temp_dir_name) / f"page_{i}.png"
            subprocess.run(["convert", musescore_out, "-background", "white", "-alpha", "remove", "-alpha", "off", final_img_name])

            with open(final_img_name, "rb") as f:
                final_img = f.read()

            final_img = base64.b64encode(final_img).decode("utf-8")
            result.append(final_img)
    logger.info("Final image(s) encoded into base64")
    return result

# given a list of lmx sequences, do the following: concatenate the sequences, delinearize, convert back to image(s), encode 
# everything into base64 strings, respond with that
@main.route("/inference/postprocess", methods=["POST"])
def prepare_results():
    data = request.json
    seqs = data["sequences"]
    avg_log_probs = data["avg_log_probs"]
    root_temp_dir = Path(data["root_temp_dir"])

    final_seq = " ".join(seqs)

    musicxml = direct_delinearize(final_seq)
    musicxml_tempfile = tempfile.NamedTemporaryFile(mode="tw", dir=root_temp_dir, delete=False, suffix=".musicxml")
    musicxml_tempfile.write(musicxml)
    logger.debug(f"Wrote musicxml to {musicxml_tempfile.name}")
    musicxml_tempfile.close()
    musicxml_path = root_temp_dir / musicxml_tempfile.name

    final_imgs = musicxml_to_imgs(musicxml_path, root_temp_dir)

    avg_confidence = torch.exp(torch.tensor(sum(avg_log_probs) / len(avg_log_probs))).item()

    return {"finalLmxSeq": final_seq, "avgConfidence": avg_confidence, "musicxmlPath": musicxml_tempfile.name, "finalImgs": final_imgs}

@main.route("/download", methods=["POST"])
def download_file():
    file_path = request.json["path"]
    logger.info(f"Sending file located at {file_path}")
    return send_file(file_path, as_attachment=True, download_name="result.musicxml")

@main.route("/clear", methods=["PUT"])
def clear_tempdir():
    dir_path = request.json["path"]
    logger.info(f"Clearing directory located at {dir_path}")
    shutil.rmtree(dir_path)
    return {"status": "ok"}
