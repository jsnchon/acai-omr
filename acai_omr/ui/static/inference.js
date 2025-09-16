// TODOs: 
// look into hosting on digitalocean
// change streaming to show progress for each inference (make sure to include a lil img i out of n progress tracker somewhere)
// test that can capture images with camera on mobile. Test with single staff systems, multiple systems, phone camera, etc

import { annotateImage, getBboxes } from "./annotate_img.js";

function showSection(elem) {
    elem.classList.add("active");
}

function hideSection(elem) {
    elem.classList.remove("active");
}

const startView = document.getElementById("start-view");
const firstSection = startView; // first section to show. Others are kept hidden until they should be activated

document.addEventListener("DOMContentLoaded", () => {
    setTimeout(() => { // brief pause so animations play nicely
        showSection(firstSection);
    }, 500);
});

const imageForm = document.getElementById("image-form");
const imageUpload = document.getElementById("image-upload");
const imagePreviewWindow = document.getElementById("image-preview-window");
const imagePreview = document.getElementById("image-preview");

const rootTempDir = await fetch("/tmpdir/create", { method: "POST" }).then((resp) => resp.json()).then((resp) => resp.path);
console.log(`Using temporary directory ${rootTempDir} for this run`)

imageUpload.addEventListener("change", (e) => {
    const imgFile = e.target.files[0];
    if (imgFile) {
        const reader = new FileReader();
        reader.onload = (ev) => {
            imagePreviewWindow.classList.add("active")
            imagePreview.src = ev.target.result;
            imagePreview.style.display = "block";
        };
        reader.readAsDataURL(imgFile);
    }
});

let imgPath = null;
let stage = null;
let layer = null;
const annotateForm = document.getElementById("annotate-form");

imageForm.addEventListener("submit", 
    async (e) => {
        e.preventDefault();
        hideSection(startView);
        
        const imgFile = imageUpload.files[0]
        const formData = new FormData(); // FormData deals with formatting request data
        formData.append("img_file", imgFile);
        formData.append("root_temp_dir", rootTempDir);

        const resp = await fetch("/upload", {
            method: "POST",
            body: formData
        }).then((resp) => resp.json());

        console.log("Server response to image upload: ", resp);
        imgPath = resp.path;

        showSection(annotateForm);
        [stage, layer] = annotateImage(imagePreview.src);
    }
);

let bboxes = null;
let totalImgs = null;
let imgDir = null;

annotateForm.addEventListener("submit", 
    async(e) => {
        e.preventDefault();
        bboxes = getBboxes(stage, layer);
        if (bboxes.length === 0) {
            alert("At least one box should be drawn");
            return;
        }
        hideSection(annotateForm);
        console.log("Submitting bounding boxes: ", bboxes);
        const resp = await fetch("/inference/setup", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ "path": imgPath, "bboxes": bboxes, "root_temp_dir": rootTempDir })
        }).then((resp) => resp.json());
        console.log("Server response to box submission: ", resp);
        totalImgs = bboxes.length;
        imgDir = resp.path;

        showSection(settingsForm);
    }
)

const settingsForm = document.getElementById("settings-form");
const encodingProgressView = document.getElementById("encoding-progress-view");
const tokenStreamView = document.getElementById("token-stream-view");
const outputElem = document.getElementById("output-window");
const overallProgressView = document.getElementById("overall-progress-view");
const overallProgressElem = document.getElementById("overall-progress-count");

settingsForm.addEventListener("submit",
    async (e) => {
        e.preventDefault();
        hideSection(settingsForm); // hide inference setup

        // load inference events stored by config.py
        const resp = await fetch("/static/inference_events.json");
        const inferenceEvents = await resp.json();

        const formData = new FormData(settingsForm);
        formData.append("path", imgDir)
        const params = new URLSearchParams(formData);

        console.log("Starting inference stream");
        const source = new EventSource(`/inference/stream?${params.toString()}`);

        updateOverallProgress(overallProgressElem, 0, totalImgs);
        showSection(overallProgressView);
        streamInference(source, inferenceEvents);
        handleStreamEnd(source, inferenceEvents);
   }
);

function updateOverallProgress(overallProgressElem, doneCount, totalCount) {
    overallProgressElem.textContent = `${doneCount}/${totalCount}`;
}

let seqs = []; // list of [sequence, average confidence] lists

function streamInference(source, inferenceEvents) {
    source.addEventListener("message", (e) => {
        const eventObj = JSON.parse(e.data);

        switch (eventObj.type) {
            case inferenceEvents.ENCODING_START:
                showSection(encodingProgressView);
                break;

            case inferenceEvents.ENCODING_FINISH:
                hideSection(encodingProgressView);
                showSection(tokenStreamView);
                break;
            
            case inferenceEvents.STEP:
                displayTokenStream(outputElem, eventObj);
                break;

            case inferenceEvents.INFERENCE_FINISH:
                // update progress counter, reset for the start of the next inference
                seqs.push([eventObj.payload.sequence, eventObj.payload.averageConfidence]);
                updateOverallProgress(overallProgressElem, seqs.length, totalImgs);

                outputElem.replaceChildren([]);
                hideSection(tokenStreamView);
                break;
       }
    });
}

function displayTokenStream(outputElem, inferenceStepEvent) {
    const tokens = inferenceStepEvent.payload.tokens.split(" ");
    tokens.forEach((token, i) => {
        const textDisplay = document.createElement("span");
        textDisplay.textContent = token + " ";
        textDisplay.classList.add("slide-in");
        textDisplay.style.opacity = 0;
        textDisplay.style.animationDelay = `${i * 0.05}s`
        outputElem.appendChild(textDisplay);
    });
}

const resultView = document.getElementById("result-view")

// listener that deals with the rest of the flow after inference finishes
function handleStreamEnd(source, inferenceEvents) {
    source.addEventListener("message", (e) => {
        const eventObj = JSON.parse(e.data);
        if (eventObj.type === inferenceEvents.ALL_INFERENCE_FINISH) {
            console.log("Inference stream finished; closing stream source");
            source.close();

            seqs.forEach((seq) => {
                console.log(seq);

            });
            
            // do more stuff: 
            // 1) send request to endpoint that converts to lmx, musicxml, and reconstructed img files
            // 2) display all the results in a window, send musicxml file to user

        }
    });
}
