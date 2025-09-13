import annotateImage from "./annotate_img.js";

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
const annotateForm = document.getElementById("annotate-form");

imageForm.addEventListener("submit", 
    async (e) => {
        e.preventDefault();
        hideSection(startView);
        
        const imgFile = imageUpload.files[0]
        const formData = new FormData();
        formData.append("img_file", imgFile);

        let resp = await fetch("/upload", {
            method: "POST",
            body: formData
        });

        resp = await resp.json();
        console.log("Server response to image upload: ", resp);
        imgPath = resp.path;

        showSection(annotateForm);
        annotateImage(imagePreview.src);
    }
);

// TODO: annotateForm.addEventListener(), on submit move onto settings-form

const settingsForm = document.getElementById("settings-form");
const encodingProgressView = document.getElementById("encoding-progress-view");
const tokenStreamView = document.getElementById("token-stream-view");
const outputElem = document.getElementById("output-window");

settingsForm.addEventListener("submit",
    async (e) => {
        e.preventDefault();
        hideSection(settingsForm); // hide inference setup

        // load inference events stored by config.py
        const resp = await fetch("/static/inference_events.json");
        const inferenceEvents = await resp.json();

        const formData = new FormData(settingsForm);
        const params = new URLSearchParams(formData);
        
        console.log("Starting inference stream");
        const source = new EventSource(`/inference/stream?${params.toString()}`);

        streamInference(source, inferenceEvents);
        handleStreamEnd(source, inferenceEvents);
   }
);

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
       }
    });
}

function displayTokenStream(outputElem, inferenceStepEvent) {
    const tokens = inferenceStepEvent.payload.tokens.split(" ");
    tokens.forEach((token, i) => {
        let textDisplay = document.createElement("span");
        textDisplay.textContent = token + " ";
        textDisplay.classList.add("slide-in");
        textDisplay.style.opacity = 0;
        textDisplay.style.animationDelay = `${i * 0.05}s`
        outputElem.appendChild(textDisplay);
    });
}

const resultView = document.getElementById("result-view")

// TODO: look into hosting on digitalocean
// test that can capture images with camera on mobile

// listener that deals with the rest of the flow after inference finishes
function handleStreamEnd(source, inferenceEvents) {
    source.addEventListener("message", (e) => {
        const eventObj = JSON.parse(e.data);
        if (eventObj.type === inferenceEvents.INFERENCE_FINISH) {
            console.log("Inference stream finished; closing stream source");
            source.close();

            
            // do more stuff: 
            // 1) send request to endpoint that converts to lmx, musicxml, and reconstructed img files
            // 2) display all the results in a window, send musicxml file to user

        }
    });
}
