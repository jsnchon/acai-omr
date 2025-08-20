function showSection(elem) {
    elem.classList.add("active");
}

function hideSection(elem) {
    elem.classList.remove("active");
}

const inferenceForm = document.getElementById("inference-form");
const firstSection = inferenceForm; // first section to show. Others are kept hidden until they should be activated

document.addEventListener("DOMContentLoaded", () => {
    setTimeout(() => {
        showSection(firstSection);
    }, 500);
});

const encodingProgressView = document.getElementById("encoding-progress-view");
const tokenStreamView = document.getElementById("token-stream-view");
const outputElem = document.getElementById("stream-output");

inferenceForm.addEventListener("submit",
    async (e) => {
        e.preventDefault();
        hideSection(inferenceForm); // hide inference setup

        // load inference events stored by config.py
        const resp = await fetch("/static/inference_events.json");
        const inferenceEvents = await resp.json();

        const formData = new FormData(inferenceForm);
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
    const beams = inferenceStepEvent.payload.beams;
    let beamDisplays = [];
    for (const beam_idx in beams) {
        const beamDisplay = document.createElement("span");
        beamDisplay.textContent = beams[beam_idx];
        beamDisplay.classList.add("slide-in");
        beamDisplays.push(beamDisplay);
    }
    outputElem.replaceChildren(...beamDisplays);
    beamDisplays = [];
}

// listener that deals with the rest of the flow after inference finishes
function handleStreamEnd(source, inferenceEvents) {
    source.addEventListener("message", (e) => {
        const eventObj = JSON.parse(e.data);
        if (eventObj.type === inferenceEvents.INFERENCE_FINISH) {
            console.log("Inference stream finished; closing stream source");
            source.close();

            
            // do more stuff
        }
    });
}
