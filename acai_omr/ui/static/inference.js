function show_section(elem) {
    elem.classList.add("active");
}

function hide_section(elem) {
    elem.classList.remove("active");
}

const inferenceForm = document.getElementById("inference-form");

document.addEventListener("DOMContentLoaded", () => {
    setTimeout(() => {
        show_section(inferenceForm);
    }, 500);
});

const setupProgressView = document.getElementById("setup-progress-view");
const encodingProgressView = document.getElementById("encoding-progress-view");
const tokenStreamView = document.getElementById("token-stream-view");
const outputElem = document.getElementById("stream-output");

inferenceForm.addEventListener("submit",
    async (e) => {
        e.preventDefault();
        hide_section(inferenceForm); // hide inference setup

        // load inference events stored by config.py
        const resp = await fetch("/static/inference_events.json");
        const inference_events = await resp.json();

        const formData = new FormData(inferenceForm);
        const params = new URLSearchParams(formData);
        
        console.log("Starting inference stream");
        const source = new EventSource(`/inference/stream?${params.toString()}`);

        show_section(setupProgressView);

        source.onmessage = (e) => {
            const event_obj = JSON.parse(e.data);

            switch (event_obj.type) {
                case inference_events.ENCODING_START:
                    hide_section(setupProgressView);
                    show_section(encodingProgressView);
                    break;

                case inference_events.ENCODING_FINISH:
                    hide_section(encodingProgressView);
                    show_section(tokenStreamView);
                    break;
                
                case inference_events.STEP:
                    break;

                case inference_events.INFERENCE_FINISH:
                    console.log("Inference stream finished; closing stream source");
                    source.close();
                    break;
            }
        }
    }
)