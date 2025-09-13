const BACKGROUND_MAX_WIDTH = 700;
const BACKGROUND_MAX_HEIGHT = 1500;

// don't create a box if user just clicks -- drags have to be beyond a threshold
const BOX_MIN_WIDTH = 0.01;
const BOX_MIN_HEIGHT = 0.01;

// TODO: manage a dict of boxes ({id: box}), return that. Add button to delete box when it's active
function annotateImage(imgSrc) {
    const stage = new Konva.Stage({
        container: "annotate-container",
    });

    const layer = new Konva.Layer();
    stage.add(layer);

    const background = new Konva.Image();
    layer.add(background);

    const imgObj = new Image();
    imgObj.src = imgSrc;

    const scale = Math.min(BACKGROUND_MAX_WIDTH / imgObj.width, BACKGROUND_MAX_HEIGHT / imgObj.height, 1.0);
    const displayWidth = imgObj.width * scale;
    const displayHeight = imgObj.height * scale;

    imgObj.onload = () => {
        background.image(imgObj);
        background.width(displayWidth);    
        background.height(displayHeight);  
        stage.width(displayWidth);
        stage.height(displayHeight);
        layer.draw();
    };

    const transform = new Konva.Transformer();
    layer.add(transform);

    let startX = null;
    let startY = null;
    let currRect = null;

    // drag a new rectangle out when a click starts on the background, drags beyond the size thresholds, and releases
    stage.on("mousedown", (e) => {
        console.log("mousedown")
        if (e.target === background) {
            transform.nodes([]);

            startX = stage.getPointerPosition().x;
            startY = stage.getPointerPosition().y;

            currRect = new Konva.Rect({
                x: startX,
                y: startY,
                width: 0,
                height: 0,
                stroke: "orange",
                strokeWidth: 2,
                draggable: true,
            });
            layer.add(currRect);
        } 
        // transformer handles are also considered instances of their parent shapes, so need to avoid repeatedly adjusting
        // transform.nodes if manipulating them in a mousedown event (eg dragging them around)
        else if (e.target instanceof Konva.Rect && e.target.getParent() !== transform) {
            transform.nodes([e.target]);
        }
    });

    stage.on("mousemove", () => {
        if (!currRect) return;
        const pos = stage.getPointerPosition();
        currRect.width(pos.x - startX);
        currRect.height(pos.y - startY);
        layer.batchDraw();
    });

    stage.on("mouseup", () => {
        console.log("mouseup")
        if (!currRect) return;

        // normalize coordinates relative to image
        const rect = currRect.getClientRect();
        const box = {
            x0: rect.x / stage.width(),
            y0: rect.y / stage.height(),
            x1: (rect.x + rect.width) / stage.width(),
            y1: (rect.y + rect.height) / stage.height(),
        };

        if (box.x1 - box.x0 < BOX_MIN_WIDTH || box.y1 - box.y0 < BOX_MIN_HEIGHT) {
            console.log("Not counting as a box");
            currRect.width(0);
            currRect.height(0);
        }
        else {
            console.log("Box:", box);
        }

        currRect = null; 
    });

    // when clicking a rectangle, attach the transformer to allow for rectangle resizing
    stage.on("click", (e) => {  
        if (e.target instanceof Konva.Rect) {
        } else {
            transform.nodes([]);
        }
        layer.draw();
    });
}

export default annotateImage;
