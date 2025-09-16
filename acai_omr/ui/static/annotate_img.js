const BACKGROUND_MAX_WIDTH = 700;
const BACKGROUND_MAX_HEIGHT = 1500;

// don't create a box if user just clicks -- drags have to be beyond a threshold
const BOX_MIN_WIDTH = 0.01;
const BOX_MIN_HEIGHT = 0.01;

const BBOX_NAME = "bbox"; // id string for created Rects so they can be filtered from layer

function setUpStage(imgSrc) {
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

    return [stage, layer, background];
}

function createDeleteButton(initFill) {
    const deleteButton = new Konva.Group({ visible: false });
    deleteButton.add(new Konva.Circle({
        radius: 10,
        fill: initFill,
        stroke: "black",
        strokeWidth: 1,
    }));
    const buttonText = new Konva.Text({
        text: "x",
        fontSize: 16,
        fill: "black",
    });
    buttonText.offsetX(buttonText.width() / 2);
    buttonText.offsetY(buttonText.height() / 2);
    deleteButton.add(buttonText);

    return deleteButton;
}

function setUpDeleteButtonListeners(deleteButton, hoverFill, transformer, layer) {
    const hoverTween = new Konva.Tween({
        node: deleteButton.findOne("Circle"), // animate the actual button part of the Group
        duration: 0.5,
        fill: hoverFill,
        easing: Konva.Easings.EaseInOut,
    });

    deleteButton.on("mouseenter", () => {
       hoverTween.play();
    });
    deleteButton.on("mouseleave", () => {
        hoverTween.reverse();
    });
    deleteButton.on("click", () => {
        const focusedRect = transformer.nodes()[0];
        focusedRect.destroy();
        transformer.nodes([]);
        deleteButton.hide();
        layer.draw();
    });
}

// callback to keep delete button position synced with the bounding boxes
function syncDeleteButton(deleteButton, transformer, layer, padding) {
    const focusedRect = transformer.nodes()[0];
    const rectBox = focusedRect.getClientRect({ relativeTo: layer });
    deleteButton.position({
        x: rectBox.x + (rectBox.width - padding),
        y: rectBox.y 
    });
    layer.batchDraw();
}

// takes a Konva.Rect instance and returns a bbox object with x and y coordinates
function convertToBbox(rect, stage) {
    const rectBox = rect.getClientRect();
    // we scaled stage so that image fits within max dimensions, so normalize coordinates to later match to the raw image coordinates
    const bbox = {
        x0: rectBox.x / stage.width(), 
        y0: rectBox.y / stage.height(),
        x1: (rectBox.x + rectBox.width) / stage.width(),
        y1: (rectBox.y + rectBox.height) / stage.height(),
    };
    return bbox;
}

// TODO: on form submit, call another function that takes stage and returns all remaining boxes
export function annotateImage(imgSrc) {
    const [stage, layer, background] = setUpStage(imgSrc);

    const initFill = "#d61d89ff"
    const deleteButton = createDeleteButton(initFill);
    layer.add(deleteButton)
    const deleteButtonPadding = 25; // prevent button from overlapping corner transform anchor

    const transformer = new Konva.Transformer({ anchorSize: 12, rotateEnabled: false });
    transformer.on("transform", () => {
        syncDeleteButton(deleteButton, transformer, layer, deleteButtonPadding);
    });
    layer.add(transformer);

    const hoverFill = "#fe1f39ff";
    setUpDeleteButtonListeners(deleteButton, hoverFill, transformer, layer);

    let startX = null;
    let startY = null;
    let currRect = null;

    // drag a new rectangle out when a click starts on the background, drags beyond the size thresholds, and releases
    stage.on("mousedown", (e) => {
        if (e.target === background) {
            // reset any Rects that were being transformed
            transformer.nodes([]); 
            deleteButton.hide();
            layer.draw();

            startX = stage.getPointerPosition().x;
            startY = stage.getPointerPosition().y;

            currRect = new Konva.Rect({
                x: startX,
                y: startY,
                width: 0,
                height: 0,
                stroke: "#de0bcf",
                strokeWidth: 2,
                draggable: true,
                name: BBOX_NAME,
            });
            layer.add(currRect);
        } 
        // attach the transformer to an already existing rectangle that's clicked. The parent check makes sure we don't 
        // reattach when transformer anchors are clicked/dragged (since they're also instances of Konva.Rect)
        else if (e.target instanceof Konva.Rect && e.target.getParent() !== transformer) {
            transformer.nodes([e.target]);
            e.target.on("dragmove", () => {
                syncDeleteButton(deleteButton, transformer, layer, deleteButtonPadding);
            });

            const rectBox = e.target.getClientRect({ relativeTo: layer });
            deleteButton.position({
                x: rectBox.x + (rectBox.width - deleteButtonPadding),
                y: rectBox.y 
            });
            deleteButton.show();
            deleteButton.moveToTop();
            layer.draw();
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
        if (!currRect) return;

        const bbox = convertToBbox(currRect, stage);
        // ignore clicks/drags that create too small of boxes
        if (bbox.x1 - bbox.x0 < BOX_MIN_WIDTH || bbox.y1 - bbox.y0 < BOX_MIN_HEIGHT) {
            currRect.destroy();
        }

        currRect = null; 
    });

    return [stage, layer];
}

export function getBboxes(stage, layer) {
    const rects = layer.find(`.${BBOX_NAME}`);
    const bboxes = [];
    rects.forEach((rect) => {
        const bbox = convertToBbox(rect, stage);
        bboxes.push(bbox);
    });
    return bboxes;
}
