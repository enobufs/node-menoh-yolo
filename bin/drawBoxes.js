'use strict';

const Canvas = require('canvas');
const fs = require('fs');
const util = require('util');

const alpha = 0.6;
const colors = [
    `rgba(255,0,0,${alpha})`,
    `rgba(0,255,0,${alpha})`,
    `rgba(0,0,255,${alpha})`,
    `rgba(255,255,0,${alpha})`,
    `rgba(0,255,255,${alpha})`,
    `rgba(255,0,255,${alpha})`,
    `rgba(200,50,100,${alpha})`,
    `rgba(100,200,50,${alpha})`,
    `rgba(50,100,200,${alpha})`,
];

const saveImageOnCanvas = util.promisify((fileName, canvas, cb) => {
    const out = fs.createWriteStream(fileName);
    const stream = canvas.pngStream();

    stream.on('data', function(chunk){
        out.write(chunk);
    });

    stream.on('end', function(){
        cb();
    });
});

function drawBoxes(ctx, boxes) {
    boxes.forEach((box, i) => {
        // draw a label box
        const label = box.label + `: ${box.score.toFixed(6)}`;
        ctx.font="16px Georgia";
        const mt = ctx.measureText(label);

        ctx.strokeStyle = colors[box.classId % colors.length];
        ctx.beginPath();
        ctx.lineWidth = 1;
        ctx.fillStyle = 'rgba(20,0,30,0.5)';
        ctx.rect(box.rec.x, box.rec.y, mt.width + 20, 24);
        ctx.fill()
        ctx.stroke();

        ctx.strokeStyle = 'white';
        ctx.fillStyle = 'white';
        ctx.fillText(   label,
                        box.rec.x + 10,
                        box.rec.y + 16);

        // draw a rectangle
        ctx.strokeStyle = colors[box.classId % colors.length];
        ctx.beginPath();
        ctx.lineWidth = Math.floor(5 * box.score) + 1;
        ctx.rect(   box.rec.x,
                    box.rec.y,
                    box.rec.w,
                    box.rec.h
        );
        ctx.stroke();
    });
}


module.exports = (fileName, boxes, rawImage) => {
    const img = new Canvas.Image;
    img.src = rawImage;
    const canvas = new Canvas(img.width, img.height);
    const ctx = canvas.getContext('2d');
    ctx.drawImage(img, 0, 0, img.width, img.height);

    drawBoxes(ctx, boxes);

    return saveImageOnCanvas(fileName, canvas);
}
