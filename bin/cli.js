#!/usr/bin/env node

'use strict';

const program = require('commander');
const util = require('util');
const path = require('path');
const fs = require('fs');
const jimp = require('jimp');
const yolo = require(path.join(__dirname, '..'));
const drawBoxes = require('./drawBoxes');

const onnxPath = path.join(__dirname, '../test/data/YOLOv2_2018_09_27.onnx')

function resolve(ipath) {
    return path.resolve(process.cwd(), ipath);
}

function readImageFile(fileName) {
    return new Promise((resolve, reject) => {
        fs.readFile(program.input, (err, buf) => {
            if (err) {
                return reject(err);
            }

            resolve(buf);
        });
    });
}
 
program
    .usage('[options] <url>')
    .version(yolo.version, '-v, --version')
    .option('-i, --input <pathname>', 'Input file path.)', resolve)
    .option('-o, --output <pathname>', 'Output file path.)', resolve)
    .option('-s, --score <number>', 'Score threshold (0, 1.0] (defaults to 0.7)', parseFloat)
    .option('-x, --overlap <number>', 'Overlap threshold (0, 1.0] (defaults to 0.7)', parseFloat)
    .option('-b, --browse', 'Open output image with default viewer.')
    .parse(process.argv);


program.output = program.output || resolve('./out.jpg');

const model = yolo.create();

readImageFile(program.input)
.then((buf) => {
    return jimp.read(buf)
    .then((image) => {
        model.build(onnxPath)
        .then(() => {
            const iBuf = model.inputBuffer;
            const originalShape = [image.bitmap.height, image.bitmap.width];

            // Resize it to (model.inSize x model.inSize).
            image.resize(model.inSize, model.inSize);

            // Now, copy the image data into to the input buffer in NCHW format.
            image.scan(0, 0, image.bitmap.width, image.bitmap.height, (x, y, idx) => {
                for (let c = 0; c < 3; ++c) {
                    let val = image.bitmap.data[idx + c];
                    val = val / 255.0;
                    iBuf.set(c, y, x, val);
                }
            });

            const runOpts = {};
            if (program.score) {
                runOpts.scoreThresh = program.score;
            }
            if (program.overlap) {
                runOpts.overlapThresh = program.overlap;
            }

            return model.run(originalShape, runOpts)
        })
        .then((boxes) => {
            return drawBoxes(program.output, boxes, buf);
        })
        .then(() => {
            if (program.browse) {
                require('opn')(program.output, { wait: false });
            }
        });
    })
})
.catch((err) => {
    console.error(err);
});

