#!/usr/bin/env node

'use strict';

const program = require('commander');
const util = require('util');
const path = require('path');
const fs = require('fs');
const jimp = require('jimp');
const yolo = require(path.join(__dirname, '..'));
const drawBoxes = require('./drawBoxes');
const os = require('os');
const download = require('download');

const fileStore = path.join(os.homedir(), '.menoh-yolo');
const onnxPath = path.join(fileStore, 'yolo_v2_voc0712.onnx')
const jsonPath = path.join(fileStore, 'yolo_v2_voc0712.json')
const mkdirp = util.promisify(require('mkdirp'));
const access = util.promisify(fs.access);

const onnxUrl = 'https://github.com/enobufs/node-menoh-yolo/releases/download/v0.0.1/yolo_v2_voc0712.onnx'
const jsonUrl = 'https://github.com/enobufs/node-menoh-yolo/releases/download/v0.0.1/yolo_v2_voc0712.json'

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
    .usage('[options]')
    .version(yolo.version)
    .option('-i, --input <pathname>', 'input file path', resolve)
    .option('-o, --output <pathname>', 'output file path', resolve)
    .option('-s, --score <number>', 'score threshold (0, 1.0] (defaults to 0.4)', parseFloat)
    .option('-x, --overlap <number>', 'overlap threshold (0, 1.0] (defaults to 0.7)', parseFloat)
    .option('-b, --browse', 'open output image with default viewer')
    .option('-v, --verbose', 'print more information on TTY')
    .parse(process.argv);


program.output = program.output || resolve('./out.jpg');

mkdirp(fileStore)
.then(() => {
    // check if onnx file exsits.
    return access(onnxPath, fs.constants.R_OK)
    .catch((err) => {
        if (err.code != 'ENOENT') {
            throw err;
        }
        console.log('downloading onnx file ...');
        return download(onnxUrl)
        .then(data => {
            fs.writeFileSync(onnxPath, data);
        });
    });
})
.then(() => {
    // check if config file exsits.
    return access(jsonPath, fs.constants.R_OK)
    .catch((err) => {
        if (err.code != 'ENOENT') {
            throw err;
        }
        console.log('downloading config file ...');
        return download(jsonUrl)
        .then(data => {
            fs.writeFileSync(jsonPath, data);
        });
    });
})
.then(() => {
    return readImageFile(program.input)
})
.then((buf) => {
    return jimp.read(buf)
    .then((image) => {
        const ts = [];
        const model = yolo.create();

        ts.push(Date.now()); // ts[0]

        model.build(onnxPath)
        .then(() => {
            const iBuf = model.inputBuffer;
            const originalShape = [image.bitmap.height, image.bitmap.width];

            ts.push(Date.now()); // ts[1]

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

            ts.push(Date.now()); // ts[2]

            return model.run(originalShape, runOpts)
        })
        .then((boxes) => {
            ts.push(Date.now()); // ts[3]

            return drawBoxes(program.output, boxes, buf);
        })
        .then(() => {
            ts.push(Date.now()); // ts[4]

            if (program.verbose) {
                console.log('ONNX loading & build : %d [msec]', ts[1] - ts[0]);
                console.log('Image resize & copy  : %d [msec]', ts[2] - ts[1]);
                console.log('Inference            : %d [msec]', ts[3] - ts[2]);
                console.log('Sort results & save  : %d [msec]', ts[4] - ts[3]);
            }

            if (program.browse) {
                require('opn')(program.output, { wait: false });
            }
        });
    })
})
.catch((err) => {
    console.error(err);
});

