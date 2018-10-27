'use strict';

const yolo = require('..');
const { INPUT_IMAGE_LIST, loadInputImages } = require('./helper');
const assert = require('assert');
const onnxPath = './test/data/YOLOv2_2018_09_27.onnx';
const { Rectangle } = require('../lib/tool');

describe('Yolo V2 tests', function () {
    let config;

    before(function () {
        // The path for require must be relative to this file.
        config = require('./data/yolo_v2_voc0712.json');
    });

    //after(function () {});

    describe('Instance tests', function () {
        let model;

        beforeEach(function () {
            model = yolo.create({
                inputName: config.input,
                outputName: config.output,
                inSize: config.insize,
                anchors: config.anchors,
                labels: config.label_names
            });

            return model.build(onnxPath);
        });

        it('test', function () {
            const iBuf = model.inputBuffer;
            const oBuf = model.outputBuffer;
            return loadInputImages()
            .then((images) => {
                images.forEach((image, batchIdx) => {
                    // Crop the input image to a square shape.
                    //cropToSquare(image);

                    // Resize it to this._config.inSize x this._config.inSize.
                    image.resize(config.insize, config.insize);

                    // Now, copy the image data into to the input buffer in NCHW format.
                    image.scan(0, 0, image.bitmap.width, image.bitmap.height, (x, y, idx) => {
                        for (let c = 0; c < 3; ++c) {
                            let val = image.bitmap.data[idx + c];
                            val = val / 255.0;
                            iBuf.set(batchIdx, c, y, x, val);
                        }
                    });
                });

                return model.run();
            })
            .then(() => {
                const result = oBuf.pick(0);
                const boxes = model.decode(result, {
                    scoreThresh: 0.7,
                    overlapThresh: 0.7,
                    originalShape: [576, 768]
                });

                assert.equal(boxes.length, 3);
                assert.equal(boxes[0].classId, 11);
                assert.equal(boxes[1].classId, 1);
                assert.equal(boxes[2].classId, 6);
                const expRecs = [
                    new Rectangle(221, 370, 455, 713),
                    new Rectangle(323, 253, 797, 532),
                    new Rectangle(583, 124, 805, 227)
                ]
                const expScores = [ 0.8, 0.79, 0.71 ]
                boxes.forEach((box, i) => {
                    assert.ok(box.score >= expScores[i], 'score is too low');
                });
            });
        });
    });
});
