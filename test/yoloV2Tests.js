'use strict';

const yolo = require('..');
const { loadInputImages } = require('./helper');
const assert = require('assert');
const sinon = require('sinon');
const onnxPath = './test/data/yolo_v2_voc0712.onnx';

describe('Yolo V2 tests', function () {
    let config;
    let imageList;

    before(function () {
        // The path for require must be relative to this file.
        config = require('./data/yolo_v2_voc0712.json');

        return loadInputImages()
            .then((images) => {
                imageList = images;
            });
    });

    //after(function () {});
    describe('Class tests', function () {
        it('config default values', function () {
            const model = yolo.create();
            assert.deepEqual(model._config, {
                inputName: 'Input_0',
                backendName: 'mkldnn',
                outputName: 'Conv_23',
                inSize: 416,
                grid: 13,
                nBoxes: 5,
                anchors: [
                    [1.73145, 1.3221],
                    [4.00944, 3.19275],
                    [8.09892, 5.05587],
                    [4.84053, 9.47112],
                    [10.0071, 11.2364]
                ],
                labels: [
                    "aeroplane", "bicycle", "bird", "boat", "bottle",
                    "bus", "car", "cat", "chair", "cow",
                    "diningtable", "dog", "horse", "motorbike", "person",
                    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
                ]
            });
        });
    });

    describe('Instance tests', function () {
        let model;
        let sandbox;

        before(function () {
            sandbox = sinon.createSandbox();
        });

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

        afterEach(function () {
            sandbox.restore();
        });

        it('dog', function () {
            const iBuf = model.inputBuffer;
            const image = imageList[0]; // dog.jpg
            const originalShape = [image.bitmap.height, image.bitmap.width];

            // Resize it to this._config.inSize x this._config.inSize.
            assert.equal(model.inSize, config.insize);
            image.resize(model.inSize, model.inSize);

            // Now, copy the image data into to the input buffer in NCHW format.
            image.scan(0, 0, image.bitmap.width, image.bitmap.height, (x, y, idx) => {
                for (let c = 0; c < 3; ++c) {
                    let val = image.bitmap.data[idx + c];
                    val = val / 255.0;
                    iBuf.set(c, y, x, val);
                }
            });

            return model.run(originalShape, {
                scoreThresh: 0.7
            }).then((boxes) => {
                assert.equal(boxes.length, 3);
                assert.equal(boxes[0].classId, 11);
                assert.equal(boxes[1].classId, 1);
                assert.equal(boxes[2].classId, 6);
                const expRecs = [
                    {x:105, y:198, w:233, h:342},
                    {x:86, y:113, w:474, h:279},
                    {x:471, y:73, w:222, h:102}
                ];
                const expScores = [ 0.81, 0.80, 0.73 ];
                boxes.forEach((box, i) => {
                    assert.ok(box.score >= expScores[i], 'score is too low');
                    assert.deepEqual(box.rec, expRecs[i]);
                });

                // Try decode again with direct docode() method call.
                boxes = model.decode(model.outputBuffer, originalShape, {
                    scoreThresh: 0.7
                });
                assert.equal(boxes.length, 3);
                assert.equal(boxes[0].classId, 11);
                assert.equal(boxes[1].classId, 1);
                assert.equal(boxes[2].classId, 6);
                boxes.forEach((box, i) => {
                    assert.ok(box.score >= expScores[i], 'score is too low');
                    assert.deepEqual(box.rec, expRecs[i]);
                });
            });
        });

        it('one box for two classes', function () {
            const iBuf = model.inputBuffer;
            const image = imageList[1]; // dog.jpg
            const originalShape = [image.bitmap.height, image.bitmap.width];

            // Resize it to this._config.inSize x this._config.inSize.
            image.resize(config.insize, config.insize);

            // Now, copy the image data into to the input buffer in NCHW format.
            image.scan(0, 0, image.bitmap.width, image.bitmap.height, (x, y, idx) => {
                for (let c = 0; c < 3; ++c) {
                    let val = image.bitmap.data[idx + c];
                    val = val / 255.0;
                    iBuf.set(c, y, x, val);
                }
            });

            const spy = sandbox.spy(model._decoder, '_getBoxes');

            return model.run(originalShape, {
                scoreThresh: 0.35
            }).then((boxes) => {
                assert.equal(boxes.length, 2);
                assert.equal(boxes[0].classId, 16); // sheep
                assert.ok(boxes[0].score >= 0.7);
                assert.equal(boxes[1].classId, 11); // dog
                assert.ok(boxes[1].score >= 0.4);

                const cboxes = spy.returnValues[0];

                // cboxes[11] and cboxes[16] must share one identical box.
                let found = false;
                for (let i = 0; !found && i < cboxes[11].length; ++i) {
                    for (let j = 0; !found && j < cboxes[16].length; ++j) {
                        if (cboxes[11][i].id === cboxes[16][j].id) {
                            assert.equal(cboxes[11][i].rec, cboxes[16][j].rec);
                            found = true;
                        }
                    }
                }
                assert.ok(found, 'one identical box should be in class 11 & 16');
            });
        });
    });
});
