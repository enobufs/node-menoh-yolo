'use strict';

const Decoder = require('../lib/decoder');
const assert = require('assert');
const helper = require('./helper');
const ndarray = require('ndarray');
const sinon = require('sinon');


describe('Docoder tests', function () {
    let config;
    let dec;
    let sandbox;

    before(function () {
        // The path for require must be relative to this file.
        config = require('./data/yolo_v2_voc0712.json');
        sandbox = sinon.createSandbox();
    });

    afterEach(function () {
        sandbox.restore();
    });

    describe('NMS tests', function () {
        it('overlap detection - no overlap (x-axis)', function () {
            const rec1 = {x:0, y:0, w:10, h:10};
            const rec2 = {x:10, y:0, w:10, h:10};
            const iou = Decoder._calculateIoU(rec1, rec2);
            assert.strictEqual(iou, 0, 'should have no overlap');
        });

        it('overlap detection - no overlap (y-axis)', function () {
            const rec1 = {x:0, y:0, w:10, h:10};
            const rec2 = {x:0, y:10, w:10, h:10};
            const iou = Decoder._calculateIoU(rec1, rec2);
            assert.strictEqual(iou, 0, 'should have no overlap');
        });

        it('no overlap in the same class', function () {
            let boxes = [{
                rec: {x:0, y:0, w:10, h:10},
                id: 1,
                score: 0.8
            }, {
                rec: {x:10, y:0, w:0, h:10},
                id: 2,
                score: 0.8
            }];
            boxes = Decoder._sortBoxes([boxes], 0.5);
            assert.equal(boxes.length, 2);
        });

        it('higher score wins', function () {
            let boxes = [{
                rec: {x:0, y:0, w:10, h:10},
                id: 1,
                score: 0.7
            }, {
                rec: {x:0, y:0, w:10, h:10},
                id: 2,
                score: 0.8
            }];
            boxes = Decoder._sortBoxes([boxes], 0.5);
            assert.equal(boxes.length, 1);
            assert.equal(boxes[0].id, 2);
        });
    });

    describe('Instance tests', function () {
        let result;

        before(function () {
            result = helper.loadDummyResult();
        });

        beforeEach(function () {
            dec = new Decoder(13, 5, 20, config.anchors);
        });

        it('decode default option', function () {
            const spy = sandbox.spy(dec, '_getBoxes');
            const dummy = ndarray(new Float32Array(13 * 13 * 125), [13, 13, 125]);
            dec.decode(dummy, [576, 768]);
            assert.strictEqual(spy.args[0][1], 0.4);
        });

        it('decode success', function () {
            const orgShape = [576, 768];
            const option = {
                scoreThresh: 0.7,
                overlapThresh: 0.7
            };
            const boxes = dec.decode(result, orgShape, option);
            //console.log('Final:\n', boxes);

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
        });
    });
});
