'use strict';

const Decoder = require('../lib/decoder');
const assert = require('assert');
const helper = require('./helper');
const { Rectangle, overlapRectangle } = require('../lib/tool');
const ndarray = require('ndarray');
const sinon = require('sinon');


describe('Docoder tests', function () {
    let config;
    let dec;
    let sandbox;
    

    before(function () {
        // The path for require must be relative to this file.
        config = require('./data/yolo_v2_voc0712.json');
        sandbox = sinon.sandbox.create();
    });

    afterEach(function () {
        sandbox.restore();
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
            const dummy = ndarray(new Float32Array(13*13*125), [13, 13, 125]);
            dec.decode(dummy, [576, 768]);
            assert.strictEqual(spy.args[0][1], 0.4);
        });

        it('decode success', function () {
            const orgShape = [576, 768];
            const option = {
                scoreThresh: 0.7,
                overlapThresh: 0.7
            };
            const boxes = dec.decode(result, orgShape, option)
            //console.log('Final:\n', boxes);

            assert.equal(boxes.length, 3);
            assert.equal(boxes[0].classId, 11);
            assert.equal(boxes[1].classId, 1);
            assert.equal(boxes[2].classId, 6);
            const expRecs = [
                new Rectangle(105, 198, 233, 342),
                new Rectangle(86, 113, 474, 279),
                new Rectangle(471, 73, 222, 102)
            ]
            const expScores = [ 0.8, 0.79, 0.71 ]
            boxes.forEach((box, i) => {
                assert.ok(box.score >= expScores[i], 'score is too low');
                assert.deepEqual(box.rec, expRecs[i]);
            });
        });
    });
});
