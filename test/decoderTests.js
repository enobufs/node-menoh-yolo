'use strict';

const Decoder = require('../lib/decoder');
const assert = require('assert');
const helper = require('./helper');
const { Rectangle, overlapRectangle } = require('../lib/tool');

describe('Docoder tests', function () {
    let config;
    let dec;

    before(function () {
        config = require('./data/yolo_v2_voc0712.json');
    });

    //after(function () {});

    describe('Instance tests', function () {
        let result;

        before(function () {
            result = helper.loadDummyResult();
        });

        beforeEach(function () {
            dec = new Decoder(13, 5, 20, config.anchors);
        });

        it('decode success', function () {
            const orgShape = [576, 768];
            const scoreThresh = 0.7;
            const overlapThresh = 0.7;
            const boxes = dec.decode(result, scoreThresh, overlapThresh, orgShape)
            //console.log('Final:\n', boxes);

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
