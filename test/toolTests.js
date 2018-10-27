'use strict';

const tool = require('../lib/tool');
const assert = require('assert');

describe('Tool tests', function () {
    it('sigmoid function test', function () {
        assert.equal(tool.sigmoid(0), 0.5);
    });

    it('rectangle axis value swap', function () {
        const rec = new tool.Rectangle(4, 5, 2, 3);
        assert.equal(rec.x0, 2);
        assert.equal(rec.y0, 3);
        assert.equal(rec.x1, 4);
        assert.equal(rec.y1, 5);
    });

    it('overlap detection - no overlap (x-axis)', function () {
        const rec1 = new tool.Rectangle(0, 0, 10, 10);
        const rec2 = new tool.Rectangle(10, 0, 20, 10);
        const overlap = tool.overlapRectangle(rec1, rec2);
        assert.strictEqual(overlap, null, 'should have no overlap');
    });

    it('overlap detection - no overlap (y-axis)', function () {
        const rec1 = new tool.Rectangle(0, 0, 10, 10);
        const rec2 = new tool.Rectangle(0, 10, 10, 20);
        const overlap = tool.overlapRectangle(rec1, rec2);
        assert.strictEqual(overlap, null, 'should have no overlap');
    });

    describe('sortBox() tests', function () {
        it('no overlap in the same class', function () {
            let boxes = [{
                rec: new tool.Rectangle(0, 0, 10, 10),
                id: 1,
                classId: 1,
                score: 0.8
            }, {
                rec: new tool.Rectangle(10, 0, 20, 10),
                id: 2,
                classId: 1,
                score: 0.8
            }]
            boxes = tool.sortBoxes(boxes, 0.7);
            assert.equal(boxes.length, 2);
        });

        it('small overlap in the same class', function () {
            let boxes = [{
                rec: new tool.Rectangle(0, 0, 10, 10),
                id: 1,
                classId: 1,
                score: 0.8
            }, {
                rec: new tool.Rectangle(0, 0, 100, 100),
                id: 2,
                classId: 1,
                score: 0.8
            }]
            boxes = tool.sortBoxes(boxes, 0.7);
            assert.equal(boxes.length, 1);
            assert.equal(boxes[0].id, 2);
        });

        it('new box almost inside the exising one', function () {
            let boxes = [{
                rec: new tool.Rectangle(0, 0, 100, 100),
                id: 1,
                classId: 1,
                score: 0.8
            }, {
                rec: new tool.Rectangle(70, 70, 101, 101),
                id: 2,
                classId: 1,
                score: 0.8
            }]
            boxes = tool.sortBoxes(boxes, 0.7);
            assert.equal(boxes.length, 1, 'there should be only one box');
            assert.equal(boxes[0].id, 1, 'the box 1 should be there');
        });

        it('complete overlap, delete the existing one', function () {
            let boxes = [{
                rec: new tool.Rectangle(0, 0, 100, 100),
                id: 1,
                classId: 1,
                score: 0.8
            }, {
                rec: new tool.Rectangle(0, 0, 100, 100),
                id: 2,
                classId: 1,
                score: 0.9
            }]
            boxes = tool.sortBoxes(boxes, 0.7);
            assert.equal(boxes.length, 1, 'there should be only one box');
            assert.equal(boxes[0].id, 2, 'the box 1 should be there');
        });

        it('small overlap for both boxes', function () {
            let boxes = [{
                rec: new tool.Rectangle(0, 0, 100, 100),
                id: 1,
                classId: 1,
                score: 0.8
            }, {
                rec: new tool.Rectangle(90, 90, 190, 190),
                id: 2,
                classId: 1,
                score: 0.9
            }]
            boxes = tool.sortBoxes(boxes, 0.7);
            assert.equal(boxes.length, 2, 'there should be two box');
            // Notice: boxes are sorted by the score.
            assert.equal(boxes[0].id, 2, 'the box 2 should be there');
            assert.equal(boxes[1].id, 1, 'the box 1 should be there');
        });
    });
});
