'use strict';

const nj = require('numjs');
const _ = require('lodash');

const {
    Rectangle,
    overlapRectangle,
    sigmoid,
    argMax,
    sortBoxes } = require('./tool');

class Decoder {
    constructor(grid, nBoxes, nLabels, anchors) {
        this._grid = grid;
        this._nBoxes = nBoxes;
        this._nLabels = nLabels;
        this._anchors = anchors;
    }

    _getBoxes(result, scoreThresh, originalShape) {
        // result: [ 13, 13, 5, 25 ]

        const boxes = [];
        for (let gy = 0; gy < this._grid; ++gy) {
            for (let gx = 0; gx < this._grid; ++gx) {
                for (let bi = 0; bi < this._nBoxes; ++bi) {
                    const a = result.pick(gy, gx, bi);
                    const loc = a.slice([0, 4]).tolist();
                    const conf = sigmoid(a.get(4));
                    const probs = nj.softmax(a.slice([5, 5 + this._nLabels]));
                    const scores = nj.multiply(probs, conf);

                    const score = nj.max(scores);
                    if (score < scoreThresh) {
                        continue;
                    }

                    const classId = argMax(scores.tolist())
                    const block = {
                        h: originalShape[0] / this._grid,
                        w: originalShape[1] / this._grid
                    }

                    const rec = (() => {
                        const x = (gx + sigmoid(loc[1])) * block.w;
                        const y = (gy + sigmoid(loc[0])) * block.h;
                        const w = Math.exp(loc[3]) * this._anchors[bi][1] * block.w
                        const h = Math.exp(loc[2]) * this._anchors[bi][0] * block.h;
                        //   +---------------------------------+ -
                        //   |                                 | ^ 
                        //   |              (x,y)              | |
                        //   |                *                | h  
                        //   |                                 | |
                        //   |                                 | v
                        //   +---------------------------------+ -
                        //   |<-------------  w  ------------->|

                        return new Rectangle(...[x - w/2, y - h/2, w, h].map(Math.floor));
                        // (x,y)                             
                        //   *---------------------------------+ -
                        //   |                                 | ^
                        //   |                                 | |
                        //   |           Rectangle             | h
                        //   |                                 | |
                        //   |                                 | v
                        //   +---------------------------------+ _
                        //   |<-------------  w  ------------->|
                    })();

                    boxes.push({
                        id: bi * gy * gx,
                        rec: rec,
                        score: score,
                        classId: classId
                    });
                }
            }
        }

        return boxes;
    }

    static overlapLen(l0, r0, l1, r1) {
        if (r0 <= l1 || r1 <= l0) {
            return 0;
        }
        return [Math.max(l0, l1), Math.min(r0, r1)];
    }

    static calculateIoU(b0, b1) {
        const dx = Decoder.overlapLen(b0.x, b0.x + b0.w, b1.x, b1.x + b1.w);
        if (!dx) return 0;
        const dy = Decoder.overlapLen(b0.y, b0.y + b0.h, b1.y, b1.y + b1.h);
        if (!dy) return 0;
        const intersection = (dx[1] - dx[0]) * (dy[1] - dy[0]);
        const union = b0.w * b0.h + b1.w * b1.h - intersection;

        return intersection / union;
    }

    static nonMaximumSuppression(boxes, thresh, limit) {
        if (boxes.length === 0) {
            return []
        }

        const selected = []

        // sort by score (in descending order)
        boxes.sort((l, r) => r.score - l.score);

        for (let i = 0; i < boxes.length; ++i) {
            const b = boxes[i];
            let skip = false;
            for (let j = 0; j < selected.length; ++j) {
                const s = selected[j];
                const iou = Decoder.calculateIoU(b.rec, s.rec);
                if (iou >= thresh) {
                    skip = true;
                    break;
                }
            }
            if (skip) {
                continue;
            }
            selected.push(b);
            if (selected.length >= limit) {
                break;
            }
        }
        return selected;
    }

    static sortBoxes(boxes, overlapThresh) {
        // split boxes into classes
        const dic = {};
        boxes.forEach((box) => {
            const classId = box.classId;
            let cboxes = dic[classId]; // classified boxes
            if (!cboxes) {
                dic[classId] = [ box ];
            } else {
                cboxes.push(box)
            }
        });

        let selected = [];
        Object.keys(dic).forEach((classId) => {
            const cboxes = dic[classId];
            const cselected = Decoder.nonMaximumSuppression(cboxes, overlapThresh, 5);
            selected = selected.concat(cselected);
        });

        return selected.sort((l, r) => r.score - l.score);
    }



    decode(result, originalShape, option) {
        option = _.defaults(option || {}, {
            scoreThresh: 0.4,
            overlapThresh: 0.7
        });

        // result: [ 13, 13, 125 ]

        result = new nj.NdArray(result);
        // result: [ 125, 13, 13 ]
        result = result.transpose(1, 2, 0);
        // result:[ 13, 13, 125 ]
        result = result.reshape(    this._grid,
                                    this._grid,
                                    this._nBoxes,
                                    (this._nBoxes + this._nLabels));
        // result: [ 13, 13, 5, 25 ]

        const boxes = this._getBoxes(result, option.scoreThresh, originalShape);
        return Decoder.sortBoxes(boxes, option.overlapThresh);
    }
}

module.exports = Decoder;

