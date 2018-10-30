'use strict';

const nj = require('numjs');
const _ = require('lodash');

const { sigmoid } = require('./tool');

class Decoder {
    constructor(grid, nBoxes, nLabels, anchors) {
        this._grid = grid;
        this._nBoxes = nBoxes;
        this._nLabels = nLabels;
        this._anchors = anchors;
    }

    _getBoxes(result, scoreThresh, originalShape) {
        // result: [ 13, 13, 5, 25 ]

        const cboxes = Array.apply(null, Array(this._nLabels)).map(() => []);

        for (let gy = 0; gy < this._grid; ++gy) {
            for (let gx = 0; gx < this._grid; ++gx) {
                for (let bi = 0; bi < this._nBoxes; ++bi) {
                    const stack = result.pick(gy, gx, bi);
                    const loc = stack.slice([0, 4]).tolist();
                    const obj = sigmoid(stack.get(4));
                    const conf = stack.slice([5, 5 + this._nLabels]).exp();
                    const sum = conf.sum();

                    const scores = nj.multiply(conf, obj / sum);

                    const block = {
                        h: originalShape[0] / this._grid,
                        w: originalShape[1] / this._grid
                    }

                    scores.tolist().forEach((score, classId) => {
                        if (score < scoreThresh) {
                            return;
                        }

                        const rec = (() => {
                            const x = (gx + sigmoid(loc[1])) * block.w;
                            const y = (gy + sigmoid(loc[0])) * block.h;
                            const w = Math.exp(loc[3]) * this._anchors[bi][1] * block.w;
                            const h = Math.exp(loc[2]) * this._anchors[bi][0] * block.h;

                            // Convert:
                            //   +---------------------------------+ -
                            //   |                                 | ^ 
                            //   |              (x,y)              | |
                            //   |                *                | h
                            //   |                                 | |
                            //   |                                 | v
                            //   +---------------------------------+ -
                            //   |<-------------  w  ------------->|
                            //
                            // To:
                            // (x,y)
                            //   *---------------------------------+ -
                            //   |                                 | ^
                            //   |                                 | |
                            //   |                                 | h
                            //   |                                 | |
                            //   |                                 | v
                            //   +---------------------------------+ _
                            //   |<-------------  w  ------------->|
                            return {
                                x: Math.floor(x - w/2),
                                y: Math.floor(y - h/2),
                                w: Math.floor(w),
                                h: Math.floor(h)
                            };
                        })();

                        cboxes[classId].push({
                            id: bi * gy * gx,
                            rec: rec,
                            score: score
                        });
                    });
                }
            }
        }

        return cboxes;
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

    static nonMaximumSuppression(boxes, thresh/*, limit*/) {
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
            /*
            if (typeof(limit) == 'number' && selected.length >= limit) {
                break;
            }
            */
        }
        return selected;
    }

    static sortBoxes(cboxes, overlapThresh) {
        let selected = [];

        cboxes.forEach((boxes, classId) => {
            const cselected = Decoder.nonMaximumSuppression(boxes, overlapThresh);
            selected = selected.concat(cselected.map((box) => {
                // Shallow copy with assigning the determined class ID
                // without altering the original box object.
                return Object.assign({ classId: classId }, box);
            }));
        });

        return selected.sort((l, r) => r.score - l.score);
    }



    decode(result, originalShape, option) {
        option = _.defaults(option || {}, {
            scoreThresh: 0.4,
            overlapThresh: 0.5
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

        const cboxes = this._getBoxes(result, option.scoreThresh, originalShape);
        return Decoder.sortBoxes(cboxes, option.overlapThresh);
    }
}

module.exports = Decoder;

