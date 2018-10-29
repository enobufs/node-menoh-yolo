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

    _getBoxes(ans, scoreThresh, originalShape) {
        const boxes = [];
        for (let gy = 0; gy < this._grid; ++gy) {
            for (let gx = 0; gx < this._grid; ++gx) {
                for (let bi = 0; bi < this._nBoxes; ++bi) {
                    const a = ans.pick(gy, gx, bi);
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
                        //   |                                 | |
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
        return sortBoxes(boxes, option.overlapThresh);
    }
}

module.exports = Decoder;
