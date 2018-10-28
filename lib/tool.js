'use strict';

const nj  = require('numjs');

class Rectangle {
    constructor(x, y, w, h) {
        this.x = x;
        this.y = y;
        this.w = w;
        this.h = h;
    }
    area() {
        return this.w * this.h
    }
}

function overlapLen(l0, r0, l1, r1) {
    if (r0 <= l1 || r1 <= l0) {
        return null;
    }
    return [Math.max(l0, l1), Math.min(r0, r1)];
}

function overlapRectangle(b0, b1) {
    const dx = overlapLen(b0.x, b0.x + b0.w, b1.x, b1.x + b1.w);
    if (!dx) return null;
    const dy = overlapLen(b0.y, b0.y + b0.h, b1.y, b1.y + b1.h);
    if (!dy) return null;
    return new Rectangle(dx[0], dy[0], dx[1] - dx[0], dy[1] - dy[0]);
}

function sigmoid(x) {
    return 1.0 / (Math.exp(-x) + 1.0)
}

function argMax(array) {
    return array.map((x, i) => [x, i]).reduce((r, a) => (a[0] > r[0] ? a : r))[1];
}

function sortBoxes(boxes, overlapThresh) {
    const dic = {};
    boxes.forEach((box) => {
        console.log('examining box %d', box.id);
        const classId = box.classId;
        const cboxes = dic[classId]; // classified boxes
        if (!cboxes) {
            console.log('adding box %d (initial)', box.id);
            dic[box.classId] = { [box.id]: box };
            return;
        }

        const toRm = [];
        let addNewBox = true;
        let reason = '';
        Object.keys(cboxes).forEach((boxId) => {
            const box2 = cboxes[boxId];
            const rec = overlapRectangle(box.rec, box2.rec);
            if (rec) {
                const overlap = rec.area();
                const r1 = overlap / box.rec.area();  // overlap ratio on the new box
                const r2 = overlap / box2.rec.area(); // overlap ratio on the existing box

                if (r1 >= overlapThresh) {
                    if (r2 >= overlapThresh) {
                        if (box.score < box2.score) {
                            addNewBox = false;
                            reason = `score is lower than box ${box2.id}`;
                        }
                        toRm.push(box2.id);
                    } else {
                        addNewBox = false;
                        reason = `large overlap with the box ${box2.id}`;
                    }
                } else {
                    if (r2 >= overlapThresh) {
                        toRm.push(box2.id);
                    }
                }
            }
        });

        if (addNewBox) {
            toRm.forEach((boxId) => {
                console.log('removing box %d', boxId);
                cboxes[boxId].id = -1;
                delete cboxes[boxId]
            });
            console.log('adding box %d', box.id);
            cboxes[box.id] = box;
        } else {
            console.log('ignoring box %d (%s)', box.id, reason);
            box.id = -1;
        }
    });

    return boxes.filter((box) => box.id >= 0).sort((l, r) => r.score - l.score);
}



exports.Rectangle = Rectangle;
exports.overlapLen = overlapLen;
exports.overlapRectangle = overlapRectangle;
exports.sigmoid = sigmoid;
exports.argMax = argMax;
exports.sortBoxes = sortBoxes;
