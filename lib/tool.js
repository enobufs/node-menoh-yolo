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



exports.Rectangle = Rectangle;
exports.overlapLen = overlapLen;
exports.overlapRectangle = overlapRectangle;
exports.sigmoid = sigmoid;
exports.argMax = argMax;
