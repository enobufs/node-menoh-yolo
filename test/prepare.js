'use strict';

const assert = require('assert');

assert.almostEqual = (() => {
    const abs = Math.abs;
    const min = Math.min;
    const FLT_EPSILON = 1.19209290e-7;

    return (a, b, errorMessage) => {
        var d = abs(a - b);
        if (d <= FLT_EPSILON) {
            return;
        }
        if (d <= FLT_EPSILON * min(abs(a), abs(b))) {
            return;
        }
        assert.strictEqual(a, b, errorMessage);
    };
})();

// Check if test data has already beenn downloaded.
