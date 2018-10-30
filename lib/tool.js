'use strict';

function sigmoid(x) {
    return 1.0 / (Math.exp(-x) + 1.0)
}

exports.sigmoid = sigmoid;
