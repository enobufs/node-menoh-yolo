'use strict';

const YoloV2 = require('./lib/yoloV2');

exports.version = require('./package').version;
exports.create = (config) => {
    return new YoloV2(config);
};

