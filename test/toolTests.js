'use strict';

const tool = require('../lib/tool');
const assert = require('assert');

describe('Tool tests', function () {
    it('sigmoid function test', function () {
        assert.equal(tool.sigmoid(0), 0.5);
    });

    describe('sortBox() tests', function () {
    });
});
