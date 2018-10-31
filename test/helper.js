'use strict';

const fs = require('fs');
const nj = require('numjs');
const jimp = require('jimp');

function loadDummyResult() {
    const json = fs.readFileSync('./test/fixture/result.json', { encoding: 'utf8' });
    const arr = JSON.parse(json);
    return (nj.array(arr, 'float32')).selection;
}

const INPUT_IMAGE_LIST = [
    './test/fixture/dog.jpg',
    './test/fixture/bedlington_terrier.jpg',
];

function loadInputImages() {
    return Promise.all(INPUT_IMAGE_LIST.map((filename) => jimp.read(filename)));
}

exports.loadDummyResult = loadDummyResult;
exports.INPUT_IMAGE_LIST = INPUT_IMAGE_LIST;
exports.loadInputImages = loadInputImages;

