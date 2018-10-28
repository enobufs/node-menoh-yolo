'use strict';


const ndarray = require('ndarray');
const dtype = require('dtype');
const menoh = require('menoh');
const Decoder = require('./decoder');
const _ = require('lodash');


class YoloV2 {
    // Initializes a new instance of YoloV2.
    // @constructor
    // @param {object) yoloV2Config Path to the onnx file.
    // @return {Promise} A promise.
    constructor(config) {
        this._config = _.defaults(config || {}, {
            inputName: 'Input_0',
            backendName: 'mkldnn',
            outputName: 'Conv_23',
            inSize: 416,
            grid: 13,
            nBoxes: 5,
            anchors: [
                [1.73145, 1.3221],
                [4.00944, 3.19275],
                [8.09892, 5.05587],
                [4.84053, 9.47112],
                [10.0071, 11.2364]
            ],
            labels: [
                "aeroplane", "bicycle", "bird", "boat", "bottle",
                "bus", "car", "cat", "chair", "cow",
                "diningtable", "dog", "horse", "motorbike", "person",
                "pottedplant", "sheep", "sofa", "train", "tvmonitor"
            ]
        });

        this._decoder = new Decoder(    this._config.grid,
                                        this._config.nBoxes,
                                        this._config.labels.length,
                                        this._config.anchors);

    }

    // Loads Yolo v2 onnx file.
    // @param {string) onnxPath Path to the onnx file.
    // @return {Promise} A promise.
    build(onnxPath) {
        return menoh.create(onnxPath)
        .then((builder) => {
            // Add input
            builder.addInput(this._config.inputName, [
                1,                      // batch size
                3,                      // number of channels
                this._config.inSize,    // height
                this._config.inSize     // width
            ]);

            // Add output
            builder.addOutput(this._config.outputName);

            // Build a new Model
            this._model = builder.buildModel({
                backendName: this._config.backendName
            })

            // Create a view for input buffer using ndarray.
            this._iData = (() => {
                const prof = this._model.getProfile(this._config.inputName);
                return ndarray(new (dtype(prof.dtype))(prof.buf.buffer), prof.dims);
            })();

            this._oData = (() => {
                const prof = this._model.getProfile(this._config.outputName);
                return ndarray(new (dtype(prof.dtype))(prof.buf.buffer), prof.dims);
            })();
        });
    }

    // @getter
    // @return {ndarray} Input image data in CHW format.
    // Each pixel value must be a floating point value
    // between 0.0 and 1.0.
    get inputBuffer() {
        return this._iData.pick(0);
    }

    // @getter
    // @return {ndarray} Output buffer.
    get outputBuffer() {
        return this._oData.pick(0);
    }

    // Run the YoloV2 inference.
    // @param {array} originalShape Original image shape used to calculate the actual
    // @param {object} option Optional parameters.
    // @param {number} option.scoreThresh A conidence of prediction. [0, 1].
    // @param {number} option.overlapThresh Overlap area ratio of overlaped binding box to be
    // fitered out.
    // location of the binding boxes.
    // @return {Promise} A promise that resolves to inference results.
    run(originalShape, option) {
        return this._model.run()
        .then(() => {
            return this.decode(this.outputBuffer, originalShape, option);
        });
    }

    // Decods the result with optional parameters.
    // @param {ndarray} res A result of one image returned by run() method.
    // @param {object} option Optional parameters.
    // @param {number} option.scoreThresh A conidence of prediction. [0, 1].
    // @param {number} option.overlapThresh Overlap area ratio of overlaped binding box to be
    // fitered out.
    // @param {number} option.originalShape Original image shape used to calculate the actual
    // location of the binding boxes.
    // @return {array} Returns an array of bounding boxes.
    decode(result, originalShape, option) {
        const boxes = this._decoder.decode(result, originalShape, option);

        // Add label to each box.
        boxes.forEach((box) => {
            box.label = this._config.labels[box.classId];
        });

        return boxes;
    }
}

module.exports = YoloV2;

