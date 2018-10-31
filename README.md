# menoh-yolo
[![npm version](https://badge.fury.io/js/menoh-yolo.svg)](https://badge.fury.io/js/menoh-yolo)
[![Build Status](https://travis-ci.org/enobufs/node-menoh-yolo.svg?branch=master)](https://travis-ci.org/enobufs/node-menoh-yolo)
[![Coverage Status](https://coveralls.io/repos/github/enobufs/node-menoh-yolo/badge.svg?branch=master)](https://coveralls.io/github/enobufs/node-menoh-yolo?branch=master)

Object classification using YOLO v2 powered by [Menoh](https://github.com/pfnet-research/menoh).

This module is made on top of npm module, [menoh](https://github.com/pfnet-research/node-menoh), which is a wrapper of Menoh (C/C++) core.
The modules supports both command line interface and API.

## Requirements
You must have the following libraries installed in advance.
* MKL_DNN Library (0.14 or later)
* Protocol Buffers (2.6.1 or later)
* Menoh (core (1.1.1 or later)
> See [Building Menoh](https://github.com/pfnet-research/menoh/blob/v1.1.1/BUILDING.md) for more
> details.

## Installation (command line use)
```
npm install -g menoh-yolo
```

## Usage

```
$ menoh-yolo --help
Usage: menoh-yolo [options]

Options:
  -v, --version            Output the version number
  -i, --input <pathname>   Input file path.
  -o, --output <pathname>  Output file path.
  -s, --score <number>     Score threshold (0, 1.0] (defaults to 0.4)
  -x, --overlap <number>   Overlap threshold (0, 1.0] (defaults to 0.5)
  -b, --browse             Show output image with default viewer.
  -h, --help               Output usage information
```

### Example
```
$ menoh-yolo -i ./dog.jpg -b
```

> When you use the CLI for the first time, it automatically downloads ONNX file and a
> configuration file into $HOME/.menoh-yolo/.

By giving `-b` option, it will open up a viewer to show the output result like below:

![Alt text](./doc/output.png?raw=true "Output Image")


## API
(TODO)

> See `./bin/cli.js` as an example.
