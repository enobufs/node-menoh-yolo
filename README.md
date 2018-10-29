# menoh-yolo
Object classification using YOLO v2 powered by [Menoh](https://github.com/pfnet-research/menoh).

This module is made on top of npm module, [menoh](https://github.com/pfnet-research/node-menoh), which is a wrapper of Menoh (C/C++) core.
The modules supports both command line interface and API.

## Installation (command line use)
```
npm install -g menoh-yolo
```

> When you use the CLI for the first time, it automatically downloads ONNX file and a
> configuration file into $HOME/.menoh-yolo/.

## Usage

```
$ menoh-yolo --help
Usage: menoh-yolo [options]

Options:
  -v, --version            output the version number
  -i, --input <pathname>   Input file path.)
  -o, --output <pathname>  Output file path.)
  -s, --score <number>     Score threshold (0, 1.0] (defaults to 0.4)
  -x, --overlap <number>   Overlap threshold (0, 1.0] (defaults to 0.7)
  -b, --browse             Open output image with default viewer.
  -h, --help               output usage information
```

### Example
```
$ menoh-yolo -i ./dog.jpg -b
```

By giving `-b` option, it will open up a viewer to show the output result like below:

![Alt text](./doc/output.png?raw=true "Output Image")


## API
(TODO)

> See `./bin/cli.js` as an example.
