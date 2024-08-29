# DreamHOI

We present DreamHOI, a novel method for zero-shot synthesis of human-object interactions (HOIs), enabling a 3D human model to realistically interact with any given object based on a textual description.

[Website](https://dreamhoi.github.io/)
[Paper](TODO)

## Installation

Please refer to [installation.md](docs/installation.md)

###

## Tricks
1. Tune hyperparams
    1. IF/MV ratios
    2. individual time steps
    3. Loss lambdas - sometimes disabling some losses works better
    4. Tune mesh position & rotation (make sure face front)
    5. Learning rate (maybe not so important)
    6. Background (random aug, eval color -> openpose)
2. Prompting
    1. Increase guidance_scale
    2. Make prompt more specific (limb positions etc, IF understands very specific prompts; describe mesh accurately so doesn't make new object)
    3. Negative prompting
3. 

## License
Some parts of this project uses third-party software. See [LICENSE](LICENSE) for their respective notices and licenses.
