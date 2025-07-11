# Training Guide

Check out a [video training guide by Thorsten Müller](https://www.youtube.com/watch?v=b_we_jma220)

For Windows, see [ssamjh's guide using WSL](https://ssamjh.nz/create-custom-piper-tts-voice/)

---

Training a voice for Piper involves 3 main steps:

1. Preparing the dataset
2. Training the voice model
3. Exporting the voice model

Choices must be made at each step, including:

* The model "quality"
  * low = 16,000 Hz sample rate, [smaller voice model](https://github.com/rhasspy/piper/blob/master/src/python/piper_train/vits/config.py#L30)
  * medium = 22,050 Hz sample rate, [smaller voice model](https://github.com/rhasspy/piper/blob/master/src/python/piper_train/vits/config.py#L30)
  * high = 22,050 Hz sample rate, [larger voice model](https://github.com/rhasspy/piper/blob/master/src/python/piper_train/vits/config.py#L45)
* Single or multiple speakers
* Fine-tuning an [existing model](https://huggingface.co/datasets/rhasspy/piper-checkpoints/tree/main) or training from scratch
* Exporting to [onnx](https://github.com/microsoft/onnxruntime/) or PyTorch

## Getting Started

Start by installing system dependencies:

``` sh
sudo apt-get install python3-dev
```

Then create a Python virtual environment:

``` sh
cd piper/src/python
python3 -m venv .venv
source .venv/bin/activate
pip3 install --upgrade pip
pip3 install --upgrade wheel setuptools
pip3 install -e .
```

Run the `build_monotonic_align.sh` script in the `src/python` directory to build the extension.

Ensure you have [espeak-ng](https://github.com/espeak-ng/espeak-ng/) installed (`sudo apt-get install espeak-ng`).

json
{
    "audio": {
        "sample_rate": 22050
    },
    "espeak": {
        "language": "en-us"
    },
    "num_symbols": 256,
    "num_speakers": 1,
    "phoneme_id_map": {
        "0": ["_"],
        "1": ["^"],
        "2": ["$"],
        "3": [" "]
    }
}
sh
python3 -m piper_train \
    --dataset-dir /path/to/training_dir/ \
    --accelerator 'gpu' \
    --devices 1 \
    --batch-size 32 \
    --validation-split 0.0 \
    --num-test-examples 0 \
    --max_epochs 10000 \
    --resume_from_checkpoint /path/to/lessac/epoch=2164-step=1355540.ckpt \
    --checkpoint-epochs 1 \
    --precision 32
sh
python3 -m piper_train.export_onnx \
    /path/to/model.ckpt \
    /path/to/model.onnx

cp /path/to/training_dir/config.json \
   /path/to/model.onnx.json
