# Model Weights

Pre-trained weights for the `Generator2D` wind field model are **not distributed
in this repository** due to file size and ongoing research use.

## How to obtain

Request the weights by emailing:

**fabian.alexis.hernandez@upc.edu**

Subject: `[FLICK] Model weights request`

Please include a brief description of your intended use.

## What you will receive

A `.pt` checkpoint file (PyTorch, ~50 MB) compatible with `Generator2D` in
`flick_urban/nn/models.py`. The file contains:

```python
{
    'model_state_dict': {...}   # Generator2D weights
}
```

## Usage

Place the checkpoint in any directory and pass the path at inference time:

```bash
python -m flick_urban.nn.inference \
    -dataset_base_path ./output/ \
    -data_sample_basename grid_of_cubes \
    -model_loading_path ./model_weights/ \
    -model_basename generator
```

Or in Python:

```python
from flick_urban.nn.models import Generator2D
from flick_urban.nn.inference import get_args, load_model

args = get_args([])
args.model_loading_path = './model_weights/'
args.model_basename     = 'generator'

model = Generator2D(args)
model = load_model(model, args)
```

## Training details

| Property        | Value                        |
|-----------------|------------------------------|
| Architecture    | Generator2D (ResNet-style)   |
| Input features  | MASK, HEGT, WDST             |
| Output features | U, V                         |
| Input size      | 256 × 256                    |
| Training data   | LES simulations (SOD2D)      |
| Error on unseen | ~40 % (geometry shift)       |

See the main [README](../README.md) and the references therein for details on
training methodology and validation.
