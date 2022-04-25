# Temporal Fusion Transformer in Pytorch

Code took and readjusted from [**mattsherar**/**Temporal_Fusion_Transform**](https://github.com/mattsherar/Temporal_Fusion_Transform).

## Quickstart

Clone the repo

    git clone https://github.com/paolodelia99/Temporal-Fusion-Transfomer-pytorch.git

The model is present in the `tft.model` module while the `TSDataset` is present the `tft.utils` module.
For use the model and the dataset adapter in your notebook you can simply copy the following lines:

```python
from tft.model import TFT, QuantileLoss
from tft.utils import TSDataset
```


## Example Notebooks

- [Italian Electricity Price forecasting](notebooks/electricity-prices-forecasting.ipynb)

## Author 

Paolo D'Elia
