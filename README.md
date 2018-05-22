FashionAI.KPS
=============

My solution to [FashionAI](https://tianchi.aliyun.com/markets/tianchi/FashionAIeng?_lang=en_US) Key Points Detection of Apparel.

### Data

download data and put in `data`

```
data
├── r1-test-a
│   ├── Images
│   └── test.csv
├── r1-test-b
│   ├── Images
│   └── test.csv
├── r2-test-a
│   ├── Images
│   └── test.csv
├── r2-test-b
│   ├── Images
│   └── test.csv
└── train
    ├── Annotations
    └── Images
```

### Notice

There's a bug in `mxnet/gluon/block.py` in `SymbolBlock`. If you encounter save and load model issue, check a patch below.

```python
class SymbolBlock(HybridBlock)
    def __init__(self, outputs, inputs, params=None):
        super(SymbolBlock, self).__init__(prefix=None, params=None)
        self._prefix = ''
        self._params = ParameterDict('', params)
        self._reg_params = self._params  # <--- add this line
```
