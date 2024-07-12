# Weekly Report 3

## So sánh performance với learning_rate và epochs khác nhau
|    |4000|8000|16000|
|----|----|----|-----|
|0.1 |train: 0.523, test: 0.537|train: 0.311, test: 0.312||
|0.05|train: 0.922, test: 0.925|train: 0.933, test: 0.934|train: 0.955, test: 0.947|
|0.01|train: 0.893, test: 0.896|train: 0.901, test: 0.908|train: 0.899, test: 0.889|
|0.005|train: 0.860, test: 0.827|train: 0.892, test: 0.868|train: 0.885, test: 0.853|
|0.001|train: 0.772, test: 0.632|train: 0.772, test: 0.771|train: 0.869, test: 0.813|
|0.0001|train: 0.321, test: 0.391|train: 0.565, test: 0.567|train: 0.706, test: 0.726|
|0.00001|train: 0.119, test: 0.110|||

## So sánh performance với layer và nodes khác nhau

Learning rate = 0.01, batch_size = 32, epochs = 16000

No Layer: train: 0.895, test: 0.897

128, ReLU: train: 0.888, test: 0.907

256, ReLU: train: 0.887, test: 0.897

512, ReLU: train: 0.906, test: 0.912

256, ReLU; 64, ReLU: train: 0.112, test: 0.113

256, LeakyReLU; 64, LeakyReLU: train: 0.950, test: 0.936

## So sánh perfomance với batch size khác nhau

epochs = 4000, layer = 256: ReLU, learning rate = 0.001

|batch size|train|test|
|---|-----|----|
|   8|0.542|0.637|
|  16|0.591|0.718|
|  64|0.655|0.725|
| 128|0.757|0.799|
| 256|0.752|0.699|