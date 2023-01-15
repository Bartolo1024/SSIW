CMP dataset finetuning based on repository contains the source code of the paper:
[Wei Yin, Yifan Liu, Chunhua Shen, Anton van den Hengel, Baichuan Sun, The devil is in the labels: Semantic segmentation from sentences](https://arxiv.org/abs/2202.02002)

# Run training

```shell
/bin/bash download.sh
conda env create -f environment.yaml
conda activate ssiw
```
```python
python -m src.tools.train --max-epochs 100
```

# Test

* train your checkpoint (by default saved as out.pth)
* use bartlo1024 checkpoint [link]

```python
python -m src.tools.test_cpm data/base/base/cmp_b0346.jpg --checkpoint-path out.pth
```