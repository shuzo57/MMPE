# MMPE (mmpose estimation)
## Installation
```Bash
git clone git@github.com:shuzo57/MMPE.git
cd MMPE
pip install -e .
```

## Download pretrained model (example)
```Bash
mim download mmpose --config rtmpose-l_8xb32-270e_coco-wholebody-384x288 --dest models
mim download mmdet --config rtmdet_m_8xb32-300e_coco --dest models
```