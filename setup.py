from setuptools import find_packages, setup

setup(
    name="mmpe",
    version="1.0.1",
    packages=find_packages(),
)

# mim download mmpose --config \
# rtmpose-l_8xb32-270e_coco-wholebody-384x288 --dest models
# mim download mmdet --config rtmdet_m_8xb32-300e_coco --dest models
