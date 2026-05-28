apt-get update

apt-get install -y wget unzip git
DEBIAN_FRONTEND=noninteractive apt-get install -y libgl1 libglib2.0-0
pip install autopep8 pylint gdown ipykernel
pip install scikit-learn
pip install pillow
pip install matplotlib
pip install importlib-resources
pip install numba
pip install lpips
pip install thop
pip install tensorflow-datasets 
pip install tensorflow-probability
pip install pyyaml