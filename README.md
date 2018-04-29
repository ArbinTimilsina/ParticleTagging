# Particle Tagging

### Create a conda environment
```
conda update -n base conda
conda create --name envParticle python=2.7
conda activate envParticle
pip install --upgrade pip
pip install -r Requirements/Requirements.txt
```

### Switch Keras backend to TensorFlow
```
KERAS_BACKEND=tensorflow python -c "from keras import backend"
```

### Setup LarCV
```
git clone https://github.com/DeepLearnPhysics/larcv2
cd larcv2 && source configure.sh && make -j
```



