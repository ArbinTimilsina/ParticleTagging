# Particle Tagging

### Git clone the source code
```
git clone https://github.com/ArbinTimilsina/ParticleTagging.git
cd ParticleTagging
```

### Create a conda environment
```
conda update -n base conda
conda create --name envParticle python=2.7
conda activate envParticle
pip install --upgrade pip
pip install -r Requirements/Requirements.txt
```
### Setup LarCV
```
git clone https://github.com/DeepLearnPhysics/larcv2
cd larcv2 && source configure.sh && make && cd ..
```

### Switch Keras backend to TensorFlow
```
KERAS_BACKEND=tensorflow python -c "from keras import backend"
```


### Remove conda environment (if desired)
```
conda deactivate
conda remove --name  envParticle --all
```