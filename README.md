# Particle Tagging

### Git clone the code
```
git clone https://github.com/ArbinTimilsina/ParticleTagging.git
cd ParticleTagging
```

### Create a conda environment
```
conda update -n base conda
conda create --name envParticle python=2.7
conda activate envParticle
```
```
pip install --upgrade pip
pip install -r Requirements/Requirements.txt
```

### Setup LArCV
```
git clone https://github.com/DeepLearnPhysics/larcv2
cd larcv2 && source configure.sh && make && cd ..
```

### Switch Keras backend to TensorFlow
```
KERAS_BACKEND=tensorflow python -c "from keras import backend"
```

### Create an IPython kernel for the environment
```
python -m ipykernel install --user --name envParticle --display-name "envParticle"
```

### Download small datasets (for development)
```
wget http://www.stanford.edu/~kterao/public_data/v0.1.0/2d/classification/five_particles/practice_train_5k.root -O InputFiles/classification_train_5k.root
wget http://www.stanford.edu/~kterao/public_data/v0.1.0/2d/classification/five_particles/practice_test_5k.root -O InputFiles/classification_test_5k.root
wget http://www.stanford.edu/~kterao/public_data/v0.1.0/2d/segmentation/multipvtx/practice_train_2k.root -O InputFiles/segmentation_train_2k.root
wget http://www.stanford.edu/~kterao/public_data/v0.1.0/2d/segmentation/multipvtx/practice_test_2k.root -O InputFiles/segmentation_test_2k.root
```

### Download large datasets (for training/testing)
```
wget http://www.stanford.edu/~kterao/public_data/v0.1.0/2d/classification/five_particles/train_50k.root -O InputFiles/classification_train_50k.root
wget http://www.stanford.edu/~kterao/public_data/v0.1.0/2d/classification/five_particles/test_40k.root -O InputFiles/classification_test_40k.root
wget http://www.stanford.edu/~kterao/public_data/v0.1.0/2d/segmentation/multipvtx/train_15k.root -O InputFiles/segmentation_train_15k.root
wget http://www.stanford.edu/~kterao/public_data/v0.1.0/2d/segmentation/multipvtx/test_10k.root -O InputFiles/segmentation_test_10k.root
```

## If running remotely
### Install emacs
```
sudo apt install emacs
```

###  Run the Jupyter notebook: First generate config file and then change the IP address config setting
```
jupyter notebook --generate-config
sed -ie "s/#c.NotebookApp.ip = 'localhost'/#c.NotebookApp.ip = '*'/g" ~/.jupyter/jupyter_notebook_config.py
jupyter notebook --ip=0.0.0.0 --no-browser
```

### Remove conda environment (if desired)
```
conda deactivate
conda remove --name envParticle --all
```