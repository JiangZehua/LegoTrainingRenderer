python -m pip install -r requirements.txt
python -m pip install -e .

# for M1 mac
conda install pytorch torchvision torchaudio -c pytorch

# Installing hydra:
python -m pip install --upgrade hydra-core                 
# store hyperparam sweeps in separate files (in "experiments") will only work with hydra 1.2.0, but not 100% sure.
python -m pip install --upgrade hydra-submitit-launcher