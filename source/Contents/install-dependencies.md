
# Install Dependencies

- Create a conda environment and install PyTorch:

    ```bash
    conda create -n ITRA python=3.10.0
    conda activate ITRA
    ```

    This repo requirs PyTorch (1.11.0) and torchvision. Please install them via https://pytorch.org/get-started/locally

    ```
    conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=10.2 -c pytorch -y
    # conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=10.2 -c pytorch
    ```

- Clone this repo:

   ```bash
   git clone https://github.com/megvii-research/protoclip
   cd protoclip
   export PYTHONPATH="$PYTHONPATH:$PWD/src"
   ```
   **Note**: If import error is occured later, run `export PYTHONPATH="$PYTHONPATH:$PWD/src"` again.

- Install additional dependencies:
    ```bash
    conda install pillow pandas scikit-learn faiss-gpu ftfy tqdm matplotlib pycocotools wandb
    conda install -c huggingface transformers 
    conda install -c conda-forge sentence-transformers
    pip install adapter-transformers
    # TODO: remove nori dependency
    pip install nori2
    ```
- ELEVATOR dependencies

    ```
    pip install yacs timm git+https://github.com/haotian-liu/CLIP_vlp.git vision-evaluation

    yacs~=0.1.8
    scikit-learn
    timm~=0.4.12
    numpy~=1.21.0
    sharedmem
    git+https://github.com/openai/CLIP.git
    git+https://github.com/haotian-liu/CLIP_vlp.git
    torch~=1.7.0
    PyYAML~=5.4.1
    Pillow~=9.0.1
    torchvision~=0.8.0
    vision-evaluation>=0.2.2
    vision-datasets>=0.2.0
    tqdm~=4.62.3
    transformers~=4.11.3
    protobuf~=3.20.1
    ftfy~=6.1.1
    nltk~=3.7
    ```**
    
