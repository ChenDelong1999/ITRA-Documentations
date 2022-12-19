
# Install Dependencies

- Create a conda environment and install PyTorch:

    ```bash
    conda create -n ITRA python=3.10.0
    conda activate ITRA
    ```

    This repo requires PyTorch (1.12) and torchvision (0.13). Please install them via [pytorch official website](https://pytorch.org/get-started/locally).

    ```bash
    conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=10.2 -c pytorch
    ```

- Clone this repo:

   ```bash
  # TODO: update repo name
   git clone https://github.com/megvii-research/protoclip
   cd protoclip
   export PYTHONPATH="$PYTHONPATH:$PWD/src"
   ```
   **Note**: If import error is occurred later, run `export PYTHONPATH="$PYTHONPATH:$PWD/src"` again.

- Install additional dependencies:
    ```bash
    conda install pillow pandas scikit-learn faiss-gpu ftfy tqdm matplotlib pycocotools wandb
    conda install -c huggingface transformers 
    conda install -c conda-forge sentence-transformers
    pip install adapter-transformers open_clip_torch
  
  pip install clip-benchmark
    
    # ELEVATOR requirements  
    pip install yacs timm git+https://github.com/haotian-liu/CLIP_vlp.git vision-evaluation
    
    # TODO: remove nori dependency
    pip install nori2
    ```
  