# Light Transport Gaussian: Relighting 3D Gaussians Based on Precomputed Radiance Transfer
Code for Project "**Light Transport Gaussian**"

### [Project Page](https://gerwang.github.io/shadowneus/) | [Paper](https://arxiv.org/abs/2211.14086) | [Video](https://www.youtube.com/watch?v=jvxJ7bVuTBk) | [Dataset](https://drive.google.com/drive/folders/1Sr30kdvCD2tXNAONzcnF5xnoMXasylyA?usp=sharing)

## Usage

### Setup

```bash
git clone git@github.com:zhanglbthu/Light-Transport-Gaussian.git
cd Light-Transport-Gaussian
conda create -n LTG python=3.7
conda activate LTG
pip install -r requirements.txt
```

### Dataset
We train the model using a dataset in the form of **LIGHT STAGE**, where the information about the camera and light source is known, specifically, the object is NeRF synthetic data and the light source is directional light.
You can download the generated dataset form [here](https://drive.google.com/drive/folders/1Sr30kdvCD2tXNAONzcnF5xnoMXasylyA?usp=sharing).
### Preprocess your own dataset
You can also preprocess your own dataset by following the steps below:

### Train

### Test