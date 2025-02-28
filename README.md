# wheel-py
POC computer vision project for popular car game

## Note against violating the EULA
This project is designed for **Education Purposes Only**, i do not condome, nor encourage anyone to violate the **Hill Climb Racing 2 EULA**, and i will not take any responsibility for *your account getting banned, or revoked*.

## Overview
Wheel-py is a tool designed as a **Proof Of Concept**.  
The goal is to completely automate **Hill Climb Racing 2** android app, using YOLO, and computer vision.  

## Installation
Considering this project started private, and it is in early stages of development, there might be problems, depending on your computer, architecture, or usage case.  
I strongly encourage, to use an NVIDIA card and also install proprietary drivers, including CUDA.  

### Windows
Install anaconda on [https://www.anaconda.com/download/success](url)  
Install platform-tools on [https://developer.android.com/tools/releases/platform-tools](url), then add platform-tools to PATH  
Open conda prompt using search bar, or add conda to path **(Not recommended!)**  
```
conda create -n hcr2-wheelpy python=3.10.13
```
Then either open the project in your desired IDE, and select python interpeter as anaconda(hcr2-wheelpy), or:  
```
conda activate hcr2-wheelpy
pip install -r requirements.txt
```

### Linux
Run nvidia-smi to check if your drivers are properly conigured, if not, install them(Depending on your linux distribution)  
I'm assuming you have *pyenv*, and *pyenv-venv* installed and configured, if not, check the installation guide depending on your linux distribution  
Install adb-platform-tools, also check for installation depending on your distribution.  
```
pyenv install 3.10.13
pyenv virtualenv 3.10.13 hcr2-wheelpy
pyenv activate hcr2-wheelpy
pip install -r requirements.txt
```
If you desire to activate hcr2-wheelpy env every time you go into the project folder, cd into wheel-py, then use:
```
pyenv local hcr2-wheelpy
```
