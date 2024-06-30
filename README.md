# Canny 3D Edge Detector

This repository contains a Canny edge detector for 3D medical images,
implemented as **Analysis of computed tomography (CT) images** assignment for the
*Biomedical Signal and Image Processing* course at the master's study at the
Faculty of Computer and Information Science Ljubljana in 2023/24. Additionally
to the Canny detector, the implementation contains 3D edge linking to extend the
detector to 3D images.

![Edge detection script animation](animation.gif)

## Environment

The project was tested on Python 3.8.18 using the packages listed in
`requirements.txt`. To create a virtual environment and install the required
packages, run:

```bash
python --version
> Python 3.8.18
python -m venv venv
. venv/bin/activate
pip install -r requirements.txt
```

## Usage

Script `edge_detector.py` takes multiple arguments, the path to the images and
at least one or multiple names of image files. For help and examples, run the
script without arguments. Script `run.sh` makes it more convenient to run the
detector on multiple images.

Script `visualize.py` visualizes the detected edges in `results` on a 3D plot.
