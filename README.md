# IAM Writer Recognition

IAM Dataset Writer Recognition Using a CNN

### Installation
  - [Install Anaconda or Miniconda](https://conda.io/docs/user-guide/install/macos.html)
  - Run ``conda env create -f environment.yml`` to install dependencies
  - Run ``source activate iam_writer_recognition`` to activate environment
  - Configure Jupyter to use the conda environment by running:
    - ``python -m ipykernel install --user --name=iam_writer_recognition``
  - Setup dataset as explained [here](./data/README.md)
  - Initialize Jupyter by running ``jupyter notebook`` in your terminal
  - In the Jupyter dashboard, navigate to [./src/solution.ipynb](./src/solution.ipynb) and execute the code


### How to run
- need to install anaconda
- in anaconda navigator you will need to install tensorflow and keras in environments under base(root)
- download iam database sentences.tgz and place it in the data folder
- extract it there
- the directory should look like this data/sentences/a01/a01-000u/a01-000u-s00-00.png
- 