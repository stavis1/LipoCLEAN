# MSDpostprocess: A machine learning based quality filter for lipid identifications from MS-Dial

## Set up a local conda environment
MSDPostprocess consists of a pair of python scripts, one for training and one for inference. The dependencies for these scripts are laid out in environments/MSDpostprocess.yml.
To set up the conda environment for these scripts:
1. Install conda, miniconda, or mamba. Instructions can be found here https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html
2. Download the MSDpostprocess repository and navigate to the `environments` directory.
3. Run `conda env create -p msdp_env --file MSDpostprocess.yml`
4. Run `conda activate ./msdp_env`

## Example analysis
1. Navigate to the `example_data` directory
2. Run `python ../scripts/MSDpostprocess-training.py --input training_data.tsv --min_rt 7 --out_dir ./`
3. Run `python ../scripts/MSDpostprocess-inference.py --input example_output_negative.txt,example_output_positive.txt --min_rt 7 --model model.dill --plots --out_dir ./`


