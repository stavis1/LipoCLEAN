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

## Export settings for inference
MATTHEW WRITE THIS BIT

## Prepare training data
1. Start with MS-Dial exports using the same settings as described above for running the inference script. 
2. Delete the metadata rows so that the column headers are now the first row of the document. 
3. Add a column named `label` which contains 0 for incorrect IDs and 1 for correct IDs. It is critical that this column be before (to the left of) the `MS/MS spectrum` column as all subsequent columns (those to the right) are assumed to be m/z data. 
4. Remove all entries that you do not manually label.
5. Save in a tab-delimited format.

The training script is capable of being trained on multiple input files. The retention time correction is run on a per-input-file basis so all of the entries in each file should have been run with the same chromatography. Multiple experiments can be used to generate training data, but it is suggested that they are input as separate files for chromatography alignment purposes. 
