# MSDpostprocess: A machine learning based quality filter for lipid identifications from MS-DIAL
There are three ways to install and run MSDpostprocess: as an executable, as a docker container, and as a python package.

## The executable version
This method reqires no installation but it is somewhat slower than the other options.
1. Download the executable for your operating system, the trained model, and the example options file from the [releases page](https://github.com/stavis1/MSDpostprocess/releases).
2. Run `MSDpostprocess.exe --print options.toml` to get a default options file.
3. Edit the options file for your experiment.
4. Run `MSDpostprocess.exe --options options.toml`

## The conda/virtualenv version
To set up the conda environment for the tool:
1. Download the MSDpostprocess repository and navigate to the `environments` directory.
2. Run `conda env create -p msdp_env --file MSDpostprocess.yml`
3. Run `conda activate ./msdp_env`
4. Navigate to the repository root.
5. Run `pip install -e .`

Otherwise, if you are using virtualenv:
1. Navigate to the repository root.
2. Run `virtualenv environments/msdp_env`
3. Activate the environment
4. Run `pip install -e .`

To use the tool with either method:
1. Download the trained models from the [releases page](https://github.com/stavis1/MSDpostprocess/releases).
2. Run `python -m MSDpostprocess --print options.toml` to get a default options file.
3. Edit the options file for your experiment.
4. Run `python -m MSDpostprocess --options options.toml`

## The Docker version
The docker container has trained models provided under /models/. To use these get the default options.toml from step 1 below:
1. Run `docker pull stavisvols/msdpostprocess`
2. Run `docker run --rm -v /path/to/your/data/:/data/ stavisvols/msdpostprocess python -m MSDpostprocess --print /data/options.toml`
3. Edit the options file for your experiment.
4. Run `docker run --rm -v /path/to/your/data/:/data/ stavisvols/msdpostprocess python -m MSDpostprocess --options /data/options.toml`
The working directory will be within the docker container's filesystem.

## Example analysis
1. Install the tool using one of the above methods.
2. Download `QE_Pro_model.zip` and `example_analysis.zip` from the releases page.
3. Extract both archives. There should be no folders nested under `QE_Pro_model/` and `example_analysis/`.
4. Run `MSDpostprocess.exe --options example_analysis/example_analysis_options.toml`

On some systems the warning `No module named 'brainpy._c.composition'` will be displayed. This is not an error and does not impact the running of the tool.

## MS-Dial export settings for inference
1. Click "Export" along the top bar
2. Select "Alignment result" in the dropdown menu
3. Navigate to the directory (folder) to which you want to save the export using the "Browse" button
4. The last alignment result selected should be listed as the export file. If this isn't the correct alignment, select the right one in the dropdown
5. Select "m/z matrix" to be exported (deselect any other exports you do not want to generate)
6. Make sure blank filtering is NOT selected
7. "Export format" should be "msp"
8. Click Export

A .txt will now be generated in the chosen directory with the information required for MSDpostprocess. The file name will start with "Mz"

## Prepare training data
1. Start with MS-DIAL exports using the same settings as described above for inference. 
2. Add a column named `label` which contains 0 for incorrect IDs, 1 for correct IDs, and is otherwise left blank. It is critical that this column be before (to the left of) the `MS/MS spectrum` column as all subsequent columns (those to the right) are assumed to be m/z data. 
3. Save in a tab-delimited format.

The tool is capable of being trained on multiple input files. The retention time correction is run on a per-input-file basis. Multiple experiments can be used to generate training data, but it is suggested that they are input as separate files for chromatography alignment purposes. 

## The datasets used for training.
| Instrument | Source | N  | Model | Organism |
| :--------- | :----- | :- | :---- | :------- |
| Q-Exactive | [MTBLS5583](https://www.ebi.ac.uk/metabolights/editor/MTBLS5583/descriptors) | 742 | QE_Pro_model | *Canis familiaris* |
| LTQ Velos Pro | in-house | 1076 | QE_Pro_model | *Aspergillus fumigatus* |
| LTQ Velos Pro | in-house | 545 | QE_Pro_model | *Laccaria bicolor* |
| TripleTOF 6600 | [MTBLS4108](https://www.ebi.ac.uk/metabolights/editor/MTBLS4108/descriptors) | 1125 | TOF_model | *Rattus rattus* |

Our tests have shown that a model will likely generalize to a family of instruments but that this has limits. We expect that the QE_Pro_model will work for all orbitrap systems. We do not have the data necessary to know how well the TOF model will generalize to all TOF instruments so if you are working with e.g. TimsTOF data it would be a good idea to do an initial validation of the output. The publicly available datasets used were reprocessed from raw files and annotated in-house.

