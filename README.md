

# LipoCLEAN: A machine learning based quality filter for lipid identifications from MS-DIAL
There are three ways to install and run LipoCLEAN: as an executable, as a Python package, and as a Docker container.

## The executable version
This method requires no installation but it is somewhat slower than the other options.
1. Download the executable for your operating system, the trained model, and the example analysis materials from the [releases page](https://github.com/stavis1/lipoCLEAN/releases).
2. Extract `example_analysis.zip` and `MSD4_QE_Pro_model.zip` into the same folder. There should be no folders nested under `MSD4_QE_Pro_model/` and all files from `example_analysis.zip` should be in the top level folder (the folder containing `MSD4_QE_Pro_model`).
3. Add `LipoCLEAN.exe` to this folder.
4. Open the folder with these files in a terminal such as `cmd.exe` or `bash`.
5. Run `LipoCLEAN.exe --options example_analysis_options.txt`. On some systems the warning `No module named 'brainpy._c.composition'` will be displayed. This is not an error and does not impact the running of the tool.
6. The results will be in a folder named `example_output/` the `example_output/QC/` folder contains several plots to assess the quality of the results.
7. If you want a default version of the options file run `LipoCLEAN.exe --print MSD4`.
8. To use the tool on other data edit the `options.txt` file.

## The Conda version
To set up the Conda environment for the tool:
1. Download the `Source code (zip)` from the [releases page](https://github.com/stavis1/lipoCLEAN/releases), extract the contents, and navigate to the `environments` directory in the extracted folder.
2. Open the directory in a terminal such as `cmd.exe` or `bash`.
3. Run `conda env create -p lipo_env --file lipoCLEAN.yml`. On some systems this may be slow.
4. Run `conda activate ./lipo_env`
5. Navigate to the repository root.
6. Run `pip install .`
7. Download the trained model, and the example analysis materials from the [releases page](https://github.com/stavis1/lipoCLEAN/releases).
8. Extract `example_analysis.zip` and `MSD4_QE_Pro_model.zip` into a single analysis folder. There should be no folders nested under `MSD4_QE_Pro_model/` and all files from `example_analysis.zip` should be in the top level of the analysis folder (the folder containing `MSD4_QE_Pro_model`).
9. Run `python -m lipoCLEAN --options example_analysis_options.txt` in the analysis folder. On some systems the warning `No module named 'brainpy._c.composition'` will be displayed. This is not an error and does not impact the running of the tool.
10. The results will be in a folder named `example_output/` the `example_output/QC/` folder contains several plots to assess the quality of the results.
11. If you want a default version of the options file run `python -m lipoCLEAN --print MSD4`
12. To use the tool on other data edit the `options.txt` file.

## The Docker version
The Docker container has trained models provided under `/models/`. To use these get the default options.txt from step 6:
1. Run `docker pull stavisvols/lipoclean`
2. Download the example analysis materials from the [releases page](https://github.com/stavis1/lipoCLEAN/releases).
3. Extract `example_analysis.zip`. All files from `example_analysis.zip` should be in the same folder.
4. Run `docker run --rm -v /path/to/your/data/:/data/ stavisvols/lipoclean python -m lipoCLEAN --options /data/docker_example_analysis_options.txt`
5. The results will be in a folder named `example_output/` the `example_output/QC/` folder contains several plots to assess the quality of the results.
6. If you want the default docker options file run `docker run --rm -v /path/to/your/data/:/data/ stavisvols/lipoclean python -m lipoCLEAN --print MSD4`
7. To use the tool on other data edit the `options.txt` file.

## MS-Dial export settings for inference
1. Click "Export" along the top bar
2. Select "Alignment result" in the dropdown menu
3. Navigate to the directory (folder) to which you want to save the export using the "Browse" button
4. The last alignment result selected should be listed as the export file. If this isn't the correct alignment, select the right one in the dropdown
5. Select "m/z matrix" to be exported (deselect any other exports you do not want to generate)
6. Make sure blank filtering is NOT selected
7. "Export format" should be "msp"
8. Click Export

A .txt will now be generated in the chosen directory with the information required for LipoCLEAN. The file name will start with "Mz"

## If you wish to prepare your own training data
1. Start with MS-DIAL exports using the same settings as described above for inference. 
2. Add a column named `label` which contains 0 for incorrect IDs, 1 for correct IDs, and is otherwise left blank. It is critical that this column be before (to the left of) the `MS/MS spectrum` column as all subsequent columns (those to the right) are assumed to be m/z data. 
3. Save in a tab-delimited format.

The tool is capable of being trained on multiple input files. The retention time correction is run on a per-input-file basis. Multiple experiments can be used to generate training data, but it is suggested that they are input as separate files for chromatography alignment purposes. 

## The datasets we used for training
| Instrument | Source | N  | Model | Organism |
| :--------- | :----- | :- | :---- | :------- |
| Q-Exactive | [MTBLS5583](https://www.ebi.ac.uk/metabolights/editor/MTBLS5583/descriptors) | 742 | QE_Pro_model | *Canis familiaris* |
| LTQ Velos Pro | in-house | 1076 | QE_Pro_model | *Aspergillus fumigatus* |
| LTQ Velos Pro | in-house | 545 | QE_Pro_model | *Laccaria bicolor* |
| TripleTOF 6600 | [MTBLS4108](https://www.ebi.ac.uk/metabolights/editor/MTBLS4108/descriptors) | 1125 | TOF_model | *Rattus norvegicus* |

Our tests have shown that a model will likely generalize to a family of instruments but that this has limits. We expect that the QE_Pro_model will work for all Orbitrap systems. We do not have the data necessary to know how well the TOF model will generalize to all TOF instruments so if you are working with e.g. TimsTOF data it would be a good idea to do an initial validation of the output. The publicly available datasets used were reprocessed from raw files and annotated in-house.

## Other information
Our tool supports both MS-DIAL 4 and 5. However, some columns were renamed and scaled differently between the two versions so a model trained on one version's data will not work with the other. Both the options file and the model are specific to a version of MS-DIAL. We provide separate default options files for each version that can be obtained with the `--print MSD4` and `--print MSD5` command line arguments, respectively.

We have tested the tool on Windows 10 and Ubuntu 22.04. Although we have not tested this, we expect that the Docker version will work on Macs with intel chips and that the Conda version should work on any machine that has Conda installed.

The `tests` and `build` directories in this repository are intended for internal development use only and the scripts they contain are not expected to work on other systems.

 If you wish to compile the executable version yourself:
 1. Create and activate a Conda environment using `environments/build.yml`.
 2. Run `pip install .` at the repository root.
 3. Navigate to `build/`.
 4. Run `pyinstaller --onefile ../src/lipoCLEAN/__main__.py -n LipoCLEAN --paths ../src/lipoCLEAN/ --add-data ../src/lipoCLEAN/*.txt:.`
 5. The executable will be found in a newly created `dist/` directory

Disclaimer: We are not in any way associated with the developers of MS-DIAL, we are merely enthusiastic users of their software.

