#! /bin/bash
pyinstaller --onefile ../src/MSDpostprocess/__main__.py -n MSDpostprocess --paths ../src/MSDpostprocess/ --add-data ../src/MSDpostprocess/example_options.toml:.
cp dist/MSDpostprocess release_stage/

python -m MSDpostprocess --options build_data/train_QE_Pro_model.toml
zip -r release_stage/QE_Pro_model.zip QE_Pro_model/

python -m MSDpostprocess --options build_data/train_TOF_model.toml
zip -r release_stage/TOF_model.zip TOF_model/

docker build --file Dockerfile ../ -t stavisvols/msdpostprocess
docker push stavisvols/msdpostprocess

zip release_stage/example_analysis.zip example_analysis_options.toml ../../data/current_datasets/training_files_with_annotations/QE_MTBLS5583_mzexport_*

gdown --fuzzy https://drive.google.com/drive/folders/1zZ-H4xzIMFjrpgyAtAly2BIeRHeUXgCm --folder
cp MSDpostprocess/MSDpostprocess.exe release_stage/
