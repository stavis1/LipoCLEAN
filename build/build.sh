#! /bin/bash
pyinstaller --onefile ../src/lipoCLEAN/__main__.py -n LipoCLEAN --paths ../src/lipoCLEAN/ --add-data ../src/lipoCLEAN/example_options.txt:.
cp dist/LipoCLEAN release_stage/

python -m lipoCLEAN --options build_data/train_QE_Pro_model.txt
zip -r release_stage/QE_Pro_model.zip QE_Pro_model/

python -m lipoCLEAN --options build_data/train_TOF_model.txt
zip -r release_stage/TOF_model.zip TOF_model/

docker build --file Dockerfile ../ -t stavisvols/lipoclean
docker push stavisvols/lipoclean

zip release_stage/example_analysis.zip example_analysis_options.txt ../../data/current_datasets/training_files_with_annotations/QE_MTBLS5583_mzexport_*

gdown --fuzzy https://drive.google.com/drive/folders/1zZ-H4xzIMFjrpgyAtAly2BIeRHeUXgCm --folder
cp lipoCLEAN/LipoCLEAN.exe release_stage/
