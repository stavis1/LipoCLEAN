#! /bin/bash
#build linux executable version
pyinstaller --onefile ../src/lipoCLEAN/__main__.py -n LipoCLEAN --paths ../src/lipoCLEAN/ --add-data ../src/lipoCLEAN/example_options.txt:.
cp dist/LipoCLEAN release_stage/

#train QE_Pro_model
python -m lipoCLEAN --options build_data/train_QE_Pro_model.txt
zip -r release_stage/QE_Pro_model.zip QE_Pro_model/

#train TOF model
python -m lipoCLEAN --options build_data/train_TOF_model.txt
zip -r release_stage/TOF_model.zip TOF_model/

#build docker version
docker build --file Dockerfile ../ -t stavisvols/lipoclean
docker push stavisvols/lipoclean

#make a zip file for an example analysis
mkdir tmp
cd tmp
ln ../../../data/current_datasets/training_files_with_annotations/QE_MTBLS5583_mzexport_* ./
ln ../example_analysis_options.txt ./
zip ../release_stage/example_analysis.zip *
cd ../
rm -r tmp

#download the windows executable from google drive
gdown --fuzzy https://drive.google.com/drive/folders/1zZ-H4xzIMFjrpgyAtAly2BIeRHeUXgCm --folder
cp lipoCLEAN/LipoCLEAN.exe release_stage/
