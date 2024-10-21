#! /bin/bash
#clean up the build directory
rm -r release_stage
mkdir release_stage

#build linux executable version
pyinstaller --onefile ../src/lipoCLEAN/__main__.py -n LipoCLEAN --paths ../src/lipoCLEAN/ --add-data ../src/lipoCLEAN/*.txt:.
cp dist/LipoCLEAN release_stage/

#train MSD4_QE_Pro_model
python -m lipoCLEAN --options build_data/train_MSD4_QE_Pro_model.txt
zip -r release_stage/MSD4_QE_Pro_model.zip MSD4_QE_Pro_model/

#train MSD5_QE_Pro_model
python -m lipoCLEAN --options build_data/train_MSD5_QE_Pro_model.txt
zip -r release_stage/MSD5_QE_Pro_model.zip MSD5_QE_Pro_model/

#train MSD4_TOF_model
python -m lipoCLEAN --options build_data/train_MSD4_TOF_model.txt
zip -r release_stage/MSD4_TOF_model.zip MSD4_TOF_model/

#build docker version
docker build --file Dockerfile ../ -t stavisvols/lipoclean
docker push stavisvols/lipoclean

#make a zip file for an example analysis
mkdir tmp
cd tmp
ln ../build_data/QE_MTBLS5583_mzexport_* ./
ln ../example_analysis_options.txt ./
ln ../docker_example_analysis_options.txt ./
zip ../release_stage/example_analysis.zip *
cd ../
rm -r tmp

#download the windows executable from google drive
gdown --fuzzy https://drive.google.com/drive/folders/1zZ-H4xzIMFjrpgyAtAly2BIeRHeUXgCm --folder
cp lipoCLEAN/LipoCLEAN.exe release_stage/
