#! /bin/bash
pyinstaller --onefile ../src/MSDpostprocess/__main__.py -n MSDpostprocess --paths ../src/MSDpostprocess/
cp dist/MSDpostprocess release_stage/

python -m MSDpostprocess --options build_data/train_QE_Pro_model.toml
zip -r release_stage/QE_Pro_model.zip QE_Pro_model/

python -m MSDpostprocess --options build_data/train_TOF_model.toml
zip -r release_stage/TOF_model.zip TOF_model/

docker build --file Dockerfile ../ -t stavisvols/msdpostprocess
docker push stavisvols/msdpostprocess
