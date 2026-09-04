export HOME=$_CONDOR_JOB_IWD
export USE_USER=abal
export QML=/work/abal/qae_hep

cd $HOME
mkdir qae_hep
cd qae_hep
scp portal1@etp.kit.edu:/work/abal/qae_hep/*.py .
scp -r portal1@etp.kit.edu:/work/abal/qae_hep/helpers .

mkdir -p data/qcd_sqrtshatTeV_13TeV_PU40_NEW_EXT_sideband_parts/{train,test,library}
scp abal@portal1.etp.kit.edu:/storage/9/abal/CASE/delphes/qcd_sqrtshatTeV_13TeV_PU40_NEW_EXT_sideband_parts/train/qcd_sqrtshatTeV_13TeV_PU40_NEW_EXT_sideband_000.h5 data/qcd_sqrtshatTeV_13TeV_PU40_NEW_EXT_sideband_parts/train/
scp abal@portal1.etp.kit.edu:/storage/9/abal/CASE/delphes/qcd_sqrtshatTeV_13TeV_PU40_NEW_EXT_sideband_parts/train/qcd_sqrtshatTeV_13TeV_PU40_NEW_EXT_sideband_001.h5 data/qcd_sqrtshatTeV_13TeV_PU40_NEW_EXT_sideband_parts/train/
scp abal@portal1.etp.kit.edu:/storage/9/abal/CASE/delphes/qcd_sqrtshatTeV_13TeV_PU40_NEW_EXT_sideband_parts/train/qcd_sqrtshatTeV_13TeV_PU40_NEW_EXT_sideband_002.h5 data/qcd_sqrtshatTeV_13TeV_PU40_NEW_EXT_sideband_parts/test/
scp abal@portal1.etp.kit.edu:/storage/9/abal/CASE/delphes/qcd_sqrtshatTeV_13TeV_PU40_NEW_EXT_sideband_parts/train/qcd_sqrtshatTeV_13TeV_PU40_NEW_EXT_sideband_003.h5 data/qcd_sqrtshatTeV_13TeV_PU40_NEW_EXT_sideband_parts/test/

sed -i -e 's|/storage/9/abal/CASE/delphes/|data/|g' helpers/path_setter.py
sed -i -e 's|/work/abal/qae_hep/saved_models/|'"$HOME"'/saved_models/|g' helpers/path_setter.py

mkdir -p $HOME/saved_models

python3 case_qml.py --train --wires 9 \
--trash-qubits 6 -b 200 -e 15 --backend autograd --save --seed ${seed} --lr 0.005 --desc "Using arbitrary 3D rotations and 3x weights" \
--train_n 250000 --valid_n 20000 --n_threads $NUM --device lightning.gpu