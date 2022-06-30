#implementation similar to http://www.bioinf.jku.at/research/lsc/
#with the following copyright information:
#Copyright (C) 2018 Andreas Mayr
#Licensed under GNU General Public License v3.0 (see http://www.bioinf.jku.at/research/lsc/LICENSE and https://github.com/ml-jku/lsc/blob/master/LICENSE)
#!/bin/bash

workdir=$(cd $(dirname $0); pwd)
ln -sf $workdir/NPTP_external/lsc/python_code/* $workdir/python_code/
ln -sf $workdir/NPTP_external/lsc/cluster $workdir/
mkdir -p ./programs/lib
mkdir ./programs/source
mkdir result

svn export svn://svn.code.sf.net/p/jcompoundmapper/code/ ./programs/jcompoundmapper-code-r55 -r 55
sed -i "59isubstructureHash=false;" programs/jcompoundmapper-code-r55/src/de/zbit/jcmapper/fingerprinters/topological/features/ECFPFeature.java
sed -i "36inewHash=seed;" programs/jcompoundmapper-code-r55/src/de/zbit/jcmapper/io/writer/ExporterHelper.java
ant -buildfile programs/jcompoundmapper-code-r55/build.xml all

wget https://sourceforge.net/projects/jcompoundmapper/files/jCMapperCLI.jar --directory=./programs

wget https://bitbucket.org/jskDr/pcfp/get/39eb310c1b95.zip --directory=./programs
unzip ./programs/39eb310c1b95.zip -d ./programs
ant -buildfile ./programs/jskDr-pcfp-39eb310c1b95/build.xml


wget https://src.fedoraproject.org/repo/pkgs/libconfig/libconfig-1.5.tar.gz/a939c4990d74e6fc1ee62be05716f633/libconfig-1.5.tar.gz --directory-prefix=./programs/source
wget http://dlib.net/files/dlib-19.0.zip --directory-prefix=./programs/source


tar -xzvf ./programs/source/libconfig-1.5.tar.gz --directory=./programs/source
unzip ./programs/source/dlib-19.0.zip -d ./programs/source


mv ./programs/source/dlib-19.0 ./programs/lib/dlib-19.0



mkdir $workdir/programs/lib/libconfig

cd $workdir/programs/source/libconfig-1.5/ 

./configure --prefix=$workdir/programs/lib/libconfig

make -C $workdir/programs/source/libconfig-1.5
make -C $workdir/programs/source/libconfig-1.5 install

#environment variable
export CPATH=$workdir/programs/lib/dlib-19.0:$CPATH
export CPATH=$workdir/programs/lib/libconfig/include:$CPATH
export LIBRARY_PATH=$workdir/programs/lib/libconfig/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=$workdir/programs/lib/libconfig/lib:$LD_LIBRARY_PATH
export PYTHONPATH=$workdir/NPTP_external:$PYTHONPATH
#change into cpp pipeline directory
make -C $workdir/cluster/exec all multiproc=1


wget https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/chembl_29/chembl_29.sdf.gz --directory=$workdir/knime-workspace/TheData/
gunzip $workdir/knime-workspace/TheData/chembl_29.sdf.gz

# download the models https://doi.org/10.5281/zenodo.6782271
cd $workdir
wget --continue https://zenodo.org/record/6782271/files/model.tar.gz
tar xvf model.tar.gz
