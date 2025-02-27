CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
#git submodule update --init --recursive
# sudo add-apt-repository ppa:deadsnakes/ppa
# sudo apt update
# sudo apt install -y python3-pip python3.7 python3.7-dev python3.7-distutils cmake g++-9
# sudo pip3 install -U virtualenv==15.2.*
# sudo python3 -m pip install mininet numpy==2.1.3 matplotlib==3.9.2 pandas==2.2.3 scienceplots


# echo "Setting up pcc vivace kernel"
# cd $CURRENT_DIR/CC/PCC-Kernel/src && make


# echo "Downloading and setting up Orca"

# cd $CURRENT_DIR/CC/Orca
# bash build.sh


echo "Downloading and setting up astraea"

python3.7 -m pip install pip --upgrade
python3.7 -m pip install protobuf==3.10.0 tensorflow==1.14.0 --upgrade
python3.7 -m pip install matplotlib==3.2
python3.7 -m pip install numpy==1.20.0
python3.7 -m pip install --upgrade --force-reinstall pillow
cd $CURRENT_DIR/CC/astraea-open-source/kernel/tcp-astraea
make
cd ../..
bash build.sh
