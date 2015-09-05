virtualenv env --no-site-packages
source env/bin/activate
pip install -e .
export DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:$(pwd)/sdk/myo.framework
