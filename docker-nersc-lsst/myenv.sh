#! /bin/bash
#echo "hello from /etc/profile.d/myenv.sh"
#which conda

# equivalent of the stuff that "conda init" adds to ~/.bashrc:
. /opt/conda/etc/profile.d/conda.sh

conda activate myenv
