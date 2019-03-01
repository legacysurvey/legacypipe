#! /bin/bash

# Tunnel to NERSC's license server
ssh -fN intel-license

# Mac OSX
LOCAL_IP=$(ipconfig getifaddr $(route get nersc.gov | grep 'interface:' | awk '{print $NF}'))

docker build  --add-host intel.licenses.nersc.gov:${LOCAL_IP} -t legacysurvey/legacypipe:nersc .
