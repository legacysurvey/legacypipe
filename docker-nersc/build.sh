#! /bin/bash

# Tunnel to NERSC's license server
ssh -fN intel-license

if [ "$(uname)" = "Darwin" ]; then
    # Mac OSX
    LOCAL_IP=$(ipconfig getifaddr $(route get nersc.gov | grep 'interface:' | awk '{print $NF}'))
else
    # Linux
    LOCAL_IP=$(ip route ls | tail -n 1 | awk '{print $NF}')
fi

docker build  --add-host intel.licenses.nersc.gov:${LOCAL_IP} -t legacysurvey/legacypipe:nersc .
