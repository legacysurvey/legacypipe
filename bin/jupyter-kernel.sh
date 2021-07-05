#! /bin/bash

# Don't add ~/.local/ to Python's sys.path
#export PYTHONNOUSERSITE=1

# Config directory nonsense
export TMPCACHE=$(mktemp -d)
echo "TMPCACHE $TMPCACHE"
mkdir "$TMPCACHE/cache"
mkdir "$TMPCACHE/config"
# astropy
export XDG_CACHE_HOME=$TMPCACHE/cache
export XDG_CONFIG_HOME=$TMPCACHE/config
mkdir "$XDG_CACHE_HOME/astropy"
cp -r "$HOME/.astropy/cache" "$XDG_CACHE_HOME/astropy"
mkdir "$XDG_CONFIG_HOME/astropy"
cp -r "$HOME/.astropy/config" "$XDG_CONFIG_HOME/astropy"
# matplotlib
export MPLCONFIGDIR=$TMPCACHE/matplotlib
mkdir "$MPLCONFIGDIR"
cp -r "$HOME/.config/matplotlib" "$MPLCONFIGDIR"
# ipython
export IPYTHONDIR=$TMPCACHE/ipython
mkdir "$IPYTHONDIR"
cp -r "$HOME/.ipython" "$IPYTHONDIR"

# (after config dir nonsense, reset HOME because $HOME in the container is /homedir...)
export HOME=$REALHOME

# I want to do the same thing is the ipykernel_launcher __main__, but
# if I just run "python -m ipythkernel_launcher", it puts cwd on the
# PYTHONPATH (even with --ignore-cwd).  The '--ignore-cwd' also makes
# it drop '' from PYTHONPATH.  Creating and running a temp script
# instead, the temp dir gets added to PYTHONPATH, but that's okay.
# /opt/conda/bin/python -m ipykernel_launcher --ignore-cwd -f $1
TMPSCRIPT=$(mktemp -d)/script.py
cat <<EOF > $TMPSCRIPT
if __name__ == '__main__':
    from ipykernel import kernelapp as app
    app.launch_new_instance()
EOF
/opt/conda/bin/python "$TMPSCRIPT" --ignore-cwd -f $1
rm "$TMPSCRIPT"

# Clean up config directory nonsens.
rm -R "$TMPCACHE"
