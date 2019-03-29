Building Docker container for use with Shifter at NERSC, using the Intel compilers
==================================================================================

This uses the Intel compilers, which introduces two complications:
- you need to access the license server at NERSC to build
- the intel compilers cannot be distributed, therefore can't be posted to Docker Hub,
  so NERSC provides a "two-stage build" recipe where you build in a full container, and
  then copy your results into a container with just the freely-distributable runtime
  components of the Intel suite.

Preliminaries:

- create an account at <https://hub.docker.com>
- ask someone to add you to the `legacysurvey` organization on Docker Hub.


First, set up your `~/.ssh/config` file, adding a stanza called `intel-license`:

```
Host intel-license
Hostname cori.nersc.gov
GatewayPorts yes
LocalForward 28519 intel.licenses.nersc.gov:28519
IdentityFile ~/.ssh/nersc
IdentitiesOnly yes
```

Second, run `sshproxy` if you haven't already done so today.

Third, run the `build.sh` script:

```
(cd docker-nersc && ./build.sh)
```

Note that the `build.sh` script tags the build as
`legacysurvey/legacypipe:nersc`.  This is a Docker Hub name;
see <https://hub.docker.com/r/legacysurvey/legacypipe/tags> for
tagged builds there.

Fourth, push the built container to Docker Hub:

```
docker push legacysurvey/legacypipe:nersc
```

You may need to do a

```
docker login
```

first.

Fifth, back at NERSC, pull the new container down from Docker Hub:

```
shifterimg pull docker:legacysurvey/legacypipe:nersc
```

Run it via:
```
shifter --image docker:legacysurvey/legacypipe:nersc bash
```
