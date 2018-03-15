# Generate checksums for non-tractor output
# Martin Landriau, LBNL, February 2018
#
# Top level
cd /global/projecta/projectdirs/cosmo/work/legacysurvey/dr6
sha256sum $(find ./ -maxdepth 1 -type f -name "*fits*" -printf "%f\n") > legacysurvey_dr6.sha256sum
# Logs (assumes already tarred and gzipped)
cd logs
sha256sum *.tar.gz > legacysurvey_dr6_logs.sha256sum
# sweep
cd ../sweep/6.0
sha256sum *.fits > legacysurvey_dr6_sweep_6.0.sha256sum
# external
cd ../../external
sha256sum *.fits > legacysurvey_dr6_external.sha256sum
# For calibs, use python code

