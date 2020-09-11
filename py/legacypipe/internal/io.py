import os, sys
import re
import fitsio

def git_version():
    """Returns `git describe, or 'unknown' if not a git repo"""
    # ADM mostly stolen from desitarget.io.gitversion()
    import os
    from subprocess import Popen, PIPE, STDOUT
    origdir = os.getcwd()
    os.chdir(os.path.dirname(__file__))
    try:
        p = Popen(['git', "describe"], stdout=PIPE, stderr=STDOUT)
    except EnvironmentError:
        return 'unknown'

    os.chdir(origdir)
    out = p.communicate()[0]
    if p.returncode == 0:
        # - avoid py3 bytes and py3 unicode; get native str in both cases.
        return str(out.rstrip().decode('ascii'))
    else:
        return 'unknown'

def get_units(filename):
    """Extract a dictionary of {FIELD: unit} from (extension 1 of) a
    Tractor (or similar FITS) file.
    """
    # ADM the header for the first extension.
    hdr = fitsio.read_header(filename, 1)

    # ADM grab the potential names of each unit for each possible field.
    tunits = ["TUNIT{}".format(i) for i in range(1, hdr["TFIELDS"]+1)]
    tfields = [hdr["TTYPE{}".format(i)] for i in range(1, hdr["TFIELDS"]+1)]

    # ADM a dictionary of the unit for each field. The dictionary will
    # ADM have an empty string for units not included in the Tractor file.
    unitdict = {tfld.upper(): hdr[tunit] if tunit in hdr.keys() else ""
                for tfld, tunit in zip(tfields, tunits)}

    return unitdict

def parse_filename(filename):
    """parse filename to check if this is a tractor brick file;
    returns brickname if it is, otherwise raises ValueError"""
    if not filename.endswith('.fits'): raise ValueError
    #- match filename tractor-0003p027.fits -> brickname 0003p027
    match = re.search('tractor-(\d{4}[pm]\d{3})\.fits', 
            os.path.basename(filename))

    if not match: raise ValueError

    brickname = match.group(1)
    return brickname

def iter_tractor(root):
    """ Iterator over all tractor files in a directory.

        Parameters
        ----------
        root : string
            Path to start looking
        
        Returns
        -------
        An iterator of (brickname, filename).

        Examples
        --------
        >>> for brickname, filename in iter_tractor('./'):
        >>>     print(brickname, filename)
        
        Notes
        -----
        root can be a directory or a single file; both create an iterator
    """

    if os.path.isdir(root):
        for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
            for filename in filenames:
                try:
                    brickname = parse_filename(filename)
                    yield brickname, os.path.join(dirpath, filename)
                except ValueError:
                    #- not a brick file but that's ok; keep going
                    pass
    else:
        try:
            brickname = parse_filename(os.path.basename(root))
            yield brickname, root
        except ValueError:
            pass
    

