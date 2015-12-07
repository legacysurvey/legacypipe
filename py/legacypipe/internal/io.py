import os, sys
import re

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
    

