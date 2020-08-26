import os
try:
    import wget
    wgetyes = True
except ImportError:
    wgetyes = False

def download_data(url, sourcedir):
    if not os.path.isdir(sourcedir):
        print('makedir', sourcedir)
        os.makedirs(sourcedir)
    # if wgetyes is True:
    if False:
        fname = os.path.basename(url)
        outfile = os.path.join(sourcedir, fname)
        wget.download(url, out=outfile)
    else:
        command = 'wget %s -P %s' % (url, sourcedir)
        os.system(command)
    return True

scriptpath = os.path.split(os.path.realpath(__file__))[0]
urlfname = os.path.join(scriptpath, 'pyhammer_dataurl.txt')
urllst = [i.strip() for i in open(urlfname)]
dicnameurl = dict([[os.path.basename(name), name] for name in urllst])


def get_url(fname):
    basename = os.path.basename(fname)
    if basename in dicnameurl:
        return dicnameurl[basename]
    else:
        return None