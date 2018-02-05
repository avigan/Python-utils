#!/usr/bin/env python
# coding: utf-8

'''
Interface for viewing images with the ds9 image viewer.
Loosely based on XPA, by Andrew Williams.

Before trying to use this, please read Requirements below.

Here is a basic summary for use:
    import RO.DS9
    import numpy
    ds9Win = RO.DS9.DS9Win()
    # show a FITS file in frame 1
    ds9Win. showFITSFile("foo/test.fits")
    # show an array in frame 2
    ds9Win.xpaset("frame 2")
    myArray = numpy.arange(10000).reshape([100,100])
    ds9Win.showArray(myArray)

For more information, see the XPA Access Points section
of the ds9 reference manual (under Help in ds9). Then experiment.

Extra Keyword Arguments:
Many commands take additional keywords as arguments. These are sent
as separate commands after the main command is executed.
Useful keywords for viewing images include: scale, orient and zoom.
Note that the value of each keyword is sent as an unquoted string.
If you want the value quoted, provide the quotes yourself, e.g.:
    foo='"quoted value"'

Template Argument:
The template argument allows you to specify which instance of ds9
you wish to command via xpa.
One common use is control more than one ds9 window.
Since ds9 can only have one window, you must launch
multiple instances of ds9 using the -title command-line
option to specify a different window title for each.
Then specify the window title as the template to RO.DS9.
See the XPA documentation for other uses for template,
such as talking to ds9 on a remote host.

For a list of local servers try % xpaget xpans

WARNING: ds9 3.0.3 and xpa 2.1.5 have several nasty bugs.
One I have not figured out to work around is that on Windows
showArray fails because the data undergoes newline translation.
See <http://staff.washington.edu/rowen/ds9andxpa.html>
for more information. I have not checked this on recent versions.

Requirements:

* Unix Requirements
- ds9 and xpa must be installed somewhere on your $PATH

* MacOS X Requirements
  If using the Aqua version of DS9 (the normal Mac application):
  - Use the version of the application that is meant for your operating system.
    For Leopard (MacOS X 10.5) download the Leopard version. For Tiger (MacOS X 10.4)
    download the Tiger version. If you try to use a Tiger version under Leopard,
    you will see a host of warning messages as RO.DS9 starts up the SAOImage DS9 application.
  - The application must be named "SAOImage DS9.app" or "SAOImageDS9.app";
    one of these should be the default for your version.
  - The application must be in one of the two standard application directories
    (~/Applications or /Applications on English systems).
  - xpa for darwin must be installed somewhere on your $PATH or in /usr/local/bin
    (unpack the package and "sudo cp" the binaries to the appropriate location).

  If using the darwin version of ds9 (x11-based):
  - ds9 for darwin must be installed somewhere on your $PATH or in /usr/local/bin
  - xpa for darwin must be installed somewhere on your $PATH or in /usr/local/bin
  Note: this module will look for xpa and ds9 in /usr/local/bin
  and will add that directory to your $PATH if necessary.
'''

import numpy
import os
import time
import warnings
import subprocess
import sys
import sysv_ipc as ipc


# initialize globals
_DebugSetup = False
_SetupError = 'Not yet setup'
_Popen = None
_DirFromWhichToRunDS9 = None
_DS9Path = None


def _platformName():
    return sys.platform


def _addToPATH(newPath):
    '''Add newPath to the PATH environment variable.
    Do nothing if newPath already in PATH.
    '''
    pathSep = ':'
    pathStr = os.environ.get('PATH', '')
    if newPath in pathStr:
        return

    if pathStr:
        pathStr = pathStr + pathSep + newPath
    else:
        pathStr = newPath
    os.environ['PATH'] = pathStr

    
def _findApp(appName, subDirs=None, doRaise=True):
    '''Find a Mac or Windows application by expicitly looking for
    the in the standard application directories.
    If found, add directory to the PATH (if necessary).
    
    Inputs:
    - appName   name of application, with .exe or .app extension
    - subDirs   subdirectories of the main application directories;
                specify None if no subdirs
    - doRaise   raise RuntimeError if not found?
    
    Returns a path to the application's directory.
    Return None or raise RuntimeError if not found.
    '''
    appDirs = ['/Applications', '~/Applications']
    if subDirs is None:
        subDirs = [None]
    dirTrials = []
    for appDir in appDirs:
        for subDir in subDirs:
            if subDir:
                trialDir = os.path.join(appDir, subDir)
            else:
                trialDir = appDir
            dirTrials.append(trialDir)
            if os.path.exists(os.path.join(trialDir, appName)):
                _addToPATH(trialDir)
                return trialDir
    if doRaise:
        raise RuntimeError('Could not find %s in %s' % (appName, dirTrials,))
    return None
    

def _findUnixApp(appName):
    '''Use the unix 'which' command to find the application on the PATH
    Return the path if found.
    Raise RuntimeError if not found.
    '''
    p = subprocess.Popen(
        args=('which', appName),
        shell=False,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        p.stdin.close()
        errMsg = p.stderr.read()
        if errMsg:
            fullErrMsg = "'which %s' failed: %s" % (appName, errMsg)
            raise RuntimeError(fullErrMsg)
        appPath = p.stdout.read().decode()
        if not appPath.startswith('/'):
            raise RuntimeError('Could not find %s on your PATH' % (appName,))
    finally:
        p.stdout.close()
        p.stderr.close()

    return appPath


def _findDS9AndXPA():
    '''Locate ds9 and xpa, and add to PATH if not already there.
    
    Returns:
    - ds9Dir    directory containing ds9 executable
    - xpaDir    directory containing xpaget and (presumably)
                the other xpa executables
    
    Sets global variables:
    - _DirFromWhichToRunDS9 (the default dir from which to open DS9)
        - On Windows set to xpaDir to make sure that ds9 on Windows can find xpans
          and register itself with xpa when it starts up.
        - Otherwise set to None
    - _DS9Path (the path to ds9 executable)
        - On MacOS X if using the aqua SAOImage DS9 application then the path to the ds9 command line
          executable inside the aqua application bundle
        - Otherwise set to "ds9"; it is assumed to be on the PATH
                
    Raise RuntimeError if ds9 or xpa are not found.
    '''
    global _DirFromWhichToRunDS9, _DS9Path
    
    _DirFromWhichToRunDS9 = None
    _DS9Path = 'ds9'
    platformName = _platformName()
    
    if platformName == 'darwin':
        # ds9 and xpa may be in any of:
        # - ~/Applications/ds9.app
        # - /Applications.ds9.app
        # - on the PATH (adding /usr/local/bin if necessary)
        
        # add DISPLAY envinronment variable, if necessary
        # (since ds9 is an X11 application and environment
        os.environ.setdefault('DISPLAY', 'localhost:0')

        # look for ds9 and xpa inside of "ds9.app" or "SAOImage DS9.app"
        # in the standard application locations
        ds9Dir = _findApp('ds9', [
            'SAOImage DS9.app/Contents/MacOS',
            'SAOImageDS9.app/Contents/MacOS',
        ], doRaise=False)
        foundDS9 = (ds9Dir is not None)
        if foundDS9:
            _DS9Path = os.path.join(ds9Dir, 'ds9')
        foundXPA = False
        if ds9Dir and os.path.exists(os.path.join(ds9Dir, 'xpaget')):
            xpaDir = ds9Dir
            foundXPA = True

        # for anything not found, look on the PATH
        # after making sure /usr/local/bin is on the PATH
        if not (foundDS9 and foundXPA):
            # make sure /usr/local/bin is on the PATH
            # (if PATH isn't being set in ~/.MacOSX.environment.plist
            # then the bundled Mac app will only see the standard default PATH).
            _addToPATH('/usr/local/bin')
            _addToPATH('~/bin')

            if not foundDS9:
                ds9Dir = _findUnixApp('ds9')
    
            if not foundXPA:
                xpaDir = _findUnixApp('xpaget')

    else:
        # unix
        ds9Dir = _findUnixApp('ds9')
        xpaDir = _findUnixApp('xpaget')
    
    if _DebugSetup:
        print('_DirFromWhichToRunDS9=%r' % (_DirFromWhichToRunDS9,))
        print('_DS9Path=%r' % (_DS9Path,))
    
    return (ds9Dir, xpaDir)
    

def setup(doRaise=False):
    '''Search for xpa and ds9 and set globals accordingly.
    Return None if all is well, else return an error string.
    The return value is also saved in global variable _SetupError.
    
    Sets global variables:
    - _SetupError   same value as returned
    - _Popen        subprocess.Popen, if ds9 and xpa found,
                    else a variant that searches for ds9 and xpa
                    first and either runs subprocess.Popen if found
                    or else raises an exception.
                    This permits the user to install ds9 and xpa
                    and use this module without reloading it
    plus any global variables set by _findDS9AndXPA (which see)
    '''
    
    global _SetupError, _Popen
    _SetupError = None
    try:
        ds9Dir, xpaDir = _findDS9AndXPA()
        if _DebugSetup:
            print('ds9Dir=%r\nxpaDir=%r' % (ds9Dir, xpaDir))
    except Exception as e:
        _SetupError = 'RO.DS9 unusable: %s' % (e,)
        ds9Dir = xpaDir = None
    
    if _SetupError:
        class _Popen(subprocess.Popen):
            def __init__(self, *args, **kargs):
                setup(doRaise=True)
                subprocess.Popen.__init__(self, *args, **kargs)
        
        if doRaise:
            raise RuntimeError(_SetupError)
    else:
        _Popen = subprocess.Popen
    return _SetupError


errStr = setup(doRaise=False)
if errStr:
    warnings.warn(errStr)

_ArrayKeys = ('dim', 'dims', 'xdim', 'ydim', 'zdim', 'bitpix', 'skip', 'arch')
_DefTemplate = 'ds9'

_OpenCheckInterval = 0.2   # seconds
_MaxOpenTime = 10.0        # seconds


def xpaget(cmd, template=_DefTemplate, doRaise=False):
    '''Executes a simple xpaget command:
        xpaget -p <template> <cmd>
    returning the reply.
    
    Inputs:
    - cmd       command to execute; may be a string or a list
    - template  xpa template; can be the ds9 window title
                (as specified in the -title command-line option)
                host:port, etc.
    - doRaise   if True, raise RuntimeError if there is a communications error,
                else issue a UserWarning warning

    Raises RuntimeError or issues a warning (depending on doRaise)
    if anything is written to stderr.
    '''
    global _Popen
    fullCmd = 'xpaget %s %s' % (template, cmd,)

    p = _Popen(
        args=fullCmd,
        shell=True,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        p.stdin.close()
        errMsg = p.stderr.read()
        if errMsg:
            fullErrMsg = '%r failed: %s' % (fullCmd, errMsg)
            if doRaise:
                raise RuntimeError(fullErrMsg)
            else:
                warnings.warn(fullErrMsg)
        return p.stdout.read()
    finally:
        p.stdout.close()
        p.stderr.close()


def xpaset(cmd, data=None, dataArray=None, template=_DefTemplate, doRaise=False):
    '''Executes a simple xpaset command:
        xpaset -p <template> <cmd>
    or else feeds data to:
        xpaset <template> <cmd>
        
    The command must not return any output for normal completion.
    
    Inputs:
    - cmd       command to execute
    - data      data to write to xpaset's stdin; ignored if dataArray specified.
                If data[-1] is not \n then a final \n is appended.
    - dataArray a numpy array that needs to be written to the memory segment
                shared with ds9
    - template  xpa template; can be the ds9 window title
                (as specified in the -title command-line option)
                host:port, etc.
    - doRaise   if True, raise RuntimeError if there is a communications error,
                else issue a UserWarning warning
    
    Raises RuntimeError or issues a warning (depending on doRaise)
    if anything is written to stdout or stderr.
    '''
    global _Popen
    if data:
        fullCmd = 'xpaset %s %s' % (template, cmd)
    else:
        fullCmd = 'xpaset -p %s %s' % (template, cmd)

    if (dataArray is not None):
        mem = ipc.SharedMemory(None, flags=ipc.IPC_CREX, size=dataArray.nbytes)
        
        fullCmd = fullCmd.format(mem.id)
        
        mem.write(dataArray.tobytes('C'))
        
        subprocess.call(fullCmd, shell=True)
        
        mem.detach()
        mem.remove()
        del mem
    else:
        mp = _Popen(
            fullCmd, 
            shell=True,
            stdin=subprocess.PIPE, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT,
        )        
        try:
            if data:
                mp.stdin.write(bytes(data, 'UTF-8'))
                if data[-1] != '\n':
                    mp.stdin.write(bytes('\n', 'UTF-8'))
            mp.stdin.close()
            reply = mp.stdout.read()
            if reply:
                fullErrMsg = '%r failed: %s' % (fullCmd, reply.strip())
                if doRaise:
                    raise RuntimeError(fullErrMsg)
                else:
                    warnings.warn(fullErrMsg)
        finally:
            mp.stdin.close()  # redundant
            mp.stdout.close()


def _computeCnvDict():
    '''Compute array type conversion dict.
    Each item is: unsupported type: type to which to convert.
    
    ds9 supports UInt8, Int16, Int32, Float32 and Float64.
    '''
    
    cnvDict = {
        numpy.int8: numpy.int16,
        numpy.uint16: numpy.int32,
        numpy.uint32: numpy.float64,    # ds9 can't handle 64 bit integer data
        numpy.int64: numpy.float64,
    }
    if hasattr(numpy, 'uint64='):
        cnvDict[numpy.uint64] = numpy.float64

    return cnvDict


_CnvDict = _computeCnvDict()


def _expandPath(fname, extraArgs=''):
    '''Expand a file path and protect it such that spaces are allowed.
    Inputs:
    - fname     file path to expand
    - extraArgs extra arguments that are to be appended
                to the file path
    '''
    filepath = os.path.abspath(os.path.expanduser(fname))
    # if windows, change \ to / to work around a bug in ds9
    filepath = filepath.replace('\\', '/')
    # quote with '{...}' to allow ds9 to handle spaces in the file path
    return '{%s%s}' % (filepath, extraArgs)


def _formatOptions(kargs):
    '''Returns a string: 'key1=val1,key2=val2,...'
    (where keyx and valx are string representations)
    '''
    arglist = ['%s=%s' % keyVal for keyVal in kargs.items()]
    return '%s' % (','.join(arglist))


def _splitDict(inDict, keys):
    '''Splits a dictionary into two parts:
    - outDict contains any keys listed in 'keys';
      this is returned by the function
    - inDict has those keys removed (this is the dictionary passed in;
      it is modified by this call)
    '''
    outDict = {}
    for key in keys:
        if key in inDict:
            outDict[key] = inDict.pop(key)
    return outDict  


class DS9Win:
    '''An object that talks to a particular window on ds9
    
    Inputs:
    - template: window name (see ds9 docs for talking to a remote ds9);
            ignored on MacOS X (unless using X11 version of ds9).
    - doOpen: open ds9 using the desired template, if not already open.
    - doRaise   if True, raise RuntimeError if there is a communications error,
            else issue a UserWarning warning.
            Note: doOpen always raises RuntimeError on failure!
    - closeFDs  True to prevent ds9 from inheriting your open file descriptors. Set True if your
            application uses demon threads, else open files may keep those threads open unnecessarily.
            False by default because it can be slow (python bug 1663329).
    '''
    def __init__(self, template=_DefTemplate, doOpen=True, doRaise=False, closeFDs=False):
        self.template = str(template)
        self.doRaise = bool(doRaise)
        self.closeFDs = bool(closeFDs)
        if doOpen:
            self.doOpen()
    
    def doOpen(self):
        '''Open the ds9 window (if necessary).
        
        Raise OSError or RuntimeError on failure, even if doRaise is False.
        '''
        if self.isOpen():
            return
        
        global _DirFromWhichToRunDS9, _DS9Path, _Popen
        _Popen(
            args=(_DS9Path, '-title', self.template, '-port', '0'),
            cwd=_DirFromWhichToRunDS9, 
        )

        startTime = time.time()
        while True:
            time.sleep(_OpenCheckInterval)
            if self.isOpen():
                break
            if time.time() - startTime > _MaxOpenTime:
                raise RuntimeError('Could not open ds9 window %r; timeout' % (self.template,))

    def isOpen(self):
        '''Return True if this ds9 window is open
        and available for communication, False otherwise.
        '''
        try:
            xpaget('mode', template=self.template, doRaise=True)
            return True
        except RuntimeError:
            return False

    def showArray(self, arr, **kargs):
        '''Display a 2-d or 3-d grayscale integer numpy arrays.
        3-d images are displayed as data cubes, meaning one can
        view a single z at a time or play through them as a movie,
        that sort of thing.
        
        Inputs:
        - arr: a numpy array; must be 2-d or 3-d:
            2-d arrays have index order (y, x)
            3-d arrays are loaded as a data cube index order (z, y, x)
        kargs: see Extra Keyword Arguments in the module doc string for information.
        Keywords that specify array info (see doc for showBinFile for the list)
        are ignored, because array info is determined from the array itself.
        
        Data types:
        - UInt8, Int16, Int32 and floating point types sent unmodified.
        - All other integer types are converted before transmission.
        - Complex types are rejected.
    
        Raises ValueError if arr's elements are not some kind of integer.
        Raises RuntimeError if ds9 is not running or returns an error message.
        '''

        # reopen ds9 if necessary
        if (self.isOpen() is False):
            self.doOpen()
        
        arr = numpy.asarray(arr)
        
        if arr.dtype.name.startswith('complex'):
            raise TypeError('ds9 cannot handle complex data')

        ndim = len(arr.shape)
        if ndim not in (2, 3):
            raise RuntimeError('can only display 2d and 3d arrays')
        dimNames = ['z', 'y', 'x'][3-ndim:]

        # if necessary, convert array type
        cnvType = _CnvDict.get(arr.dtype)
        if cnvType:
            # print('converting array from %s to %s' % (arr.dtype, cnvType))
            arr = arr.astype(cnvType)

        # determine byte order of array (^ is xor)
        isBigEndian = arr.dtype.isnative ^ numpy.little_endian
        
        # compute bits/pix; ds9 uses negative values for floating values
        bitsPerPix = arr.itemsize * 8
        if arr.dtype.name.startswith('float'):
            # array is float; use negative value
            bitsPerPix = -bitsPerPix
    
        # remove array info keywords from kargs; we compute all that
        _splitDict(kargs, _ArrayKeys)

        # generate array info keywords; note that numpy
        # 2-d images are in order [y, x]
        # 3-d images are in order [z, y, x]
        arryDict = {}
        for axis, size in zip(dimNames, arr.shape):
            arryDict['%sdim' % axis] = size
        
        arryDict['bitpix'] = bitsPerPix
        if (isBigEndian):
            arryDict['arch'] = 'bigendian'
        else:
            arryDict['arch'] = 'littleendian'
            
        self.xpaset(
            cmd='shm array shmid {0} [%s]' % (_formatOptions(arryDict),),
            dataArray=arr,
        )
        
        for keyValue in kargs:
            self.xpaset(cmd=' '.join(keyValue))
    
    def showFITSFile(self, fname, **kargs):
        '''Display a fits file in ds9.
        
        Inputs:
        - fname name of file (including path information, if necessary)
        kargs: see Extra Keyword Arguments in the module doc string for information.
        Keywords that specify array info (see doc for showBinFile for the list)
        must NOT be included.
        '''
        filepath = _expandPath(fname)
        self.xpaset(cmd='fits "%s"' % filepath)

        # remove array info keywords from kargs; we compute all that
        arrKeys = _splitDict(kargs, _ArrayKeys)
        if arrKeys:
            raise RuntimeError('Array info not allowed; rejected keywords: %s' % arrKeys.keys())
        
        for keyValue in kargs:
            self.xpaset(cmd=' '.join(keyValue))

    def xpaget(self, cmd):
        '''Execute a simple xpaget command and return the reply.
        
        The command is of the form:
            xpaset -p <template> <cmd>
        
        Inputs:
        - cmd       command to execute
    
        Raises RuntimeError if anything is written to stderr.
        '''

        # reopen ds9 if necessary
        if (self.isOpen() is False):
            self.doOpen()
                
        return xpaget(
            cmd=cmd,
            template=self.template,
            doRaise=self.doRaise,
        )
    

    def xpaset(self, cmd, data=None, dataArray=None):
        '''Executes a simple xpaset command:
            xpaset -p <template> <cmd>
        or else feeds data to:
            xpaset <template> <cmd>
            
        The command must not return any output for normal completion.
        
        Inputs:
        - cmd       command to execute
        - data      data to write to xpaset's stdin; ignored if dataArray specified
        - dataArray a numpy array that needs to be written to the memory segment
                    shared with ds9
        
        Raises RuntimeError if anything is written to stdout or stderr.
        '''

        # reopen ds9 if necessary
        if (self.isOpen() is False):
            self.doOpen()
                
        return xpaset(
            cmd=cmd,
            data=data,
            dataArray=dataArray,
            template=self.template,
            doRaise=self.doRaise,
        )

    
class DS9Viewer:
    '''
    User-friendly interface for using ds9 from Python.

    Requirements:
     * working ds9
     * xpa library
     * RO python library

    History:
    2015-10-05 - Arthur Vigan - first version
    2017-02-05 - Arthur Vigan - simplified syntax
    '''

    _defName = 'PyVis'
    DS9 = DS9Win(_defName)
    
    def __init__(self, name=_defName):
        pass

    def show(self, array, new=False):
        '''Display an image or cube in ds9

        Parameters
        ----------
        array : array
            The array to display

        new : bool
            Create a new frame
        '''

        if new:
            self.frame('new')
        
        self.DS9.showArray(array)

        
    def point(self, center, marker='cross', color='green', thick=1, fixed=False):
        '''Display a point on the current image

        Parameters
        ----------
        center : tuple
            Point center        

        marker : str 
            Type of the point (default: cross, circle, box, diamond,
            x, arrow). Default is cross

        color : str
            Color of the point (default: green, red, blue, cyan,
            magenta, yellow, black, white). Default is green

        thick : int 
            Thickness of the line. Default is 1

        fixed : bool
            Position is fixed. Default is False

        '''

        # decompose center + FITS convention
        cx = center[0] + 1
        cy = center[1] + 1
        
        # fixed
        if fixed:
            move = 0
        else:
            move = 1

        # format command
        cmd  = 'regions'
        data = 'physical; point {:f} {:f} '.format(cx, cy) + \
               '# point={:s} color={:s} width={:.0f} move={:d}'.format(marker, color, thick, move)
        
        # send command
        self.DS9.xpaset(cmd, data=data)

        
    def box(self, center, w, h, angle=0, color='green', thick=1, fixed=False):
        '''
        Display a box on the current image

        Parameters
        ----------
        center : tuple
            Box center

        width : float
            Box width

        height : float
            Box height

        angle : float
            Orientation angle of the box, in degrees. Default is 0

        color : str
            Color of the point (default: green, red, blue, cyan,
            magenta, yellow, black, white). Default is green

        thick : int 
            Thickness of the line. Default is 1

        fixed : bool
            Position is fixed. Default is False
        '''
        
        # decompose center + FITS convention
        cx = center[0] + 1
        cy = center[1] + 1

        # fixed
        if fixed:
            move = 0
        else:
            move = 1

        # format command
        cmd  = 'regions'
        data = 'physical; box {:f} {:f} {:f} {:f} {:f} '.format(cx, cy, w, h, angle) + \
               '# color={:s} width={:.0f} move={:d}'.format(color, thick, move)

        # send command
        self.DS9.xpaset(cmd, data=data)

        
    def circle(self, center, radius=20, color='green', thick=1, fixed=False):
        '''
        Display a circle on the current image

        Parameters
        ----------
        center : tuple
            Circle center

        radius : float
            Circle radius. Default is 20 pixels

        color : str
            Color of the point (default: green, red, blue, cyan,
            magenta, yellow, black, white). Default is green

        thick : int 
            Thickness of the line. Default is 1

        fixed : bool
            Position is fixed. Default is False
        '''
        
        # decompose center + FITS convention
        cx = center[0] + 1
        cy = center[1] + 1

        # fixed
        if fixed:
            move = 0
        else:
            move = 1

        # format command
        cmd  = 'regions'
        data = 'physical; circle {:f} {:f} {:f} '.format(cx, cy, radius) + \
               '# color={:s} width={:.0f} move={:d}'.format(color, thick, move)
        
        # send command
        self.DS9.xpaset(cmd, data=data)

    
    def ellipse(self, center, xradius=20, yradius=20, angle=0, color='green', thick=1, fixed=False):
        '''
        Display an ellipse on the current image

        Parameters
        ----------
        center : tuple
            Ellipse center

        xradius : float
            Ellipse x radius. Default is 20 pixels

        yradius : float
            Ellipse y radius. Default is 20 pixels

        angle : float
            Orientation angle of the ellipse, in degrees. Default is 0

        color : str
            Color of the point (default: green, red, blue, cyan,
            magenta, yellow, black, white). Default is green

        thick : int 
            Thickness of the line. Default is 1

        fixed : bool
            Position is fixed. Default is False
        '''
        
        # decompose center + FITS convention
        cx = center[0] + 1
        cy = center[1] + 1

        # fixed
        if fixed:
            move = 0
        else:
            move = 1

        # format command
        cmd  = 'regions'
        data = 'physical; ellipse {:f} {:f} {:f} {:f} {:f} '.format(cx, cy, xradius, yradius, angle) + \
               '# color={:s} width={:.0f} move={:d}'.format(color, thick, move)

        # send command
        self.DS9.xpaset(cmd, data=data)

        
    def line(self, center0, center1, color='green', thick=1, fixed=False, text=''):
        '''
        Display a line on the current image

        Parameters
        ----------
        center0 : tuple
            First end center

        center1 : tuple
            Second end center

        color : str
            Color of the point (default: green, red, blue, cyan,
            magenta, yellow, black, white). Default is green

        thick : int 
            Thickness of the line. Default is 1

        fixed : bool
            Position is fixed. Default is False
        '''
        
        # decompose center + FITS convention
        cx0 = center0[0] + 1
        cy0 = center0[1] + 1
        cx1 = center1[0] + 1
        cy1 = center1[1] + 1

        # fixed
        if fixed:
            move = 0
        else:
            move = 1

        # format command
        cmd  = 'regions'
        data = 'physical; line {:f} {:f} {:f} {:f} {:f} '.format(cx0, cy0, cx1, cy1) + \
               '# color={:s} width={:.0f} move={:d} '.format(color, thick, move) + \
               'text={{"{:s}"}}'.format(text)

        # send command
        self.DS9.xpaset(cmd, data=data)

        
    def frame(self, cmd, num=None):
        '''
        Frame operations

        List of frame operations:
          - new: new frame
          - delete, del: delete current frame
          - delete all, del all: delete all frames
          - tile: tile frames
          - single: single out current frame
          - previous, prev, p: got to previous frame
          - next, n: go to next frame
          - set: set frame to number

        Parameters
        ----------
        cmd : str
            Frame command

        num : int
            Frame number
        '''
        cmd = cmd.lower()

        if cmd == 'new':
            self.DS9.xpaset('frame new')
        elif (cmd == 'delete') or (cmd == 'del'):
            self.DS9.xpaset('frame delete')
        elif (cmd == 'delete all') or (cmd == 'del all'):
            self.DS9.xpaset('frame delete all')
        elif cmd == 'tile':
            self.DS9.xpaset('tile')
        elif cmd == 'single':
            self.DS9.xpaset('single')
        elif (cmd == 'previous') or (cmd == 'prev') or (cmd == 'p'):
            self.DS9.xpaset('frame prev')
        elif (cmd == 'next') or (cmd == 'n'):
            self.DS9.xpaset('frame next')
        elif cmd == 'set':
            if (num is None):
                raise ValueError('You must pass a frame number')
            self.DS9.xpaset('frame frameno {0}'.format(num))
        else:
            print('Unknown command {:s} for frame.'.format(cmd))


    def scale(self, scl='linear', cuts='minmax'):
        '''
        Image scale operations

        Parameters
        ----------
        scl : str
            Scale mode

        cuts : str
            Cuts mode
        '''
        
        scl  = scl.lower()
        cuts = cuts.lower()

        if (scl == 'linear') or (scl == 'lin'):
            self.DS9.xpaset('scale linear')
        elif scl == 'sqrt':
            self.DS9.xpaset('scale sqrt')
        elif scl == 'log':
            self.DS9.xpaset('scale log 100')

        if (cuts == 'minmax') or (cuts == 'mm'):
            self.DS9.xpaset('scale mode minmax')
        elif (cuts == 'zscale') or (cuts == 'z'):
            self.DS9.xpaset('scale mode zscale')

            
    def imexam(self):
        '''
        Read point coordinates from user-selection on image

        Returns
        -------
        center : tuple
            Center of the selected point
        '''

        # format command
        cmd = 'imexam coordinate image'

        # send command
        point = self.DS9.xpaget(cmd).split()

        # FITS convention
        cx = float(point[0]) - 1
        cy = float(point[1]) - 1

        return cx, cy

    
    def zoom(self, zoom):
        '''
        Zoom image to a given value

        Parameters
        ----------
        zoom : float
            Zoom value
        '''

        # format command
        cmd = 'zoom to {:f}'.format(zoom)

        # send command
        self.DS9.xpaset(cmd)

        
    def pan(self, center=None):
        '''
        Pan image to a specific location

        Parameters
        ----------
        center : tupple
            Pan location. Default is None to recenter image
        '''
        
        if center is None:
            # format command
            cmd = 'frame center'
        else:
            # decompose center + FITS convention
            cx = center[0] + 1
            cy = center[1] + 1

            # format command
            cmd = 'pan to {:f} {:f}'.format(cx, float(cy))

        # send command
        self.DS9.xpaset(cmd)

        
if __name__ == '__main__':
    myArray = numpy.arange(10000).reshape([100, 100])
    myArray = numpy.random.normal(loc=0., scale=1., size=(100, 100))
    v = ds9('Test')
    v.show(myArray)
    v.circle(20, 45, 5, color='Blue', fixed=True)
    v.scale('linear')
    v.zoom(4.4)
    v.pan(0, 0)
    v.pan()
    p = v.imexam()
    print(p)
