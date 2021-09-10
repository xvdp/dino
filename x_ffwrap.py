"""
mini wrapper to ffmpeg to read and write files
from vidi @ git+https://github.com/xvdp/vidi
"""
import platform
import subprocess as sp
import warnings
import os
import os.path as osp
import json
from numpy.core.fromnumeric import size
import psutil
import numpy as np

class Col:
    AU = '\033[0m'
    BB = '\033[94m\033[1m'
    GB = '\033[92m\033[1m'
    YB = '\033[93m\033[1m'
    RB = '\033[91m\033[1m'
    B = '\033[1m'

class FFcap:
    """
        open pipe and write frames

        Args
            name        name of output video
            height      int 480
            width       int 640
            rate        int, float [30] framerate
            increment   bool [True], early closure does not corrupt file
            overwrite   bool [True],  overwrite file if found

            pix_fmt     str ["rgb24"], | # TODO write something other than rgb24  yuv420p
            vcodec="rawvideo"   # TODO write something other than rawvideo / "libx264" fails on incremental write
            channels=3,         # TODO derive channels from pix_fmt
            # TODO Maybe instead:- translate after recording raw

        Example:
            with vidi.FFcap("myvid.mp4", pix_fmt='yuv420p, fps=29.97, size=(480,640), overwrite=True') as F:
                F.add_frame(ndarray)

    """
    def __init__(self, name='vid.avi', height=480, width=640, rate=30, increment=True, overwrite=True,
                 pix_fmt="rgb24", vcodec="rawvideo", channels=3, **kwargs):
        #yuv420p
        self.name = name

        # options
        self.width = width
        self.height = height

        if "size" in kwargs:
            size = kwargs["size"]
            warnings.warn("'size' arg deprecated, use height=, width=", DeprecationWarning)
            if isinstance(size, int):
                self.width = size
                self.height = size
            else:
                self.height = size[0]
                self.width = size[1]
        if "fps" in kwargs:
            warnings.warn("'fps' arg deprecated, use rate=", DeprecationWarning)


        self.fps = rate

        self.increment = increment
        self.overwrite = overwrite
        self.pix_fmt = pix_fmt
        self.vcodec = vcodec

        # channels and pix format are tied in, 
        self.channels = channels
        self.shape = (self.height, self.width, self.channels)
        self.src_type = "stdin"

        self.audio = False

        self._cmd = None
        self._ffmpeg = 'ffmpeg' if platform.system() != 'Windows' else 'ffmpeg.exe'
        self._pipe = None
        self._framecount = 0


    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


    def open(self):
        if self._cmd is None:
            if self.src_type == "stdin":
                self.init_frames()
        if self._pipe is not None:
            self.close()
        self._pipe = sp.Popen(self._cmd, stdin=sp.PIPE, stderr=sp.PIPE)

    def close(self):
        if self._pipe is not None:
            self._pipe.stdin.flush()
            self._pipe.stdin.close()
            self._pipe.stderr.close()
            self._pipe.wait()

        self._pipe = None

    def init_frames(self):
        """given image frames
        """
        self._cmd = [self._ffmpeg]
        if self.overwrite:
            self._cmd += ['-y']

        #vlc unsupported codec 28 or profile 244
        # source video chroma type not supported

        # if self.vcodec == "rawvideo":
        self._cmd += ['-f', self.vcodec]
        self._cmd += ['-vcodec', self.vcodec]
        # else:
        #     self.increment = False
        #     self._cmd += ['-vcodec', 'libx264']

        self._cmd += ['-s', '%dx%d'%(self.width, self.height)]
        self._cmd += ['-pix_fmt', self.pix_fmt]

        # frames per second in resulting video
        self._cmd += ['-r', str(self.fps)]

        # from stream
        self._cmd += ['-i', '-']

        # audio
        if not self.audio:
            self._cmd += ['-an']
        else:
            #check, record desktop or record mic
            self._cmd += ["-thread_queue_size", "1024"]
            self._cmd += ["-f", "alsa", "-ac", "2", "-i", "pulse"]

        if self.increment:
            self._cmd += ["-movflags", "frag_keyframe"]

        self._cmd += [self.name]

    def add_frame(self, frame):
        """ pix_fmt rgb24 requires uint8 RGB
        Args
            frame   ndarray uint8 RGB shape (height, shape, 3) or (nb_frames, height, shape, 3)
        """
        _msg = "incorrect frame size <%s>; requires: <%s>"
        assert frame.ndim in (3,4) and frame.shape[-3:] == self.shape, _msg%(str(frame.shape), str(self.shape))
        assert frame.dtype == np.uint8, f"dtype supported np.uint8 found '{frame.dtype}'"
        if frame.ndim == 4:
            for _frame in frame:
                self.add_frame(_frame)
        else:
            self._pipe.stdin.write(frame.tobytes())
            self._framecount += 1


class FF:
    """wrapper class to ffmpeg, ffprobe, ffplay
        Examples:
            >>> from vidi import FF
            >>> f = FF('MUCBCN.mp4')
            >>> print(f.stats)
            >>> c = f.clip(start=100, nb_frames=200) #output video clip
            # output scalled reformated video clip
            >>> d = f.clip(start=500, nb_frames=300, out_format=".webm", scale=0.25)
            # output images
            >>> e = f.get_frames(out_format='.png', start=100, nb_frames=5, scale=0.6)
            >>> f.play(c)

    """
    def __init__(self, fname=None):

        self.ffplay = 'ffplay'
        self.ffmpeg = 'ffmpeg'
        self.ffprobe = 'ffprobe'
        self._if_win()
        self.stats = {}
        self.file = fname
        if osp.isfile(fname):
            self.get_video_stats()

    def _if_win(self):
        """ have not tested windows in a while, should work"""
        if platform.system() == 'Windows':
            self.ffplay = 'ffplay.exe'
            self.ffmpeg = 'ffmpeg.exe'
            self.ffprobe = 'ffprobe.exe'

    def get_video_stats(self, stream=0, entries=None, verbose=False):
        """file statistics
            returns full stats, stores subset to self.stats
            if video stream has rotation, rotates width and height in subset

        stream  (int[0]) subset of stats for video stream
        entries (list,tuple [None]) extra queries
        verbose (bool [False])
        """
        if not osp.isfile(self.file):
            print(f"{self.file} not a valid file")

        _cmd = f"{self.ffprobe} -v quiet -print_format json -show_format -show_streams {self.file}"
        with os.popen(_cmd) as _fi:
            stats = json.loads(_fi.read())

        if 'streams' not in stats:
            print(f"No streams found\n{stats}")
            return stats

        videos = [s for s in stats['streams'] if s['codec_type'] == 'video']
        audios = [s for s in stats['streams'] if s['codec_type'] == 'audio']

        if stream >= len(videos):
            stream = len(videos) - 1
            print(f"only {len(videos)} streams found, returning stream {stream}")

        # subset video stream
        _stats = videos[stream]
        if 'tags' in _stats and 'rotate' in _stats['tags'] and abs(eval(_stats['tags']['rotate'])) == 90:
            self.stats['width'] = _stats['height']
            self.stats['height'] = _stats['width']
        else:
            self.stats['width'] = _stats['width']
            self.stats['height'] = _stats['height']
        if 'r_frame_rate' in _stats:
            self.stats['rate'] = eval(_stats['r_frame_rate'])
        elif 'avg_frame_rate' in _stats:
            self.stats['rate'] = eval(_stats['avg_frame_rate'])

        self.stats['nb_frames'] = eval(_stats['nb_frames'])
        self.stats['type'] = 'video'
        self.stats['pad'] = "%%0%dd" %len(str(self.stats['nb_frames']))

        self.stats['file'] = self.file
        self.stats['pix_fmt'] = _stats['pix_fmt']

        if entries is not None:
            entries = [key for key in entries if key in _stats and key not in self.stats]
            for key in entries:
                self.stats[key] = eval(_stats[key])

        if verbose:
            print(_cmd)
            print(f"{len(audios)} audio streams, {len(videos)} video streams")
            print(f"\nStats for video stream: {stream}")
            print(self.stats)
            print("\nFull stats")
            print(json.dumps(stats, indent=2))
        return stats

    def frame_to_time(self, frame=0):
        """convert frame number to time"""
        if not self.stats:
            self.get_video_stats(stream=0)
        outtime = frame/self.stats['rate']
        return self.strftime(outtime)

    def time_to_frame(self, intime):
        frame = int(intime * self.stats['rate'])
        return frame

    def strftime(self, intime):
        return '%02d:%02d:%02d.%03d'%((intime//3600)%24, (intime//60)%60, intime%60, (int((intime - int(intime))*1000)))

    def play(self, loop=0, autoexit=True, fullscreen=False, noborder=True, showframe=False, fontcolor="white"):
        """ff play video

        ffplay -i metro.mov
        ffplay -start_number 5486 -i metro%08d.png
        Args
            loop        (int[0]) : number of loops, 0: forever
            autoexit    (bool [True]) close window on finish
            fullscreen  (bool [False])
            noborder    (bool[True])
            showframe   (bool[False]): draw current frame number
            fontcolor   (str [white])

        """
        if not self.stats:
            self.get_video_stats(stream=0)

        _fcmd = [self.ffplay, "-loop", str(loop)]
        if autoexit:
            _fcmd += ["-autoexit"]
        if noborder:
            _fcmd += ["-noborder"]
        if fullscreen:
            _fcmd += ["-fs"]
        _fcmd += ['-i', self.file ]
        if showframe:
            _cmd = f"drawtext=fontfile=Arial.ttf: x=(w-tw)*0.98: y=h-(2*lh): fontcolor={fontcolor}: fontsize=h*0.0185: " + "text='%{n}'"
            _fcmd += ["-vf", _cmd]   

        print(" ".join(_fcmd))
        print("-------Interaction--------")
        print(" 'q', ESC        Quit")
        print(" 'f', LMDC       Full Screen")
        print(" 'p', SPACE      Pause")
        print(" '9'/'0'         Change Volume")
        print(" 's'             Step one frame")
        print("  RT,DN / LT,UP  10sec jump")
        print("  RMC            jump to percentage of film")
        print("--------------------------")

        sp.call(_fcmd)
        #sp.Popen(_fcmd, stdin=sp.PIPE, stdout=sp.PIPE, stderr=sp.PIPE)
    
    @staticmethod
    def get_images(folder='.', name=None, fmt=('.jpg', '.jpeg', '.png'), max_imgs=None, as_str=False):
        start_frame = None
        folder = osp.abspath(folder)
        assert osp.isdir(folder), "<%s> not a valid folder"
        if name is None:
            name = sorted([f for f in os.listdir(folder) if osp.splitext(f)[1].lower() in fmt])

        if isinstance(name, list):
            name = name[:max_imgs]

        if isinstance(name, list):
            name = [osp.join(folder, f) for f in name]
            if name and as_str:
                name = 'concat:"' + '|'.join(name) +'"'

        elif isinstance(name, str):
            assert '%' in name, "template name not recognized expected format name%08d.png"
            if not osp.isdir(osp.split(name)[0]):
                name = osp.join(folder, name)
            
            name_root = osp.basename(name.split("%")[0])
            fmt = osp.splitext(name)[1]
            #print(name_root, fmt, name)
            first_file = sorted([f for f in os.listdir(folder) if osp.splitext(f)[1].lower() in fmt and name_root in f])[0]

            start_frame = int(osp.splitext(first_file.split(name_root)[1])[0])
            
        else:
            print("name must be None, template string or list, found ", type(name))
            name = False
        
        return name, start_frame

    def playfiles(self, fname=None, folder=".", max_frames=None):

        imgs, start = self.get_images(folder=folder, name=fname, fmt=('.jpg', '.jpeg', '.png'), max_imgs=max_frames)

        print(self.ffplay)
        print(start)
        print(self.ffplay)
        print(self.ffplay)
        if not imgs:
            return None

        _fcmd = [self.ffplay]
        if start or start is not None:
            _fcmd += ["-start_number", str(start)]
        _fcmd += ['-i', imgs]

        print(" ".join(_fcmd))
        print("-------Interaction--------")
        print(" 'q', ESC        Quit")
        print(" 'f', LMDC       Full Screen")
        print(" 'p', SPACE      Pause")
        print(" '9'/'0'         Change Volume")
        print(" 's'             Step one frame")
        print("  RT,DN / LT,UP  10sec jump")
        print("  RMC            jump to percentage of film")
        print("--------------------------")
        sp.call(_fcmd)

        return True

    def get_size(self, size=None):
        if size is not None:
            if isinstance(size, str):
                if 'x' not in size:
                    assert size.isnumeric(size), "size can be tuple of ints, string of ints sepparated by 'x' or single int"
                    size = int(size)
            if isinstance(size, int):
                size = (size, size)
            if isinstance(size, (tuple, list)):
                size = "%dx%d"%size
        return size

    def stitch(self, dst, src, audio, fps, size, start_img, max_img, pix_fmt="yuv420p"):
        """
        ffmpeg -r 29.97 -start_number 5468 -i metro%08d.png -vcodec libx264 -pix_fmt yuv420p /home/z/metropolis_color.mov
        ffmpeg -r 29.97 -start_number 5468 -i metro%08d.png -vcodec libx264 -pix_fmt yuv420p -vframes 200 /home/z/metro_col.mov #only 200 frames
        """
        _ff = 'ffmpeg' if platform.system() != "Windows" else 'ffmpeg.exe'
        _fcmd = [_ff, '-r', str(fps)]

        # has to be before input
        if start_img is not None:
            _fcmd += ['-start_number', str(start_img)]

        _fcmd += ['-i', src]

        if audio is not None:
            _fcmd += ['-i', audio]

        size = self.get_size(size)
        if size is not None:
            _fcmd += ['-s', size]

        # codecs
        _fcmd += ['-vcodec', 'libx264']
        if audio is not None:
            _fcmd += ['-acodec', 'copy']

        _fcmd += ['-pix_fmt', pix_fmt]
        _fcmd += [dst]

        # number of frames # has to be just before outpyut
        if max_img is not None:
            _fcmd += ['-vframes', str(max_img)]

        print(" ".join(_fcmd))
        sp.call(_fcmd)

    def _export(self, start=0, nb_frames=None, scale=1, step=1, stream=0, crop=None):
        """ common to clip and frame exports
        """
        if not self.stats:
            self.get_video_stats(stream=stream)
        _rate = self.stats['rate']
        _height = self.stats['height']
        _width = self.stats['width']

        # range start, frame and time
        if isinstance(start, float):
            time_start = self.strftime(start)
            start = self.time_to_frame(start)
        else:
            time_start = self.frame_to_time(start)

        # range size, number  and time
        _max_frames = self.stats['nb_frames'] - start
        if nb_frames is None:
            nb_frames = _max_frames
        else:
            nb_frames = min(_max_frames, nb_frames)
        time_end = str(nb_frames/_rate)

        # export command
        # '-vframes', str(nb_frames)]
        cmd =  [self.ffmpeg, '-i', self.file, '-ss', time_start, '-t', time_end]

        # resize
        if scale != 1:
            cmd += ['-s', '%dx%d'%(int(_width * scale), int(_height * scale))]

        if step > 1:
            cmd += ['-r', str(_rate // step), '-vf', 'mpdecimate,setpts=N/FRAME_RATE/TB']

        if isinstance(crop, (list, tuple)) and len(crop) == 4:
            _crop = f"crop={crop[0]}:{crop[1]}:{crop[2]}:{crop[3]}"
            if step == 1:
                cmd += ["-vf", _crop]
            else:
                cmd[-1] += ", "+_crop

        return cmd

    def export_frames(self, out_name=None, start=0, nb_frames=None, scale=1, step=1, stream=0, out_folder=None, **kwargs):
        """ extract frames from video
        Args
            out_name    (str)   # if no format in name ".png
            start       (int|float [0])       default: start of input clip, if float, start is a time
            nb_frames   (int [None]):   default: to end of clip
            scale       (float [1]) rescale output
            stream      (int [0]) if more than one stream in video
            out_folder  optional, save to folder

        kwargs
            out_format  (str) in (".png", ".jpg", ".bmp") format override
            crop        (list, tuple (w,h,x,y)
        """
        cmd = self._export(start=start, nb_frames=nb_frames, scale=scale, step=step, stream=stream, **kwargs)

        # resolve name
        if out_name is None:
            out_name = osp.splitext(self.file)[0]


        out_name, out_format = osp.splitext(out_name)

        if "out_format" in kwargs:
            out_format = kwargs["out_format"]

        if out_format.lower() not in (".png", ".jpg", ".jpeg", ".bmp"):
            out_format = ".png"

        if scale != 1:
            _pad = ""
            if "%0" in out_name:
                out_name, _pad = out_name.split("%0")
                if out_name[-1] == "_":
                    out_name = out_name[:-1]
                _pad = "_%0" + _pad
            out_name += f"_{scale}" + _pad

        if not "%0" in out_name:
            out_name += "_" + self.stats['pad']

        out_name += out_format
        if out_folder is not None:
            out_folder = osp.abspath(out_folder)
            os.makedirs(out_folder, exist_ok=True)
            out_name = osp.join(out_folder, out_name)

        cmd.append(out_name)
        print(" ".join(cmd))
        sp.Popen(cmd, stdin=sp.PIPE, stderr=sp.PIPE)

        return osp.abspath(out_name)

    def export_clip(self, out_name=None, start=0, nb_frames=None, scale=1, step=1, stream=0, out_folder=None, **kwargs):
        """ extract video clip
        Args:
            out_name    (str [None]) default is <input_name>_framein_frame_out<input_format>
            start       (int | float [0])  default: start of input clip, if float: time
            nb_frames   (int [None]):      default: to end of clip
            scale       (float [1]) rescale output
            stream      (int [0]) if more than one stream in video
        kwargs
            out_format  (str) in (".mov", ".mp4") format override
            crop        (list, tuple (w,h,x,y)
        TODO: generate intermediate video with keyframes then cut
        ffmpeg -i a.mp4 -force_key_frames 00:00:09,00:00:12 out.mp4

        # TODO copy codecs.
        <> -vcodec copy -acodec copy <>
        """
        cmd = self._export(start=start, nb_frames=nb_frames, scale=scale, step=step, stream=stream, **kwargs)

        # resolve name
        if out_name is None:
            if nb_frames is None:
                nb_frames = self.stats["nb_frames"]
            out_name = osp.splitext(self.file)[0]+ '_' + str(start) + '-' + str(nb_frames+start)

        # format
        out_name, out_format = osp.splitext(out_name)
        if not out_format:
            out_format = osp.splitext(self.file)[1]
        if "out_format" in kwargs:
            out_format = kwargs["out_format"]

        if scale != 1:
            out_name += f"_{scale}"

        out_name += out_format
        if out_folder is not None:
            out_folder = osp.abspath(out_folder)
            os.makedirs(out_folder, exist_ok=True)
            out_name = osp.join(out_folder, out_name)

        if out_format == '.gif' or out_format == '.webm':
            cmd = cmd + ['-bitrate', '3000k']

        cmd.append(out_name)
        print(" ".join(cmd))
        sp.Popen(cmd, stdin=sp.PIPE, stderr=sp.PIPE)

        return osp.abspath(out_name)


    def fits_in_memory(self, nb_frames=None, dtype_size=1, with_grad=0, scale=1, stream=0,
                       memory_type="CPU", step=1):
        """
            returns max number of frames that fit in memory
        """
        if not self.stats:
            self.get_video_stats(stream=stream)

        nb_frames = self.stats['nb_frames'] if nb_frames is None else min(self.stats['nb_frames'], nb_frames)

        width = self.stats['width'] * scale
        height = self.stats['height'] * scale
        _requested_ram = (nb_frames * width * height * dtype_size * (1 + with_grad))/step//2**20
        _available_ram = CPUse().available # removed gpu, we are not loading video to gpu!

        if _requested_ram > _available_ram:
            max_frames = int(nb_frames * _available_ram // _requested_ram)
            _msg = f"{Col.B}Cannot load [{nb_frames},{height},{width},3] frames in CPU {Col.RB}"
            _msg += f"{_available_ram} MB{Col.YB}, loading only {max_frames} frames {Col.AU}"
            print(_msg)
            nb_frames = max_frames
        return nb_frames

    def to_numpy(self, start=0, nb_frames=None, scale=1, stream=0, step=1, dtype=np.uint8, memory_type="CPU"):
        """
        read video to numpy
        Args
            start   (int [0]) start frame
            nb_frames   (int [None])
            scale       (float [1])
            stream      (int [0]) video stream to load
            step        (int [1]) step thru video

        TODO: loader iterator yield
        TODO: crop or transform
        TODO check input depth bytes, will fail if not 24bpp
        """
        if not self.stats:
            self.get_video_stats(stream=stream)
        nb_frames = self.stats['nb_frames'] if nb_frames is None else min(self.stats['nb_frames'], nb_frames + start)

        if isinstance(start, float):
            _time = self.strftime(start)
            start = self.time_to_frame(start)
        else:
            _time = self.frame_to_time(start)

        _fcmd = [self.ffmpeg, '-i', self.file, '-ss', _time, '-start_number', str(start),
                 '-f', 'rawvideo', '-pix_fmt', 'rgb24']

        width = self.stats['width']
        height = self.stats['height']
        if scale != 1:
            width = int(self.stats['width'] * scale)
            height = int(self.stats['height'] * scale)
            _scale = ['-s', '%dx%d'%(width, height)]
            _fcmd = _fcmd + _scale
        _fcmd += ['pipe:']

        bufsize = width*height*3

        nb_frames = self.fits_in_memory(nb_frames, dtype_size=np.dtype(dtype).itemsize, scale=scale,
                                        stream=stream, memory_type=memory_type, step=step)

        proc = sp.Popen(_fcmd, stdin=sp.PIPE, stdout=sp.PIPE, stderr=sp.PIPE, bufsize=bufsize)
        out = self._to_numpy_proc(start, nb_frames, step, width, height, dtype, bufsize, proc)

        proc.stdout.close()
        proc.wait()

        return out

    def _to_numpy_proc(self, start, nb_frames, step, width, height, dtype, bufsize, proc):
        """ read nb_frames at step from open pipe
        """
        out = []
        for i in range(start, nb_frames):
            if not i%step:
                buffer = proc.stdout.read(bufsize)
                if len(buffer) != bufsize:
                    break
                out += [np.frombuffer(buffer, dtype=np.uint8).reshape(height, width, 3).astype(dtype)]
        out = np.stack(out, axis=0)

        # TODO: this assumes that video input is uint8 but thats not a given - must validate pixel format
        if dtype in (np.float32, np.float64):
            out /= 255.
        del buffer

        return out



class CPUse:
    """thin wrap to psutil.virtual_memory to matching nvidia-smi syntax"""
    def __init__(self, units="MB"):
        cpu = psutil.virtual_memory()
        self.total = cpu.total
        self.used = cpu.used
        self.available= cpu.available
        self.percent = cpu.percent
        self.units = units if units[0].upper() in ('G', 'M') else 'MB'
        self._fix_units()

    def _fix_units(self):
        _scale = 20
        if self.units[0].upper() == "G":
            self.units = "GB"
            _scale = 30
        else:
            self.units = "MB"
        self.total //= 2**_scale
        self.used //= 2**_scale
        self.available //= 2**_scale

    def __repr__(self):
        return "CPU: ({})".format(self.__dict__)
