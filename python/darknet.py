from ctypes import *
import random
import argparse
import os
import logging
import ConfigParser
import cv2
import numpy as np
import io
from PIL import Image

#local
import process_logging
logger = logging.getLogger(__name__)


def sample(probs):
    s = sum(probs)
    probs = [a/s for a in probs]
    r = random.uniform(0, 1)
    for i in range(len(probs)):
        r = r - probs[i]
        if r <= 0:
            return i
    return len(probs)-1

def c_array(ctype, values):
    arr = (ctype*len(values))()
    arr[:] = values
    return arr

class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]

class DETECTION(Structure):
    _fields_ = [("bbox", BOX),
                ("classes", c_int),
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("sort_class", c_int)]


class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]

class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]

    
class darknet():
    def __init__(self, vocData):
        self.lib = CDLL("../libdarknet.so", RTLD_GLOBAL)

        self.lib.network_width.argtypes = [c_void_p]
        self.lib.network_width.restype = c_int
        self.lib.network_height.argtypes = [c_void_p]
        self.lib.network_height.restype = c_int

        self.predict = self.lib.network_predict
        self.predict.argtypes = [c_void_p, POINTER(c_float)]
        self.predict.restype = POINTER(c_float)

        self.set_gpu = self.lib.cuda_set_device
        self.set_gpu.argtypes = [c_int]

        self.make_image = self.lib.make_image
        self.make_image.argtypes = [c_int, c_int, c_int]
        self.make_image.restype = IMAGE

        self.get_network_boxes = self.lib.get_network_boxes
        self.get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int)]
        self.get_network_boxes.restype = POINTER(DETECTION)

        self.make_network_boxes = self.lib.make_network_boxes
        self.make_network_boxes.argtypes = [c_void_p]
        self.make_network_boxes.restype = POINTER(DETECTION)

        self.free_detections = self.lib.free_detections
        self.free_detections.argtypes = [POINTER(DETECTION), c_int]

        self.free_ptrs = self.lib.free_ptrs
        self.free_ptrs.argtypes = [POINTER(c_void_p), c_int]

        self.network_predict = self.lib.network_predict
        self.network_predict.argtypes = [c_void_p, POINTER(c_float)]

        self.reset_rnn = self.lib.reset_rnn
        self.reset_rnn.argtypes = [c_void_p]

        self.load_net = self.lib.load_network
        self.load_net.argtypes = [c_char_p, c_char_p, c_int]
        self.load_net.restype = c_void_p

        self.do_nms_obj = self.lib.do_nms_obj
        self.do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

        self.do_nms_sort = self.lib.do_nms_sort
        self.do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

        self.free_image = self.lib.free_image
        self.free_image.argtypes = [IMAGE]

        self.save_image = self.lib.save_image
        self.save_image.argtypes = [IMAGE, c_char_p]

        self.letterbox_image = self.lib.letterbox_image
        self.letterbox_image.argtypes = [IMAGE, c_int, c_int]
        self.letterbox_image.restype = IMAGE

        self.load_meta = self.lib.get_metadata
        self.lib.get_metadata.argtypes = [c_char_p]
        self.lib.get_metadata.restype = METADATA

        self.load_image = self.lib.load_image_color
        self.load_image.argtypes = [c_char_p, c_int, c_int]
        self.load_image.restype = IMAGE

        self.rgbgr_image = self.lib.rgbgr_image
        self.rgbgr_image.argtypes = [IMAGE]

        self.predict_image = self.lib.network_predict_image
        self.predict_image.argtypes = [c_void_p, IMAGE]
        self.predict_image.restype = POINTER(c_float)


        #lrt
        self.free_float = self.lib.free_float
        self.free_float.argtypes = [POINTER(c_float)]

        self.what_time_is_it_now = self.lib.what_time_is_it_now
        self.what_time_is_it_now.restype = c_double

        self.float_to_image_lrt = self.lib.float_to_image_lrt
        self.float_to_image_lrt.argtrypes = [c_int,c_int,c_int,c_int,POINTER(c_char_p),POINTER(IMAGE)]

        self.image_to_float_lrt = self.lib.image_to_float_lrt
        self.image_to_float_lrt.argtypes = [IMAGE]
        self.image_to_float_lrt.restype = POINTER(c_float)

        self.load_alphabet = self.lib.load_alphabet
        self.load_alphabet.restype = POINTER(POINTER(IMAGE))

        self.draw_detections = self.lib.draw_detections
        self.draw_detections.argtypes = [IMAGE,POINTER(DETECTION),c_int,
                c_float,POINTER(c_char_p), POINTER(POINTER(IMAGE)), c_int]


        data_options = self.read_data_cfg(vocData)
        self.lib.log_init(data_options["detect_log"])

        #in your project the log file change yourself
        logger.info("test")

        weights = os.listdir(data_options["backup"])
        t1 = 0
        weightfile = None
        for w in weights:
            if not w.endswith('.weights'):
                continue
            f = os.path.join(data_options["backup"],w)
            t = os.path.getctime(f)
            if t > t1:
                t1 = t
                weightfile = f

        tmp_cfg_name = os.path.split(data_options["network"])
        detect_cfg_file = os.path.join(tmp_cfg_name[0], "detect-{0}".format(tmp_cfg_name[1]))
        if not os.path.exists(detect_cfg_file):
            nf = open(detect_cfg_file,'w')
            with open(data_options["network"],'r') as f:
                tmp = f.readlines()

                for a in tmp:
                    b = a.split('=')
                    if b == 0 :
                        continue
                    if b[0].lower() == 'batch':
                        a= 'batch=1\n'
                    elif b[0].lower() == 'subdivisions':
                        a= 'subdivisions=1\n'
                    nf.write(a)
            nf.close()


        logger.debug("Load cfg network file:{0}".format(detect_cfg_file))
        logger.debug("Load weight file:{0}".format(weightfile))

        self.net = self.load_net(detect_cfg_file, weightfile, 0)
        self.meta = self.load_meta(vocData)

        #lrt end


    def read_data_cfg(self, datacfg):
        options = dict()
        options['gpus'] = ''
        options['num_workers'] = '10'
        with open(datacfg, 'r') as fp:
            lines = fp.readlines()

        for line in lines:
            line = line.strip()
            if line == '':
                continue
            key,value = line.split('=')
            key = key.strip()
            value = value.strip()
            options[key] = value
        return options


    def classify(self, net, meta, im):
        out = self.predict_image(net, im)
        res = []
        for i in range(meta.classes):
            res.append((meta.names[i], out[i]))
        res = sorted(res, key=lambda x: -x[1])
        return res

    def detect(self, im, thresh=.5, hier_thresh=.5, nms=.45, use_alphabet=1):
        res = []
        num = c_int(0)
        pnum = pointer(num)

        self.predict_image(self.net, im)
        dets = self.get_network_boxes(self.net, im.w, im.h, thresh, hier_thresh, None, 1, pnum)
        num = pnum[0]
        if (nms): self.do_nms_obj(dets, num, self.meta.classes, nms);

        image_alphabet = None
        if use_alphabet:
            #TODO
            """TODO:It'll out of memory if use self.load_alphabet"""
            image_alphabet = self.load_alphabet()

        self.draw_detections(im, dets, num, thresh, 
                self.meta.names,
                image_alphabet, 
                self.meta.classes);

        for j in range(num):
            classe = -1
            for i in range(self.meta.classes):
                if dets[j].prob[i] > 0:
                    b = dets[j].bbox
                    res.append((self.meta.names[i], dets[j].prob[i], (b.x, b.y, b.w, b.h)))
        res = sorted(res, key=lambda x: -x[1])

        self.free_detections(dets, num)
        return res

    def camera(self,cv_image):
        h, w, c= cv_image.shape
        step = cv_image.dtype.itemsize * c * w

        img_data = cv_image.ravel()
        if not img_data.flags['C_CONTIGUOUS']:
            img_data = np.ascontiguous(img_data, dtype=img_data.dtype)
        img_data_ctypes_ptr = img_data.ctypes.data_as(POINTER(c_char_p))

        im = self.make_image(w, h, c)
        self.float_to_image_lrt(w,h,c,step,img_data_ctypes_ptr,im)
        self.rgbgr_image(im)

        res = self.detect(im,thresh=.5, hier_thresh=.5, nms=.45,use_alphabet=0)

        aaa = self.image_to_float_lrt(im)
        detect_image = np.ctypeslib.as_array(aaa,(im.h, im.w, im.c))
        detect_image = detect_image.astype(np.uint8)
        self.free_float(aaa)
        self.free_image(im)

        return detect_image,res


    def run(self,imgfile, imgpath):
        if imgfile:
            im = self.load_image(imgfile, 0, 0)
            r = self.detect(im)
            for a in r:
                print a
            output = os.path.basename(imgfile)
            self.save_image(im,imgfile.replace(output,"detect_{0}".format(output)))
            self.free_image(im)

        if imgpath:
            filelist = os.listdir(imgpath)
            for file in filelist:
                if file.lower().endswith('.jpg'):
                    imgfile = os.path.join(imgpath,file)
                    im = self.load_image(imgfile, 0, 0)
                    r = self.detect(im)
                    self.free_image(im)
                    #print r

    
if __name__ == "__main__":
    process_logging.initLogging("/tmp/test_detect.log")
    parser = argparse.ArgumentParser()
    parser.add_argument('vocData',help=('The voc.data of this model trained'))
    parser.add_argument('-i',help=('To detect image file'))
    parser.add_argument('-p',help=('To detect image path'))
    parser.add_argument('-d',action='store_true',default=False,help=('The debug model will show image'))
    parser.add_argument('-f',action='store_true',default=False,help=('Detect the image path forever'))
    parser.add_argument('-c',action='store_true',default=False,help=('camera'))
    args = parser.parse_args()

    d = darknet(args.vocData)
    d.run(args.i,args.p)
    logger.info("test over")
    import datetime
    print datetime.datetime.now()
    
