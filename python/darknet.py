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
import time

logger = logging.getLogger(__name__)

def array_to_image(arr):
    arr = arr.transpose(2,0,1)
    c = arr.shape[0]
    h = arr.shape[1]
    w = arr.shape[2]
    arr = (arr/255.0).flatten()
    data = c_array(c_float, arr)
    im = IMAGE(w,h,c,data)
    return im

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
    def __init__(self, vocData, soFile='../libdarknet.so', input_width=None, input_height=None, input_channel=None):
        self.lib = CDLL(soFile, RTLD_GLOBAL)

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
        self.image_to_float_lrt.argtypes = [IMAGE,POINTER(c_char_p)]
        #self.image_to_float_lrt.restype = POINTER(c_float)

        self.load_alphabet = self.lib.load_alphabet
        self.load_alphabet.restype = POINTER(POINTER(IMAGE))

        self.draw_detections = self.lib.draw_detections
        self.draw_detections.argtypes = [IMAGE,POINTER(DETECTION),c_int,
                c_float,POINTER(c_char_p), POINTER(POINTER(IMAGE)), c_int]


        data_options = self.read_data_cfg(vocData)
        self.lib.log_init(data_options["detect_log"])

        #-------------------------------------------
        #in your project the log file change yourself
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

        self.input_height = input_height
        self.input_width = input_width
        self.input_channel = input_channel
        self.dtype_itemsize = 1
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
        dets = self.get_network_boxes(self.net, im.w, im.h, thresh, hier_thresh, None, 0, pnum)
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
        t1 = time.time()
        if self.input_width is None or self.input_height is None:
            self.input_height, self.input_width, self.input_channel = cv_image.shape
            self.step = self.dtype_itemsize * self.input_channel * self.input_width

        img_data_ctypes_ptr = cv_image.ravel().ctypes.data_as(POINTER(c_char_p))
        im = self.make_image(self.input_width, self.input_height, self.input_channel)
        self.float_to_image_lrt(self.step, img_data_ctypes_ptr, im)
        self.rgbgr_image(im)

        res = self.detect(im,thresh=.5, hier_thresh=.5, nms=.45,use_alphabet=0)
        self.image_to_float_lrt(im, img_data_ctypes_ptr)
        self.free_image(im)

        print 'DARKNET CAMERA Use time:{}'.format(time.time() - t1)
        return res, cv_image


    def run(self,toDetect,args):
        t1 = time.time()
        if args.i:
            im = self.load_image(toDetect, 0, 0)
            r = self.detect(im)
            output = os.path.basename(toDetect)
            self.save_image(im, toDetect.replace(output,"detect_{0}".format(output)))
            self.free_image(im)
            print 'DARKNET Use time:{}'.format(time.time() - t1)
            print 'RESULT:{}'.format(r)

        if args.p:
            filelist = os.listdir(toDetect)
            for file in filelist:
                if file.lower().endswith('.jpg'):
                    imgfile = os.path.join(toDetect,file)
                    output = os.path.basename(imgfile)
                    im = self.load_image(imgfile, 0, 0)
                    r = self.detect(im)
                    self.save_image(im,imgfile.replace(output,"detect_{0}".format(output)))
                    self.free_image(im)
                    print 'RESULT:{}'.format(r)
        if args.e:
            image = cv2.imread(toDetect)
            im = array_to_image(image)
            res = self.detect(im)
            for output in res:
                box = output[-1]
                top, left, bottom, right = box
                w = int(bottom)
                h = int(right)
                x = int(top) - w / 2
                y = int(left) - h / 2
                cv2.rectangle(image, (x,y), (x+w,y+h), (255,0,0), 1)
                #cv2.putText(image, predicted_class, text_origin, self.font_face, self.font_scale, self.font_color, self.font_thickness)
            print 'NEW TEST Use time:{}'.format(time.time() - t1)
            print 'RESULT:{}'.format(res)

        if args.c:
            image = cv2.imread(toDetect)
            self.camera(image)



    
if __name__ == "__main__":
    import process_logging
    process_logging.initLogging("/tmp/test_detect.log")
    parser = argparse.ArgumentParser()
    parser.add_argument('vocData',help=('The voc.data of this model trained'))
    parser.add_argument('toDetectImage',help=('To be detect Image file'))
    parser.add_argument('-i', action='store_true', default=False, help=('To detect image file with darknet pointer'))
    parser.add_argument('-p', action='store_true', default=False, help=('To detect image path with darknet pointer'))
    parser.add_argument('-c', action='store_true', default=False, help=('To detect image file with cv2 and darknet pointer'))
    parser.add_argument('-e', action='store_true', default=False, help=('To detect image file with cv2 easy change'))
    args = parser.parse_args()

    d = darknet(args.vocData)
    d.run(args.toDetectImage,args)
    logger.info("test over")
    
