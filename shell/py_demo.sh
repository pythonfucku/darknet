#cp python/darknet.py bin_detect; cd bin_detect ;python darknet.py /home/crow/workspace/BBA/aiDATA/a_image_VOC_factory/VOCdevkit/sq/cfg/voc.data -i /home/crow/workspace/BBA/aiDATA/a_image_VOC_factory/VOCdevkit/sq/test/00000015.jpg;cd ..

#cp python/darknet.py bin_detect; cd bin_detect ;python darknet.py /home/crow/workspace/BBA/aiDATA/a_image_VOC_factory/VOCdevkit/test/cfg/voc.data -i data/dog.jpg;cd ..

cp python/camera_demo.py bin_detect;cd bin_detect ;python camera_demo.py /home/crow/workspace/BBA/aiDATA/a_image_VOC_factory/VOCdevkit/test/cfg/voc.data ;cd ..
