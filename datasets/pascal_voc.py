import cPickle
import os
import uuid
import cv2
import xml.etree.ElementTree as ET

import numpy as np
import scipy.sparse

# from functools import partial

from imdb import ImageDataset
from voc_eval import voc_eval
# from utils.yolo import preprocess_train

import pdb

class VOCDataset(ImageDataset):
    def __init__(self, imdb_name, datadir, batch_size, im_processor, processes=3, shuffle=True, dst_size=None):
        super(VOCDataset, self).__init__(imdb_name, datadir, batch_size, im_processor, processes, shuffle, dst_size)
        meta = imdb_name.split('_')
        self._year = meta[1]
        self._image_set = meta[2]
        self._devkit_path = os.path.join(datadir, 'VOCdevkit{}'.format(self._year))
        self._data_path = os.path.join(self._devkit_path, 'VOC{}'.format(self._year))
        assert os.path.exists(self._devkit_path), 'VOCdevkit path does not exist: {}'.format(self._devkit_path)
        assert os.path.exists(self._data_path), 'Path does not exist: {}'.format(self._data_path)

        #self._classes = ('aeroplane', 'bicycle', 'bird', 'boat',
        #                 'bottle', 'bus', 'car', 'cat', 'chair',
        #                 'cow', 'diningtable', 'dog', 'horse',
        #                 'motorbike', 'person', 'pottedplant',
        #                 'sheep', 'sofa', 'train', 'tvmonitor')
        # gabriel: to nyu classes
        self._classes = ('__background__', # always index 0
            'bathtub', 'bed', 'bookshelf', 'box', 'chair', 'counter', 'desk',
            'door', 'dresser', 'garbage_bin', 'lamp', 'monitor', 'night_stand',
            'pillow', 'sink', 'sofa', 'table', 'tv', 'toilet');


        self._class_to_ind = dict(zip(self.classes, range(self.num_classes)))
        self._image_ext = '.jpg'

        self._salt = str(uuid.uuid4())
        self._comp_id = 'comp4'

        # PASCAL specific config options
        self.config = {'cleanup': True,
                       'use_salt': True}

        self.load_dataset()
        # self.im_processor = partial(process_im, image_names=self._image_names, annotations=self._annotations)
        # self.im_processor = preprocess_train

    def load_dataset(self):
        # set self._image_index and self._annotations
        self._image_indexes = self._load_image_set_index()
        self._image_names = [self.image_path_from_index(index) for index in self.image_indexes]
        self._annotations = self._load_pascal_annotations()

    def evaluate_detections(self, all_boxes, repo, epoch, output_dir=None): # gabriel
        """
        all_boxes is a list of length number-of-classes.
        Each list element is a list of length number-of-images.
        Each of those list elements is either an empty list []
        or a numpy array of detection.

        all_boxes[class][image] = [] or np.array of shape #dets x 5
        """
        self._write_voc_results_file(all_boxes, repo, epoch) # gabriel
        #self._write_voc_results_file(all_boxes)
        self._do_python_eval(repo, epoch, output_dir) # gabriel
        if self.config['cleanup']:
            for cls in self._classes:
                if cls == '__background__':
                    continue
                #filename = self._get_voc_results_file_template().format(cls)
                filename = self._get_voc_results_file_template(cls, repo, epoch) # gabriel
                #os.remove(filename)

    # -------------------------------------------------------------
    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        image_path = os.path.join(self._data_path, 'JPEGImages',
                                  index + self._image_ext)
        assert os.path.exists(image_path), \
            'Path does not exist: {}'.format(image_path)
        return image_path

    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        # Example path to image set file:
        # self._devkit_path + /VOCdevkit2007/VOC2007/ImageSets/Main/val.txt
        image_set_file = os.path.join(self._data_path, 'ImageSets', 'Main',
                                      self._image_set + '.txt')
        assert os.path.exists(image_set_file), \
            'Path does not exist: {}'.format(image_set_file)
        with open(image_set_file) as f:
            image_index = [x.strip() for x in f.readlines()]
        return image_index

    def _load_pascal_annotations(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} gt roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        gt_roidb = [self._annotation_from_index(index)
                    for index in self.image_indexes]
        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote gt roidb to {}'.format(cache_file)

        return gt_roidb

    def _annotation_from_index_original(self, index): # gabriel: load voc annotations
        """
        Load image and bounding boxes info from XML file in the PASCAL VOC
        format.
        """
        filename = os.path.join(self._data_path, 'Annotations', index + '.xml')
        tree = ET.parse(filename)
        objs = tree.findall('object')
        # if not self.config['use_diff']:
        #     # Exclude the samples labeled as difficult
        #     non_diff_objs = [
        #         obj for obj in objs if int(obj.find('difficult').text) == 0]
        #     # if len(non_diff_objs) != len(objs):
        #     #     print 'Removed {} difficult objects'.format(
        #     #         len(objs) - len(non_diff_objs))
        #     objs = non_diff_objs
        num_objs = len(objs)

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        # "Seg" area for pascal is just the box area
        seg_areas = np.zeros((num_objs), dtype=np.float32)
        ishards = np.zeros((num_objs), dtype=np.int32)

        # Load object bounding boxes into a data frame.
        for ix, obj in enumerate(objs):
            bbox = obj.find('bndbox')
            # Make pixel indexes 0-based
            x1 = float(bbox.find('xmin').text) - 1
            y1 = float(bbox.find('ymin').text) - 1
            x2 = float(bbox.find('xmax').text) - 1
            y2 = float(bbox.find('ymax').text) - 1

            diffc = obj.find('difficult')
            difficult = 0 if diffc == None else int(diffc.text)
            ishards[ix] = difficult

            cls = self._class_to_ind[obj.find('name').text.lower().strip()]
            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0
            seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)

        overlaps = scipy.sparse.csr_matrix(overlaps)

        return {'boxes': boxes,
                'gt_classes': gt_classes,
                'gt_ishard': ishards,
                'gt_overlaps': overlaps,
                'flipped': False,
                'seg_areas': seg_areas}

    def _annotation_from_index(self, index): # gabriel: load bv annotations
        """ 
        Load image and bounding boxes info from txt files of INRIAPerson.
        """
        filename = os.path.join(self._data_path, 'Annotations', index + '.txt')
        # print 'Loading: {}'.format(filename)
        with open(filename) as f:
                data = f.read()
        import re
        #objs = re.findall('\(\d+, \d+\)[\s\-]+\(\d+, \d+\)', data)
        objs = re.findall('\(\d+, \d+\)[\s\-]+\(\d+, \d+\)[\s\-]+\(\w+\)', data)

        num_objs = len(objs)

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)

        # Load object bounding boxes into a data frame.
        for ix, obj in enumerate(objs):
            # Make pixel indexes 0-based
            coor = re.findall('\d+', obj)
            x1 = float(coor[0])
            y1 = float(coor[1])
            x2 = float(coor[2])
            y2 = float(coor[3])
            #cls = self._class_to_ind['person']
            cls = self._class_to_ind[re.findall('\w+', obj)[-1]]
            boxes[ix, :] = [x1, y1, x2, y2] 
            gt_classes[ix] = cls 
            overlaps[ix, cls] = 1.0 

        overlaps = scipy.sparse.csr_matrix(overlaps)

        return {'boxes' : boxes,
                'gt_classes': gt_classes,
                'gt_overlaps' : overlaps,
                'flipped' : False}



    def _get_voc_results_file_template(self, cls, repo, epoch):
        # VOCdevkit/results/VOC2007/Main/<comp_id>_det_test_aeroplane.txt
        filename = '%s.txt'%(cls) # gabriel
        #filename = self._get_comp_id() + '_det_' + self._image_set + '_{:s}.txt'
        filedir = os.path.join(self._devkit_path, '../..','results', repo, str(epoch)) # gabriel
        #filedir = os.path.join(self._devkit_path, 'results', 'VOC' + self._year, 'Main')
        if not os.path.exists(filedir):
            os.makedirs(filedir)
        path = os.path.join(filedir, filename)
        return path

    def _write_voc_results_file(self, all_boxes, repo, epoch): # gabriel
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            print 'Writing {} VOC results file'.format(cls)
            #filename = self._get_voc_results_file_template().format(cls)
            filename = self._get_voc_results_file_template(cls, repo, epoch) # gabriel
            with open(filename, 'wt') as f:
                for im_ind, index in enumerate(self.image_indexes):
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                    # the VOCdevkit expects 1-based indices
                    for k in xrange(dets.shape[0]):
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(index, dets[k, -1],
                                       dets[k, 0] + 1, dets[k, 1] + 1,
                                       dets[k, 2] + 1, dets[k, 3] + 1))

    def _do_python_eval(self, repo, epoch, output_dir='output'): # gabriel
        annopath = os.path.join(
            self._devkit_path,
            'VOC' + self._year,
            'Annotations',
            '{:s}.txt')
            #'{:s}.xml') # gabriel: xml to txt
        imagesetfile = os.path.join(
            self._devkit_path,
            'VOC' + self._year,
            'ImageSets',
            'Main',
            self._image_set + '.txt')
        cachedir = os.path.join(self._devkit_path, 'annotations_cache')
        aps = []; recs = []
        # The PASCAL VOC metric changed in 2010
        use_07_metric = True if int(self._year) < 2010 else False
        print 'VOC07 metric? ' + ('Yes' if use_07_metric else 'No')
        if output_dir is not None and not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        # gabriel: write result to file
        resultdir = os.path.join(self._devkit_path, '../..','results', repo, str(epoch))
        resultf = open(resultdir + '/' + 'results.txt','w')
        for i, cls in enumerate(self._classes):
            if cls == '__background__':
                continue
            #filename = self._get_voc_results_file_template().format(cls)
            filename = self._get_voc_results_file_template(cls, repo, epoch) # gabriel
            rec, prec, ap, pred_box, gt_box = voc_eval(
                filename, annopath, imagesetfile, cls, cachedir, ovthresh=0.5,
                use_07_metric=use_07_metric)
            aps += [ap]
            recs += [rec[-1]] # gabriel
            print('AP = {:.4f}, maxRec = {:.4f} for {:11s}, pred box={:5d}, gt={:5d}'.format(ap, rec[-1], cls, pred_box, gt_box)) #gabriel
            resultf.write('AP = {:.4f}, maxRec = {:.4f} for {:11s}, pred box={:5d}, gt={:5d}\n'.format(ap, rec[-1], cls, pred_box, gt_box))
            if output_dir is not None:
                with open(os.path.join(output_dir, cls + '_pr.pkl'), 'w') as f:
                    cPickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
        resultf.close()
        print('Mean AP = {:.4f}'.format(np.mean(aps)))
        print('Mean maxRecall = {:.4f}'.format(np.mean(recs))) # gabriel
        #print('~~~~~~~~')
        #print('Results:')
        #for ap in aps:
        #    print('{:.3f}'.format(ap))
        #print('{:.3f}'.format(np.mean(aps)))
        #print('~~~~~~~~')
        #print('')
        #print('--------------------------------------------------------------')
        #print('Results computed with the **unofficial** Python eval code.')
        #print('Results should be very close to the official MATLAB eval code.')
        #print('Recompute with `./tools/reval.py --matlab ...` for your paper.')
        #print('-- Thanks, The Management')
        #print('--------------------------------------------------------------')

    def _get_comp_id(self):
        comp_id = (self._comp_id + '_' + self._salt if self.config['use_salt']
                   else self._comp_id)
        return comp_id
