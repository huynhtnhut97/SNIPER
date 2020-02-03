# --------------------------------------------------------------
# SNIPER: Efficient Multi-Scale Training
# Licensed under The Apache-2.0 License [see LICENSE for details]
# SNIPER demo
# by Mahyar Najibi
# --------------------------------------------------------------
import init
import matplotlib
matplotlib.use('Agg')
from configs.faster.default_configs import config, update_config, update_config_from_list
import mxnet as mx
import argparse
from train_utils.utils import create_logger, load_param
import os
from PIL import Image
from iterators.MNIteratorTest import MNIteratorTest
from easydict import EasyDict
from inference import Tester
from symbols.faster import *
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'


def parser():
    arg_parser = argparse.ArgumentParser('SNIPER demo module')
    arg_parser.add_argument('--cfg', dest='cfg', help='Path to the config file',
    							default='configs/faster/sniper_res101_e2e_pascal_voc.yml',type=str)
    arg_parser.add_argument('--save_prefix', dest='save_prefix', help='Prefix used for snapshotting the network',
                            default='SNIPER', type=str)
    arg_parser.add_argument('--im_path', dest='im_path', help='Path to the image', type=str,
                            default='data/demo/demo.jpg')
    arg_parser.add_argument('--set', dest='set_cfg_list', help='Set the configuration fields from command line',
                            default=None, nargs=argparse.REMAINDER)
    return arg_parser.parse_args()


def main():
    args = parser()
    update_config(args.cfg)
    if args.set_cfg_list:
        update_config_from_list(args.set_cfg_list)

    # Use just the first GPU for demo
    context = [mx.gpu(int(config.gpus[0]))]

    if not os.path.isdir(config.output_path):
        os.mkdir(config.output_path)

    images_path = "/mnt/data/nhuthuynh/sequences/"
    results_path = "./results"
    for folder in os.listdir(images_path):
        folder_path = os.path.join(images_path,folder)
        for image in os.listdir(folder_path):
            image_path = os.path.join(folder_path,image)
            # Get image dimensions
            width, height = Image.open(image_path).size

            # Pack image info
            roidb = [{'image': image_path, 'width': width, 'height': height, 'flipped': False}]

            # Creating the Logger
            logger, output_path = create_logger(config.output_path, args.cfg, config.dataset.image_set)

            # Pack db info
            db_info = EasyDict()
            db_info.name = 'coco'
            db_info.result_path = 'data/demo'

            # Categories the detector trained for:
            db_info.classes = [u'ignored_regions', u'pedestrian', u'people', u'bicycle', u'car', u'van',
                               u'truck', u'tricycle', u'awning-tricycle', u'bus', u'motor', u'others']
            db_info.num_classes = len(db_info.classes)

            # Create the model
            sym_def = eval('{}.{}'.format(config.symbol, config.symbol))
            sym_inst = sym_def(n_proposals=400, test_nbatch=1)
            sym = sym_inst.get_symbol_rcnn(config, is_train=False)
            test_iter = MNIteratorTest(roidb=roidb, config=config, batch_size=1, nGPUs=1, threads=1,
                                       crop_size=None, test_scale=config.TEST.SCALES[0],
                                       num_classes=db_info.num_classes)
            # Create the module
            shape_dict = dict(test_iter.provide_data_single)
            sym_inst.infer_shape(shape_dict)
            mod = mx.mod.Module(symbol=sym,
                                context=context,
                                data_names=[k[0] for k in test_iter.provide_data_single],
                                label_names=None)
            mod.bind(test_iter.provide_data, test_iter.provide_label, for_training=False)

            # Initialize the weights
            model_prefix = os.path.join(output_path, args.save_prefix)
            arg_params, aux_params = load_param(model_prefix, config.TEST.TEST_EPOCH,
                                                convert=True, process=True)
            mod.init_params(arg_params=arg_params, aux_params=aux_params)

            # Create the tester
            tester = Tester(mod, db_info, roidb, test_iter, cfg=config, batch_size=1)

            # Sequentially do detection over scales
            # NOTE: if you want to perform detection on multiple images consider using main_test which is parallel and faster
            all_detections= []
            for s in config.TEST.SCALES:
                # Set tester scale
                tester.set_scale(s)
                # Perform detection
                all_detections.append(tester.get_detections(vis=False, evaluate=False, cache_name=None))

            # Aggregate results from multiple scales and perform NMS
            tester = Tester(None, db_info, roidb, None, cfg=config, batch_size=1)
            file_name, out_extension = os.path.splitext(os.path.basename(args.im_path))
            all_detections = tester.aggregate(all_detections, vis=True, cache_name=None, vis_path='./data/demo/',
                                                  vis_name='{}_detections'.format(file_name), vis_ext=out_extension,filename=image_path)
    return all_detections
if __name__ == '__main__':
    main()