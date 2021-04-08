import os
import json
import tempfile
from contextlib import redirect_stdout
import torch
from apex import amp
from apex.parallel import DistributedDataParallel as DDP
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort

from .data import DataIterator
from .model import Model
from .utils import Profiler
from .utils import show_detections, show_MOT

import cv2
import numpy
import time
import shutil
from tqdm import tqdm




def mapClasses(classes):
    for i in range(len(classes)):
        cls = classes[i]
        if cls == 0:
            classes[i] = 1
        elif cls in [2,3,4,6,8]:
            classes[i] = 2
        else:
            classes[i] = 0
    return classes


def time_synchronized():
    # pytorch-accurate time
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()

def infer(model, data_path, detections_file, resize, max_size, batch_size, config_deepsort, mixed_precision=False, is_master=True, world=0, original_annotations=None, use_dali=True, is_validation=False, verbose=False, save_images = False, output_path = './'):
    'Run inference on images from path'
    # import pdb;pdb.set_trace()
    if os.path.isdir(output_path):
        shutil.rmtree(output_path)
    os.mkdir(output_path)
    print('model',model)
    backend = 'pytorch' if isinstance(model, Model) or isinstance(model, DDP) else 'tensorrt'

    #print("backend",backend)
    stride = model.module.stride if isinstance(model, DDP) else model.stride


    # TensorRT only supports fixed input sizes, so override input size accordingly
    if backend == 'tensorrt': max_size = max(model.input_size)



    cfg = get_config()
    cfg.merge_from_file(config_deepsort)


    conf_threshold = cfg.DEEPSORT.MIN_CONFIDENCE
    # Prepare model
    if backend is 'pytorch':
        # If we are doing validation during training,
        # no need to register model with AMP again
        if not is_validation:
            if torch.cuda.is_available(): model = model.cuda()
            model = amp.initialize(model, None,
                               opt_level = 'O2' if mixed_precision else 'O0',
                               keep_batchnorm_fp32 = True,
                               verbosity = 0)

        model.eval()

    if verbose:
        print('   backend: {}'.format(backend))
        print('    device: {} {}'.format(
            world, 'cpu' if not torch.cuda.is_available() else 'gpu' if world == 1 else 'gpus'))
        print('     batch: {}, precision: {}'.format(batch_size,
            'unknown' if backend is 'tensorrt' else 'mixed' if mixed_precision else 'full'))
    
    print('Running inference on {}'.format(os.path.basename(data_path)))

    results = []
    profiler = Profiler(['infer', 'fw'])




    def processResult(results, data_iterator):
        p_detections = []
        C = data_iterator.coco
        for d in results:
            
            id, outputs, ratios = d

            img = C.loadImgs([id])
            filename = img[0]['file_name']
            result = ['',[],[]]
            result[0] = os.path.join(path, filename)
            if len(outputs) > 0:
                # import pdb;pdb.set_trace()
                outputs[:,:4] = outputs[:,:4] / ratios
                result[1] = outputs
            A = C.loadAnns(C.getAnnIds([id]))
            # import pdb;pdb.set_trace()
            for a in A:
                x1, y1, w , h = a['bbox'] 
                a['bbox'] = [x1,y1,x1+ w, y1+h]
            result[2] = A
            p_detections += [result]
        return p_detections

    path =  data_path #+ 'sequences/'
    videoList = os.listdir(path)
    
    # Prepare dataset
    if verbose: print('Preparing dataset...')

    # Create annotations if none was provided
    if not original_annotations:
        return
    else:
        annotations = original_annotations

    data_iterator =  DataIterator(
        path, resize, max_size, batch_size, stride,
        world, annotations, training=False)


    detection_results = []

    id_count = 0
    sort_time = 0
    with torch.no_grad():
        for i, (data, ids, ratios) in enumerate(tqdm(data_iterator)):

            video = os.path.dirname(data_iterator.coco.loadImgs(ids.item())[0]['file_name'])
            if not os.path.isfile(os.path.join(output_path, video + '.txt')):
                id_count = i
                open(os.path.join(output_path, video + '.txt'),"w+")
                deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,\
                    max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,\
                    nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,\
                    max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,\
                    use_cuda=True)
                
                print(id_count)
                if save_images and len(results) > 0: 
                    output_anno = processResult(results, data_iterator=data_iterator)

                    print("saving output images...")
                    save_path = os.path.dirname(data_path) + '/outputs/' + video
                    if os.path.isdir(save_path):
                        shutil.rmtree(save_path)
                    os.mkdir(save_path)
                    show_MOT( save_path, output_anno)
                results = []

            # print("data:",data)
            # import pdb;pdb.set_trace()
            profiler.start('fw')
            t1 = time_synchronized()
            scores, boxes, classes = model(data)
            profiler.stop('fw')
            detection_results.append([scores, boxes, classes, ids, ratios])

            # import pdb;pdb.set_trace()

            t2 = time_synchronized()

            im =  data[0].permute(1,2,0).cpu().numpy()
            xywhs = torch.stack([torch.stack([x1 + (x2 - x1 + 1)/2, y1 + (y2 - y1 + 1)/2, x2 - x1 + 1, y2 - y1 + 1]) for x1,y1,x2,y2 in boxes[0].round()]).cpu()
            

            t3 = time_synchronized()
            outputs = deepsort.update(xywhs, scores[0].cpu(), im, mapClasses(classes[0].cpu()))

            # outputs = torch.Tensor(outputs).reshape(1,-1,5)
            t4 = time_synchronized()
            sort_time += t4 - t3
            if len(outputs) > 0:
                
                outputs[:,:4] = outputs[:,:4] / ratios[0].item()
            # print(t2-t1,t3-t2,t4-t3)
            
            results.append([ids[0].item(), outputs, 1])
            
            # write result to txt
            if len(outputs) != 0:
                for j, output in enumerate(outputs):
                    bbox_left = output[0]
                    bbox_top = output[1]
                    bbox_w = output[2] - output[0]
                    bbox_h = output[3] - output[1]
                    identity = output[-2]
                    cls = output[-1]
                    # if cls == 2:
                    #     continue
                    # import pdb;pdb.set_trace()

                    with open(os.path.join(output_path, video + '.txt'), 'a') as f:
                        f.write(('%g,' * 10 + '\n') % (ids[0].item() - id_count, identity, bbox_left,
                                                        bbox_top, bbox_w, bbox_h, 1, cls, -1, -1))  # label format

            profiler.bump('infer')
            if verbose and  (profiler.totals['infer'] > 60 or i == len(data_iterator) - 1):
                size = len(data_iterator.ids)
                msg  = '[{:{len}}/{}]'.format(min((i + 1) * batch_size,
                    size), size, len=len(str(size)))
                msg += ' {:.3f}s/{}-batch'.format(profiler.means['infer'], batch_size)
                msg += ' (fw: {:.3f}s)'.format(profiler.means['fw'])
                msg += ', {:.1f} im/s'.format(batch_size / profiler.means['infer'])
                msg += ', {:.3f} in deepsort'.format(t4-t3)

                print(msg, flush=True)

                profiler.reset()


    
        
        print("Average FPS = {}".format(i/profiler.totals['infer']))
        print("Average tracking time = {}".format(sort_time/i))

    # Gather results from all devices
    if verbose: print('Gathering results...')
    

    detection_results = [torch.cat(r, dim=0) for r in zip(*detection_results)]
    if world > 1:
        for r, result in enumerate(detection_results):
            all_result = [torch.ones_like(result, device=result.device) for _ in range(world)]
            torch.distributed.all_gather(list(all_result), result)
            detection_results[r] = torch.cat(all_result, dim=0)
    
    # import pdb; pdb.set_trace()

    if is_master:

        # Copy buffers back to host
        detection_results = [r.cpu() for r in detection_results]

        # Collect detections
        detections = []
        processed_ids = set()
        count = [0,0,0]
        for scores, boxes, classes, image_id, ratios in zip(*detection_results):

            image_id = image_id.item()
            if image_id in processed_ids:
                continue
            processed_ids.add(image_id)
              
            keep = (scores > 0).nonzero()
            scores = scores[keep].view(-1)
            boxes = boxes[keep, :].view(-1, 4) / ratios
            # classes = classes[keep].view(-1).int()
            # import pdb; pdb.set_trace()
            classes = mapClasses(classes[keep].view(-1).int())

            #print('classes', classes)

            for score, box, cat in zip(scores, boxes, classes):
                x1, y1, x2, y2 = box.data.tolist()
                cat = cat.item()
                if 'annotations' in data_iterator.coco.dataset:
                    

                    cat = data_iterator.coco.getCatIds()[cat]
                    #if cat !=3:
                      #continue
                    #print('cat',cat)
                    count[cat] += 1

                if cat != 0:
                    detections.append(
                    {
                        'image_id': image_id,
                        'score': score.item(),
                        'bbox': [x1, y1, x2 - x1 + 1, y2 - y1 + 1],
                        'category_id': cat,
                        'identity': 1
                    })

        print(count)
        if detections:
            # import pdb;pdb.set_trace()
            # Save detections
            if detections_file and verbose: print('Writing {}...'.format(detections_file))
            detections = { 'annotations': detections }
            detections['images'] = data_iterator.coco.dataset['images']

            if 'categories' in data_iterator.coco.dataset:
                detections['categories'] = [data_iterator.coco.dataset['categories']]
            if detections_file:
                json.dump(detections, open(detections_file, 'w'), indent=4)



            # Evaluate model on dataset
            if 'annotations' in data_iterator.coco.dataset:
                if verbose: print('Evaluating model...')
                with redirect_stdout(None):
                    coco_pred = data_iterator.coco.loadRes(detections['annotations'])
                    coco_eval = COCOeval(data_iterator.coco, coco_pred, 'bbox')
                    coco_eval.evaluate()
                    coco_eval.accumulate()
                coco_eval.summarize()
        else:
            print('No detections!')
