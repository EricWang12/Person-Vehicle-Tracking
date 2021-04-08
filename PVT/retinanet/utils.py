import os.path
import time
import json
import warnings
import signal
from datetime import datetime
from contextlib import contextmanager
from PIL import Image, ImageDraw, ImageFont
import requests


def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)


def show_MOT(path, results):
    'Show image with drawn detections'

    for image, result, annotation in results:
        im = Image.open(image).convert('RGBA')
        overlay = Image.new('RGBA', im.size, (255,255,255,0))
        draw = ImageDraw.Draw(overlay)
        # detections.sort(key=lambda d: d['score'])
        # import pdb;pdb.set_trace()

        for detection in result:
            if detection[5] == 0:
                continue
            box = detection[:4].tolist()
            alpha = int(255)
            color = compute_color_for_labels(detection[4])
            # font = ImageFont.truetype(size=8)
            draw.rectangle(box, outline=(0, color[1], color[2], alpha))
            draw.text((box[0]+2, box[1]), 'label:{}'.format(detection[5]), 
                fill=(0, color[1], color[2], alpha))
            draw.text((box[0]+2, box[1]+10), 'ID: {}'.format(detection[4]), 
                fill=(0, color[1], color[2], alpha))
        
        
        for anno in annotation:
            box = anno['bbox']
            alpha = int(123)
            target_id = anno['target_id']
            color = compute_color_for_labels(target_id)

            draw.rectangle(box, outline=(color[0], 0, 0, alpha))
            draw.text((box[0]+2, box[1]), 'label:{}'.format(anno['category_id']),
                fill=(color[0], 0, 0, alpha))
            draw.text((box[0]+2, box[1]+10), 'ID: {}'.format(target_id),
                fill=(color[0], 0, 0, alpha))

        im = Image.alpha_composite(im, overlay)
        im = im.convert('RGB')
        im.save(os.path.join(path, os.path.basename(image)))
        # import pdb;pdb.set_trace()
        
    return im

def show_detections(path, detections):
    'Show image with drawn detections'

    for image, detections, annotations in detections:
        im = Image.open(image).convert('RGBA')
        overlay = Image.new('RGBA', im.size, (255,255,255,0))
        draw = ImageDraw.Draw(overlay)
        detections.sort(key=lambda d: d['score'])
        for detection in detections:
            box = detection['bbox']
            alpha = int(detection['score'] * 255)
            draw.rectangle(box, outline=(0, 255, 0, alpha))
            draw.text((box[0]+2, box[1]), '[{}]'.format(detection['category_id']),
                fill=(0, 255, 0, alpha))
            draw.text((box[0]+2, box[1]+10), '{:.2}'.format(detection['score']),
                fill=(0, 255, 0, alpha))
        for anno in annotations:
            box = anno['bbox']
            alpha = int(123)
            draw.rectangle(box, outline=(255, 0, 0, alpha))
            draw.text((box[0]+2, box[1]), '[{}]'.format(detection['category_id']),
                fill=(255, 0, 0, alpha))
            draw.text((box[0]+2, box[1]+10), '{:.2}'.format(detection['score']),
                fill=(255, 0, 0, alpha))

        im = Image.alpha_composite(im, overlay)
        im = im.convert('RGB')
        im.save(os.path.join(path, os.path.basename(image)))
        # import pdb;pdb.set_trace()
        
    return im

def save_detections(path, detections):
    print('Writing detections to {}...'.format(os.path.basename(path)))
    with open(path, 'w') as f:
        json.dump(detections, f)

@contextmanager
def ignore_sigint():
    handler = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, handler)

class Profiler(object):
    def __init__(self, names=['main']):
        self.names = names
        self.lasts = { k: 0 for k in names }
        self.totals = self.lasts.copy()
        self.counts = self.lasts.copy()
        self.means = self.lasts.copy()
        self.reset()

    def reset(self):
        last = time.time()
        for name in self.names:
            self.lasts[name] = last
            self.totals[name] = 0
            self.counts[name] = 0
            self.means[name] = 0

    def start(self, name='main'):
        self.lasts[name] = time.time()

    def stop(self, name='main'):
        self.totals[name] += time.time() - self.lasts[name]
        self.counts[name] += 1
        self.means[name] = self.totals[name] / self.counts[name]

    def bump(self, name='main'):
        self.stop(name)
        self.start(name)

def post_metrics(url, metrics):
    try:
        for k, v in metrics.items():
            requests.post(url,
                data={ 'time': int(datetime.now().timestamp() * 1e9), 
                        'metric': k, 'value': v })
    except Exception as e:
        warnings.warn('Warning: posting metrics failed: {}'.format(e))

