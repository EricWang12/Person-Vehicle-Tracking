import json
import os
from PIL import Image
import shutil
from tqdm import tqdm


# original
dataDir = '/home/user/EricWang/HSL/HSL-interview/dataset/VD-test/'
convertClass = False
Ratio = 0.1
save_video = True





dirs = os.listdir( dataDir + "sequences/" )   
input_path =  dataDir + "sequences/"
output_path = dataDir + "sequences-{}/".format(Ratio)
if os.path.isdir(output_path):
    shutil.rmtree(output_path)
os.mkdir(output_path)


def mapToPV(id):
    if id in [1,2]:
        return 1
    elif id in range(3,11):
        return 2
    else:
        return 0

# new
# dsDir = '/home/user/EricWang/HSL/HSL-interview/dataset/VD-test/'
# train set
# dataType = 'train'
# val set


# Change working directory to COCO dataset path
# os.mkdir(dataDir)
os.chdir(dataDir)

data = {}
data['info'] = {"description": "COCO 2015 Dataset", "url": "http://cocodataset.org", "version": "1.0",
                "year": 2015, "contributor": "COCO Consortium", "date_created": "2015/09/01"}
data['licenses'] = [{"url": "http://creativecommons.org/licenses/by-nc-sa/2.0/",
                     "id": 1, "name": "Attribution-NonCommercial-ShareAlike License"}]
images = []
annotations = []
videoList = os.listdir(dataDir+'sequences')
# breakpoint()
videoList.sort()
videoIndex = -1
anid = 1
numTrunc = 0
numOcclu = 0
# Number of objects that are neither truncated nor occluded
numComplete = 0
numType = [0] * 12
imageIndex = -1
frameMap = {}
frameStep = 1



def resize( ratio):

    for dir in  tqdm(dirs):
        for item in tqdm(sorted(os.listdir(input_path+dir)), leave = False):
            im_file = input_path+dir+'/' +item
            if os.path.isfile(im_file):
                image = Image.open(im_file, 'r')
                # file_path, extension = os.path.splitext(path+item)
                # print(im_file)

                new_image_height = int(image.size[0] / (1/ratio))
                new_image_length = int(image.size[1] / (1/ratio))

                image = image.resize((new_image_height, new_image_length), Image.ANTIALIAS)
                if not os.path.isdir(output_path + dir):
                    os.mkdir(output_path+dir)
                image.save(os.path.join(output_path + dir , item), "JPEG")

                # import pdb; pdb.set_trace()
                # print(output_path + dir + item)
if Ratio != 1 and save_video:
    resize( Ratio)

output_anno_path = dataDir+'annotations-{}/'.format(Ratio)
if os.path.isdir(output_anno_path):
    shutil.rmtree(output_anno_path)
os.mkdir(output_anno_path)


for vid in videoList:
    outfile = open('annotation{}.json'.format('-'+ str(Ratio) if Ratio != 1. else '' ), 'w')
    videoIndex += 1
    frameList = os.listdir(dataDir+'sequences/'+vid)
    frameList.sort()
    for i in range(0, len(frameList), frameStep):
        frame = frameList[i]
        imageIndex+=1
        im = Image.open(dataDir+'sequences/'+vid+'/'+frame)
        item = {}
        item['license'] = 1
        item['file_name'] = vid+'/'+frame
        item['coco_url'] = 'http://images.cocodataset.org/val2017/'+frame
        item['height'], item['width'] = im.size
        item['date_captured'] = '2013-11-15 00:09:17'
        item['flickr_url'] = ''
        item['id'] = imageIndex
        frameMap[videoIndex*10000+i+1] = imageIndex
        images.append(item)


        # shutil.copy2(dataDir+'sequences/'+vid+'/'+frame, dataDir+'sequences-{}/'.format(Ratio)+vid+'/'+frame)
    annoFile = open(dataDir+'annotations/'+vid+'.txt', 'r')
    lines = annoFile.readlines()
    for i in range(len(lines)):
        line = lines[i]
        # print(line)
        if len(line) < 2:
            continue
        anitem = {'segmentations': [[]]}
        nums = [int(x) for x in line.split(',')]
        if (nums[0]-1)%frameStep is not 0:
            continue
        anitem['area'] = nums[4]*nums[5]
        anitem['target_id'] = nums[1]
        anitem['iscrowd'] = 0
        anitem['image_id'] = frameMap[videoIndex*10000+nums[0]]
        anitem['bbox'] = [i * Ratio for i in nums[2:6]]
        if convertClass:
            nums[7] = mapToPV(nums[7])
        else:
            nums[2:6] = [int(i * Ratio) for i in nums[2:6]]
        anitem['category_id'] = nums[7]
        anitem['id'] = anid
        anid += 1
        anitem['trunc'] = nums[8]
        anitem['occlu'] = nums[9]
        if nums[8] == 0 and nums[9] == 0:
            numComplete += 1
        if nums[8] == 1:
            numTrunc += 1
        if nums[9] == 1:
            numOcclu += 1
        numType[nums[7]] += 1
        annotations.append(anitem)

        lines[i] =  ''.join([str(s) + ',' for s in nums[:-1]])
        lines[i] += str(nums[-1]) + '\n'

    # breakpoint()
    count = [0,0,0]
    for l in lines:
        nums = [int(x) for x in l.split(',')] 
        count[nums[7]] += 1
    print(count)
    # breakpoint()

    if Ratio != 1:


        annoFile = open(output_anno_path +vid+'.txt', 'w+')
        annoFile.writelines(lines)

    


data['images'] = images
data['annotations'] = annotations
data['categories'] = [{'supercategory': 'other', 'id': 0, 'name': 'other'},
                      {'supercategory': 'person', 'id': 1, 'name': 'people'},
                      {'supercategory': 'vehicle', 'id': 2, 'name': 'vehicle'}
                      ]
print(numComplete/anid)
print(numTrunc/anid)
print(numOcclu/anid)
print(numType)
json.dump(data, outfile, indent=4)

