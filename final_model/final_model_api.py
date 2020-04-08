#%% install necessary package
# !pip install facenet-pytorch
# !pip install mmcv
# !pip install face-alignment

# %%
# REVIEW load all necessary package
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt # for plotting
import torch.optim as optim #for gradient descent
from torchvision import datasets, transforms
import torchvision.models
import mmcv, cv2
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image, ImageDraw
from skimage import io
from facenet_pytorch import MTCNN, InceptionResnetV1
import face_alignment
import gc
from pprint import pprint

torch.manual_seed(1) # set the random seed
device = 'cpu'

# %% 
# REVIEW Our Model
classes = ['0: Neutral', '1: Happiness', 
           '2: Sadness', '3: Surprise', 
           '4: Fear', '5: Disgust', '6: Anger']
           
tensor_path = '../Collection of models/googlenet_tensor/model_FaceRec_bs128_lr0.001_epoch6'
HA_tensor_path = "../Collection of models/binary_model/model_HA_distinguisher_bs128_lr0.0001_epoch3"
six_nary_tensor_path = "../Collection of models/6nary_model/model_FaceRec_bs128_lr0.0001_epoch2"

model = torchvision.models.googlenet(pretrained=True)
model.fc = nn.Linear(1024, 7)
torch.set_flush_denormal(True)
model.load_state_dict(torch.load(tensor_path, map_location=torch.device(device)))
model.eval()

version = "0.91"

def run_model(inputs):
    """
    run with our final model
    input:
        inputs [x, 3, 224, 224] input picture
    output:
        [x, 7] output predictions (with softmax)
    """
    with torch.no_grad():
        out = model(inputs)
        out = torch.nn.functional.softmax(out, dim = 1)
        del inputs
        return out

class MyAlexNet(nn.Module):

    def __init__(self, num_classes=1):
        super(MyAlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 100),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(100, 1),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def my_alexnet(pretrained=False, progress=True, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = MyAlexNet(**kwargs)
    return model

HA_distinguisher = my_alexnet()

HA_distinguisher.load_state_dict(torch.load(HA_tensor_path, map_location=torch.device(device)))
HA_distinguisher.eval()

six_nary_distinguisher = torchvision.models.googlenet(pretrained=True)
six_nary_distinguisher.fc = nn.Linear(1024, 6)
six_nary_distinguisher.load_state_dict(torch.load(six_nary_tensor_path, map_location=torch.device(device)))
six_nary_distinguisher.eval()

def run_modelv2(inputs):
    with torch.no_grad():
        out_ha = HA_distinguisher(inputs)
        out_ha = np.array(torch.sigmoid(out_ha))

        out_six = six_nary_distinguisher(inputs)
        out_six = np.array(torch.nn.functional.softmax(out_six, dim = 1))

        result = np.array([[0.] * 7] * len(inputs))
        result[:, 0] = out_six[:, 0] * (np.ones(len(inputs)) - out_ha[:, 0])
        result[:, 1] = out_ha[:, 0]
        for i in range(2, 7):
            result[:, i] = out_six[:, i-1] * (np.ones(len(inputs)) - out_ha[:, 0])

        del inputs
        return torch.tensor(result)

def run_each_img(input):
    with torch.no_grad():
        HA_test = (HA_distinguisher(input) > 0.0).squeeze().long()
        if HA_test == 1:
          return 1
        six_nary_test = six_nary_distinguisher(input)
        pred = six_nary_test.max(1, keepdim=True)[1]
        if (pred).cpu().numpy()[0][0] == 0:
          return 0
        return (pred+1).cpu().numpy()[0][0]


# %%
# REVIEW load preprocessed models
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device=device)
mtcnn = MTCNN(keep_all=True, device=device)

def get_faces(img, im_size=224, debug=False):
    faces = []
    locs = []
    img = Image.fromarray(np.uint8(img))
    boxes, _ = mtcnn.detect(img)
    if not isinstance(boxes, np.ndarray): return (faces, locs)
    if debug: pprint(boxes.shape)
    for i, box in enumerate(boxes):
        plt.subplot((boxes.shape[0]+4)/5, 5, i+1)
        plt.axis('off')

        if debug: pprint(box)

        left = box[0]
        top = box[1]
        w = box[2] - box[0]
        h = box[3] - box[1]
        size = max(w, h)
        right = left + size
        bottom = top + size
        loc = (left, top, right, bottom)

        cropped = img.crop(loc)
        resized = cropped.resize((im_size, im_size), Image.ANTIALIAS)
        np_img = np.array(resized)
        if np_img.shape[:2] != (im_size, im_size): 
            # Size is not actually wrong, but we may be out of the original picture
            print("Wrong Size")
        else:
            faces.append(np_img)
            locs.append(loc)
        if debug: plt.imshow(np_img)
    
    return (faces, locs)

def relative_loc(xy_pair, mins, maxs):
    # mins = np.array([mins[1], mins[0]])
    # maxs = np.array([maxs[1], maxs[0]])
    orig_x, orig_y  = tuple(xy_pair)
    w = maxs[0] - mins[0]
    h = maxs[1] - mins[1]
    x = (orig_x - mins[0]) / w
    y = (orig_y - mins[1]) / h
    return np.array([x, y])

def get_landmarks(img, im_size=224, debug=False):
    with torch.no_grad():
        preds = fa.get_landmarks(img)
    if not isinstance(preds, list) or len(preds) == 0: return [], []
    np_preds = np.asarray(preds).astype('float64')
    if debug: pprint(np_preds.shape)
    discon = [0, 8, 9, 17, 22, 27, 31, 36, 42, 48, 60, 68]

    land_ims = []
    land_xys = []
    for landmarks in np_preds:
        mins = landmarks.min(0)
        maxs = landmarks.max(0)
        points = [relative_loc(xy, mins, maxs) for xy in landmarks]
        canvas = np.zeros(shape=[im_size, im_size, 3], dtype=np.uint8) # black image
        for (k, p) in enumerate(points):
            if k not in discon:
                cv2.line(canvas, tuple((220*p0+2).astype(np.int)), 
                         tuple((220*p+2).astype(np.int)), 
                         (3*k+20,172,3*(69-k), 1), 1)
            elif k in [36, 42, 48, 60]:
                t = discon[discon.index(k) + 1] - 1
                p0 = points[t]
                cv2.line(canvas, tuple((220*p0+2).astype(np.int)), 
                         tuple((220*p+2).astype(np.int)), 
                         (3*k+20,172,3*(69-k), 1), 1)
            p0 = p

        for (k, p) in enumerate(points):
            cv2.circle(canvas, tuple((220*p+2).astype(np.int)), 2, (3*k+40,255,3*(69-k)), 1)
        cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        if debug: plt.imshow(canvas)
        land_ims.append(canvas)
        land_xys.append(landmarks)
    
    return land_ims, land_xys

def preprocess_input(img, debug=False):
    results = []
    res_index = 0

    torch.cuda.empty_cache()
    gc.collect()
    face_imgs, face_boxes = get_faces(img)
    torch.cuda.empty_cache()
    gc.collect()
    land_imgs, land_xys = [], []
    for i, box in enumerate(face_boxes):
        left = box[0]
        top = box[1]
        w = box[2] - box[0]
        h = box[3] - box[1]

        left = max(0, int(left - w * 0.1))
        top = max(0, int(top - h * 0.1))
        right = min(img.shape[1], int(left + w * 1.2))
        btm = min(img.shape[0], int(top + h * 1.2))

        t_img = img[top : btm, left : right]
        print(box, (left, right, top, btm), img.shape, (w, h))

        imgs, xys = get_landmarks(cv2.resize(t_img, (224,224)))
        # land_imgs.extend(imgs)
        # land_xys.extend(xys)

        if len(xys) > 0:
            results.append((res_index, face_imgs[i], imgs[-1], box, xys[-1]))
            res_index += 1

        # print(len(xys))
        del imgs, t_img, xys
        gc.collect()

    torch.cuda.empty_cache()
    gc.collect()

    # since the results are from two different sources
    # there may be faces without landmarks or vice versa


    # check_point_indices = [27, 48, 54, 57] # check if these points are inside a face
    # res_index = 0
    # for i, box in enumerate(face_boxes):
    #     for j, xys in enumerate(land_xys):
    #         insides = [box[0] <= xys[k][0] <= box[2] and box[1] <= xys[k][1] <= box[3] for k in check_point_indices]
    #         inside = all(insides)
    #         if debug: print((i,j), box, inside)
    #         if inside:
    #             results.append((res_index, face_imgs[i], land_imgs[j], face_boxes[i], land_xys[j]))
    #             res_index += 1

    return results


# %%
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

from time import time
def run_api(img, crop=True, landmarks=False):
    s_time = time()
    
    if crop:
        faces, face_locs = get_faces(img)
        faces = np.array(faces)
    else:
        faces = [img]
        face_locs = [(0, 0, *img.shape[:2])]

    # normalize input
    inputs = []
    for face in faces:
        inputs.append(transform(face))

    if len(inputs) == 0:
        return ({
            'model_version': version,
            'count': len(faces),
            'time': time() - s_time,
            'faces': []
        })

    inputs = torch.stack(inputs)

    result = run_modelv2(inputs)
    pred = result.max(1, keepdim=True)[1]
    # pred = run_each_img(inputs)
    # for i, p in enumerate(pred):
    #     print('Result: {}, confidence: {:5.2f}%'.format(classes[p], 100 * float(result[i][p])))

    face_results = []
    for i, f in enumerate(faces):
        face_results.append({})
        face_results[i]['locations'] = face_locs[i]
        face_results[i]['class'] = classes[pred[i]]
        face_results[i]['details'] = tuple(result[i].tolist())
        face_results[i]['confidence'] = 100 * float(result[i][pred[i]])

    output = {
        'model_version': version,
        'count': len(faces),
        'time': time() - s_time,
        'faces': face_results
    }

    return output

if __name__ == "__main__":
    import wget
    print("Test started")
    url = 'https://images.unsplash.com/photo-1523422914648-23a7478946cb?ixlib=rb-1.2.1&ixid=eyJhcHBfaWQiOjEyMDd9&auto=format&fit=crop&w=1385&q=80'
    filename = wget.download(url, out='pic.jpg')
    img = io.imread(filename)
    print()
    pprint(run_api(img))
