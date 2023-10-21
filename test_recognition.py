import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image

app = FaceAnalysis(allowed_modules=['detection'], providers=['CUDAExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))
img = ins_get_image('tom3')
faces = app.get(img)
print("faces:", faces)

handler = insightface.model_zoo.get_model('models/webface_r50.onnx')
handler.prepare(ctx_id=0)
img = ins_get_image('tom3')
feature = handler.get(img, faces[0])
print("size 0f feature:", len(feature))
print("feature:", feature)
