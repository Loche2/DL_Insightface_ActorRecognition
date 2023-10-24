import cv2
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image

app = FaceAnalysis(providers=['CUDAExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))
img = ins_get_image('image350')
faces = app.get(img)
print('face:', faces)
rimg = app.draw_on(img, faces)
cv2.imwrite("./image350.jpg", rimg)
