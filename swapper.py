import cv2
import insightface
from insightface.app import FaceAnalysis

face_analyzer = FaceAnalysis(name='buffalo_l')
face_analyzer.prepare(ctx_id=0, det_size=(640, 640))
face_swapper = insightface.model_zoo.get_model('models/inswapper_128.onnx', download=True, download_zip=True)

source_img = cv2.imread('face_db/pyy.png')
source_faces = face_analyzer.get(source_img)
source_faces = sorted(source_faces, key=lambda x: x.bbox[0])
assert len(source_faces) == 1
source_face = source_faces[0]

cap = cv2.VideoCapture(0)
while True:
    # 从摄像头读取一帧数据
    ret, target_img = cap.read()

    # 处理目标图片
    target_faces = face_analyzer.get(target_img)
    target_faces = sorted(target_faces, key=lambda x: x.bbox[0])
    for target_face in target_faces:
        target_img = face_swapper.get(target_img, target_face, source_face, paste_back=True)

    # 显示处理后的视频帧
    cv2.imshow('', target_img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
