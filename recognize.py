import argparse
import glob
import os.path

import numpy as np
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image

import cv2

app = FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=0, det_size=(640, 640))


def get_face_db_features():
    face_db_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'face_db/TBBT')
    img_gen = glob.glob(face_db_dir + '/*')
    args = argparse.Namespace()
    assert len(img_gen) != 0, '人脸库图片数量为0'
    face_db_list = []
    for img_path in img_gen:
        args.img = img_path
        image = cv2.imread(args.img)
        face = app.get(image)
        assert len(face) == 1, f"图片:{img_path}人脸数量不唯一"
        name = os.path.splitext(os.path.basename(img_path))[0]
        face_db_list.append(dict(name=name, feat=face[0].normed_embedding))
    return face_db_list


face_db_list = get_face_db_features()


def recognize(image):
    recognition_face_list = []
    faces = app.get(image)
    for face in faces:
        recognition_face_list.append(dict(bbox=face.bbox, feat=face.normed_embedding))
    result = []
    for face_db in face_db_list:
        temp_list = []
        for idx, face in enumerate(recognition_face_list):
            name = face_db['name']
            feat1 = face_db['feat']
            feat2 = face['feat']
            sim = np.dot(feat1, feat2)
            # temp_list.append(dict(idx=idx, name=name, sim=sim))
            if sim >= 0.22:
                temp_list.append(dict(idx=idx, name=name, sim=sim))
        if temp_list:
            result.append(temp_list)
    if not result:
        return []
    for data in result:
        data.sort(key=lambda x: x['sim'], reverse=True)
    rimg = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    for row in result:
        face_info = row[0]
        idx = face_info["idx"]
        box = recognition_face_list[idx]["bbox"].astype(np.int32)
        color = (0, 0, 255)
        cv2.rectangle(rimg, (box[0], box[1]), (box[2], box[3]), color, 2)
        sim_bar = int((box[3] - box[1]) * (int(face_info['sim'] * 100) / 100))
        cv2.rectangle(rimg, (box[2], (box[1] + (box[3] - box[1])) - sim_bar), (box[2] + 10, box[3]), color, -1)  # 绘制竖
        cv2.putText(rimg, face_info['name'], (box[0], box[1] - 10), font, 1, color, 2)
    return rimg


if __name__ == '__main__':
    cap = cv2.VideoCapture('video/TBBT.S12E24.mp4')
    while True:
        # print('cap',cap.read()[1])
        # print('ins',ins_get_image('tbbt'))
        # break
        img = cap.read()[1]
        rimg = recognize(img)
        if len(rimg) != 0:
            cv2.imshow('', rimg)
        else:
            cv2.imshow('', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
