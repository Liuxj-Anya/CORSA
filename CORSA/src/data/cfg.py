CLASS_NUM = 3

"anchor box是对coco数据集聚类获得"
ANCHORS_GROUP_KMEANS = {
    28: [[10, 13], [16, 30], [33, 23]],
    14: [[30, 61], [62, 45], [59, 119]],
    7: [[116, 90], [156, 198], [373, 326]]}  # 大特征图小感受野，小特征图大感受野

ANCHORS_GROUP = {
    7: [[360, 360], [360, 180], [180, 360]],
    14: [[180, 180], [180, 90], [90, 180]],
    28: [[90, 90], [90, 45], [45, 90]]}  # 根据经验指定框的建议框

ANCHORS_GROUP_AREA = {
    7: [w * h for w, h in ANCHORS_GROUP[7]],  # 建议框的面积（与实际框的面积可以求IOU值）
    14: [w * h for w, h in ANCHORS_GROUP[14]],
    28: [w * h for w, h in ANCHORS_GROUP[28]],
}

if __name__ == '__main__':

    for feature_size, anchors in ANCHORS_GROUP.items():
        print(feature_size)
        print(anchors)
    for feature_size, anchor_area in ANCHORS_GROUP_AREA.items():
        print(feature_size)
        print(anchor_area)