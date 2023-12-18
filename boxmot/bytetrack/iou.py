import numpy as np


def bbox_ious(boxes, query_boxes):
    """
    Parameters
    ----------
    boxes: (N, 4) ndarray of float
    query_boxes: (K, 4) ndarray of float
    Returns
    -------
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    N = boxes.shape[0]
    K = query_boxes.shape[0]
    overlaps = np.zeros((N, K), dtype=np.float32)
    
    for k in range(K):
        box_area = ( # 第二组边界框的面积
            (query_boxes[k, 2] - query_boxes[k, 0] + 1) *
            (query_boxes[k, 3] - query_boxes[k, 1] + 1)
        )
        for n in range(N):
            iw = ( # 交集宽度
                min(boxes[n, 2], query_boxes[k, 2]) -
                max(boxes[n, 0], query_boxes[k, 0]) + 1
            )
            if iw > 0:
                ih = ( # 交集高度
                    min(boxes[n, 3], query_boxes[k, 3]) -
                    max(boxes[n, 1], query_boxes[k, 1]) + 1
                )
                if ih > 0:
                    ua = float( # 并集面积
                        (boxes[n, 2] - boxes[n, 0] + 1) *
                        (boxes[n, 3] - boxes[n, 1] + 1) +
                        box_area - iw * ih
                    )
                    overlaps[n, k] = iw * ih / ua # IoU
    return overlaps

def bbox_gious(boxes, query_boxes):
    """
    Parameters
    ----------
    boxes: (N, 4) ndarray of float
    query_boxes: (K, 4) ndarray of float
    Returns
    -------
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    N = boxes.shape[0]
    K = query_boxes.shape[0]
    overlaps = np.zeros((N, K), dtype=np.float32)
    
    for k in range(K):
        box_area = ( # 第二组边界框的面积
            (query_boxes[k, 2] - query_boxes[k, 0] + 1) *
            (query_boxes[k, 3] - query_boxes[k, 1] + 1)
        )
        for n in range(N):
            iw = ( # 交集宽度
                min(boxes[n, 2], query_boxes[k, 2]) -
                max(boxes[n, 0], query_boxes[k, 0]) + 1
            )
            if iw > 0:
                ih = ( # 交集高度
                    min(boxes[n, 3], query_boxes[k, 3]) -
                    max(boxes[n, 1], query_boxes[k, 1]) + 1
                )
                if ih > 0:
                    ua = float( # 并集面积
                        (boxes[n, 2] - boxes[n, 0] + 1) *
                        (boxes[n, 3] - boxes[n, 1] + 1) +
                        box_area - iw * ih
                    )
                    overlaps[n, k] = iw * ih / ua # IoU

                    # 计算封闭框坐标
                    enclose_x1 = min(boxes[n, 0], query_boxes[k, 0])
                    enclose_y1 = min(boxes[n, 1], query_boxes[k, 1])
                    enclose_x2 = max(boxes[n, 2], query_boxes[k, 2])
                    enclose_y2 = max(boxes[n, 3], query_boxes[k, 3])

                    # 计算封闭框面积
                    enclose_area = (
                        (enclose_x2 - enclose_x1 + 1) *
                        (enclose_y2 - enclose_y1 + 1)
                    )

                    # 计算 GIoU
                    giou = overlaps[n, k] - (enclose_area - ua) / enclose_area
                    overlaps[n, k] = giou
    return overlaps

def bbox_dious(boxes, query_boxes):
    """
    Parameters
    ----------
    boxes: (N, 4) ndarray of float
    query_boxes: (K, 4) ndarray of float
    Returns
    -------
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    N = boxes.shape[0]
    K = query_boxes.shape[0]
    overlaps = np.zeros((N, K), dtype=np.float32)
    
    for k in range(K):
        box_area = ( # 第二组边界框的面积
            (query_boxes[k, 2] - query_boxes[k, 0] + 1) *
            (query_boxes[k, 3] - query_boxes[k, 1] + 1)
        )
        for n in range(N):
            iw = ( # 交集宽度
                min(boxes[n, 2], query_boxes[k, 2]) -
                max(boxes[n, 0], query_boxes[k, 0]) + 1
            )
            if iw > 0:
                ih = ( # 交集高度
                    min(boxes[n, 3], query_boxes[k, 3]) -
                    max(boxes[n, 1], query_boxes[k, 1]) + 1
                )
                if ih > 0:
                    ua = float( # 并集面积
                        (boxes[n, 2] - boxes[n, 0] + 1) *
                        (boxes[n, 3] - boxes[n, 1] + 1) +
                        box_area - iw * ih
                    )
                    overlaps[n, k] = iw * ih / ua # IoU

                    # 计算封闭框坐标
                    enclose_x1 = min(boxes[n, 0], query_boxes[k, 0])
                    enclose_y1 = min(boxes[n, 1], query_boxes[k, 1])
                    enclose_x2 = max(boxes[n, 2], query_boxes[k, 2])
                    enclose_y2 = max(boxes[n, 3], query_boxes[k, 3])

                    # 计算封闭框面积
                    enclose_area = (
                        (enclose_x2 - enclose_x1 + 1) *
                        (enclose_y2 - enclose_y1 + 1)
                    )

                    c = (enclose_x2 - enclose_x1 + 1) ** 2 + (enclose_y2 - enclose_y1 + 1) ** 2 # 勾股定理
                    diagonal = np.sqrt(c) # 封闭框对角线距离
                    center_distance = np.sqrt( # 检测框中心距离
                        ((boxes[n, 0] + boxes[n, 2]) / 2 - (query_boxes[k, 0] + query_boxes[k, 2]) / 2) ** 2 +
                        ((boxes[n, 1] + boxes[n, 3]) / 2 - (query_boxes[k, 1] + query_boxes[k, 3]) / 2) ** 2
                    )
                    
                    # 计算 DIoU
                    diou = overlaps[n, k] - (center_distance ** 2) / (diagonal ** 2 + np.finfo(float).eps)
                    overlaps[n, k] = diou
    return overlaps

def bbox_cious(boxes, query_boxes):
    """
    Parameters
    ----------
    boxes: (N, 4) ndarray of float
    query_boxes: (K, 4) ndarray of float
    Returns
    -------
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    N = boxes.shape[0]
    K = query_boxes.shape[0]
    overlaps = np.zeros((N, K), dtype=np.float32)
    
    for k in range(K):
        box_area = ( # 第二组边界框的面积
            (query_boxes[k, 2] - query_boxes[k, 0] + 1) *
            (query_boxes[k, 3] - query_boxes[k, 1] + 1)
        )
        for n in range(N):
            iw = ( # 交集宽度
                min(boxes[n, 2], query_boxes[k, 2]) -
                max(boxes[n, 0], query_boxes[k, 0]) + 1
            )
            if iw > 0:
                ih = ( # 交集高度
                    min(boxes[n, 3], query_boxes[k, 3]) -
                    max(boxes[n, 1], query_boxes[k, 1]) + 1
                )
                if ih > 0:
                    ua = float( # 并集面积
                        (boxes[n, 2] - boxes[n, 0] + 1) *
                        (boxes[n, 3] - boxes[n, 1] + 1) +
                        box_area - iw * ih
                    )
                    overlaps[n, k] = iw * ih / ua # IoU

                    # 计算封闭框坐标
                    enclose_x1 = min(boxes[n, 0], query_boxes[k, 0])
                    enclose_y1 = min(boxes[n, 1], query_boxes[k, 1])
                    enclose_x2 = max(boxes[n, 2], query_boxes[k, 2])
                    enclose_y2 = max(boxes[n, 3], query_boxes[k, 3])

                    # 计算封闭框面积
                    enclose_area = (
                        (enclose_x2 - enclose_x1 + 1) *
                        (enclose_y2 - enclose_y1 + 1)
                    )

                    c = (enclose_x2 - enclose_x1 + 1) ** 2 + (enclose_y2 - enclose_y1 + 1) ** 2 # 勾股定理
                    diagonal = np.sqrt(c) # 封闭框对角线距离
                    center_distance = np.sqrt( # 检测框中心距离
                        ((boxes[n, 0] + boxes[n, 2]) / 2 - (query_boxes[k, 0] + query_boxes[k, 2]) / 2) ** 2 +
                        ((boxes[n, 1] + boxes[n, 3]) / 2 - (query_boxes[k, 1] + query_boxes[k, 3]) / 2) ** 2
                    )
                    
                    diou = overlaps[n, k] - (center_distance ** 2) / (diagonal ** 2 + np.finfo(float).eps)
                    # v 度量长宽比的相似性
                    v = (4 / (np.pi ** 2)) * np.square(np.arctan((boxes[n, 2] - boxes[n, 0] + 1) / (boxes[n, 3] - boxes[n, 1] + 1)) - np.arctan((query_boxes[k, 2] - query_boxes[k, 0] + 1) / (query_boxes[k, 3] - query_boxes[k, 1] + 1)))
                    # 计算 alpha 权重函数
                    alpha = v / (1 - overlaps[n, k] + v)
                    
                    # 计算 CIOU
                    ciou = diou - alpha * v
                    overlaps[n, k] = ciou
    return overlaps