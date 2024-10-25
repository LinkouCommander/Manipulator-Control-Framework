# import math

# # 定義兩個向量的起始和結束點
# point1_A = (0, 2)
# point2_A = (0, 0)
# point1_B = (2, 0)
# point2_B = (0, 0)

# def get_radians(point1, point2):
#     # 計算向量內積
#     if point1 == (0, 0) or point2 == (0, 0):
#         return 0
    
#     x1, y1 = point1
#     x2, y2 = point2
    
#     dot_product = x1 * x2 + y1 * y2

#     # 計算向量的長度（模）
#     magnitude_A = math.sqrt(x1**2 + y1**2)
#     magnitude_B = math.sqrt(x2**2 + y2**2)

#     # 確保內積比值在 [-1, 1] 之間
#     cos_theta = dot_product / (magnitude_A * magnitude_B)
#     cos_theta = max(-1, min(1, cos_theta))  # 將 cos_theta 限制在 [-1, 1]

#     # 計算夾角的弧度
#     angle_radians = math.acos(dot_product / (magnitude_A * magnitude_B))
    

#     return angle_radians

# print("radians = ", get_radians(point1_A, point1_B))


import cv2

# 開啟視訊串流
cap = cv2.VideoCapture(0)

# 取得 FPS (每秒幀數)
fps = cap.get(cv2.CAP_PROP_FPS)
print("Frames per second:", fps)

# 記得在最後釋放視訊串流
cap.release()
