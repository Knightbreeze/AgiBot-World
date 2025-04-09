import lmdb
import cv2
import numpy as np

# 指定 LMDB 文件路径
env = lmdb.open('/fs-computility/efm/shared/datasets/Banana/kongweijie/Data/Real_Filt/1000/0000000/lmdb', readonly=True, lock=False)

# 创建事务（transaction）
with env.begin() as txn:
    # 遍历所有键值对
    for key, value in txn.cursor():
        # 假设键是字符串（根据实际情况解码）
        key_str = key.decode('utf-8')
        
        # 假设值是图像字节流（示例）
        # 使用 OpenCV 解码图像
        image = cv2.imdecode(np.frombuffer(value, dtype=np.uint8), cv2.IMREAD_COLOR)
        
        # 或者直接处理二进制数据（如文本、序列化对象）
        print(f"Key: {key_str}, Value Length: {len(value)}")

# 关闭环境
env.close()
