# import các gói thư viện cần thiết
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation, Flatten, Dense, Dropout
from keras.layers import BatchNormalization, MaxPooling2D
from keras import backend as K

class ShallowNet:
    @staticmethod
    def build(width, height, depth, classes):
        # khởi tạo mô hình
        # height, width, depth: tương ứng 3 chiều của dữ liệu ảnh đầu vào
        # classes: tổng số lớp mà mạng dự đoán, phụ thuộc vào dữ liệu
        # Dữ liệu Animal: 3 lớp; Dữ liệu CIFAR-10: 10 lớp

        model = Sequential()
        inputShape = (height, width, depth)

        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)

        # Định nghĩa mạng CONV => RELU layer
        model.add(Conv2D(32, (3, 3), padding="same",input_shape = inputShape))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.1))


        model.add(Conv2D(32, (3, 3), padding="same",input_shape = inputShape))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.2))


        # Bộ phân lớp sử dụng hàm softmax
        model.add(Flatten())       # Chuyển thành vector
        model.add(Dense(classes))  # Định nghĩa Full Connected layer
        model.add(Dropout(0.3))
        model.add(Activation("softmax"))   # Kích hoạt hàm softmax để phân lớp
        # Trả về model kiến trúc mạng
        return model

