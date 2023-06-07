import skimage.feature as skf
import pandas as pd
import numpy as np
import cv2 as cv
import joblib
import tensorflow as tf
from tensorflow import keras
from keras import backend as K
from skimage.filters import sobel
from keras.utils import get_custom_objects
SVM_DIM = 392
CNN_DIM = 216
RF_DIM = 128
DISTANCE = 10
THETA = 90


class PredictionHelper:
    def __init__(self):
        def mish(x):
            return tf.keras.layers.Lambda(lambda x: x*K.tanh(K.softplus(x)))(x)

        get_custom_objects().update({'mish': mish})

        self.path_to_image = None
        self.coocurrence_matrix = None
        self.features = np.zeros(5)
        self.scaled_features = None
        self.cnn_img = []
        self.scaler = joblib.load('artifacts/scaler2.joblib')
        self.svm_model = joblib.load('artifacts/svm2.joblib')
        self.cnn_model = keras.models.load_model('artifacts/cnn1.h5')
        self.rf_model = joblib.load('artifacts/ene.joblib')

    def set_image_path(self, path):
        self.path_to_image = path

    def glcm_feature(self, feature_name):
        feature = skf.graycoprops(self.coocurrence_matrix, feature_name)
        result = np.average(feature)
        return result

    def rf_feature_extractor(self, input_image):
        df = pd.DataFrame()

        img = input_image.copy()

        num = 1
        kernels = []
        for theta in range(2):
            theta = theta / 4. * np.pi
            for sigma in (1, 3):
                lamda = np.pi / 4
                gamma = 0.5
                gabor_label = 'Gabor' + str(num)
                kernel = cv.getGaborKernel(
                    (9, 9), sigma, theta, lamda, gamma, 0, ktype=cv.CV_32F)
                kernels.append(kernel)
                fimg = cv.filter2D(img, cv.CV_8UC3, kernel)
                filtered_img = fimg.reshape(-1)
                df[gabor_label] = filtered_img
                num += 1

        edge_sobel = sobel(img)
        edge_sobel1 = edge_sobel.reshape(-1)
        df['Sobel'] = edge_sobel1

        return df

    def preprocess(self):
        img = cv.imread(self.path_to_image)
        rf_img = cv.resize(img, (RF_DIM, RF_DIM))
        rf_img = rf_img / 255.0

        rf_features = self.rf_feature_extractor(rf_img)
        rf_features = np.expand_dims(rf_features, axis=0)
        rf_features = np.reshape(rf_features, (1, -1))
        pred_rf = self.rf_model.predict(rf_features)

        if pred_rf[0] == 'non eye':
            return False

        color_img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        gray_img = cv.cvtColor(color_img, cv.COLOR_RGB2GRAY)
        thresh_img = cv.adaptiveThreshold(
            gray_img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 11, 3)

        cnts = cv.findContours(
            thresh_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        cnts = sorted(cnts, key=cv.contourArea, reverse=True)
        for c in cnts:
            x, y, w, h = cv.boundingRect(c)
            gray_img = gray_img[y:y+h, x:x+w]
            color_img = color_img[y:y+h, x:x+w]
            break

        gray_img = cv.resize(gray_img, (SVM_DIM, SVM_DIM))
        color_img = cv.resize(color_img, (CNN_DIM, CNN_DIM))
        color_img = color_img / 255.0

        self.cnn_img = np.expand_dims(color_img, axis=0)

        self.coocurrence_matrix = skf.graycomatrix(gray_img, [DISTANCE], [THETA], 256,
                                                   symmetric=True, normed=True)
        self.features[0] = self.glcm_feature('contrast')
        self.features[1] = self.glcm_feature('homogeneity')
        self.features[2] = self.glcm_feature('energy')
        self.features[3] = self.glcm_feature('correlation')
        self.features[4] = self.glcm_feature('dissimilarity')

        return True

    def scale_data(self):
        data = pd.DataFrame([self.features], columns=[
                            'contrast', 'homogeneity', 'energy', 'correlation', 'dissimilarity'])
        self.scaled_features = self.scaler.transform(data)

    def predict_svm(self):
        result = self.svm_model.predict_proba(self.scaled_features)
        return result

    def predict_cnn(self):
        result = self.cnn_model.predict(self.cnn_img)
        return result

    def run(self) -> dict[str, float]:
        is_eye = self.preprocess()

        if not is_eye:
            return {"eye": is_eye, "svm": 0, "cnn": 0}

        self.scale_data()
        r_svm = self.predict_svm()
        r_cnn = self.predict_cnn()
        return {"eye": is_eye, "svm": round(r_svm[0][1]*100, 4), "cnn": round(r_cnn[0][1]*100, 4)}
