import skimage.feature as skf
import pandas as pd
import numpy as np
import cv2 as cv
import joblib
SVM_DIM = 392
CNN_DIM = 216
RF_DIM = 128
DISTANCE = 10
THETA = 90


class PredictionHelper:
    def __init__(self):
        self.path_to_image = None
        self.coocurrence_matrix = None
        self.features = np.zeros(5)
        self.scaled_features = None
        self.cnn_img = []
        self.scaler = joblib.load('artifacts/scaler2.joblib')
        self.svm_model = joblib.load('artifacts/svm2.joblib')
        self.cnn_model = joblib.load('artifacts/cnn1.joblib')

    def set_image_path(self, path):
        self.path_to_image = path

    def glcm_feature(self, feature_name):
        feature = skf.graycoprops(self.coocurrence_matrix, feature_name)
        result = np.average(feature)
        return result

    def preprocess(self):
        img = cv.imread(self.path_to_image)
        temp_img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        gray_img = cv.cvtColor(temp_img, cv.COLOR_RGB2GRAY)
        thresh_img = cv.adaptiveThreshold(
            gray_img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 11, 3)

        cnts = cv.findContours(
            thresh_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        cnts = sorted(cnts, key=cv.contourArea, reverse=True)
        for c in cnts:
            x, y, w, h = cv.boundingRect(c)
            thresh_img = img[y:y+h, x:x+w]
            temp_img = temp_img[y:y+h, x:x+w]
            break

        img = cv.resize(thresh_img, (SVM_DIM, SVM_DIM))
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        temp_img = cv.resize(temp_img, (CNN_DIM, CNN_DIM))

        self.cnn_img = np.expand_dims(temp_img, axis=0)

        self.coocurrence_matrix = skf.graycomatrix(img, [DISTANCE], [THETA], 256,
                                                   symmetric=True, normed=True)
        self.features[0] = self.glcm_feature('contrast')
        self.features[1] = self.glcm_feature('homogeneity')
        self.features[2] = self.glcm_feature('energy')
        self.features[3] = self.glcm_feature('correlation')
        self.features[4] = self.glcm_feature('dissimilarity')

    def preprocess(self):
        pass

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

    def run(self):
        self.preprocess()
        self.scale_data()
        r_svm = self.predict_svm()
        r_cnn = self.predict_cnn()
        return [round(r_svm[0][1]*100, 4), round(r_cnn[1]*100, 4)]
