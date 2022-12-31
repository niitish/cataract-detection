import skimage.feature as skf
import pandas as pd
import numpy as np
import cv2 as cv
import joblib
HEIGHT = 392
WIDTH = 392
DISTANCE = 10
THETA = 90


class PredictionHelper:
    def __init__(self, path):
        self.path_to_image = path
        self.coocurrence_matrix = None
        self.features = np.zeros(5)
        self.scaled_features = None

    def glcm_feature(self, feature_name):
        feature = skf.graycoprops(self.coocurrence_matrix, feature_name)
        result = np.average(feature)
        return result

    def preprocess(self):
        img = cv.imread(self.path_to_image)
        test_img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        gray_img = cv.cvtColor(test_img, cv.COLOR_RGB2GRAY)
        thresh_img = cv.adaptiveThreshold(
            gray_img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 11, 3)

        cnts = cv.findContours(
            thresh_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        cnts = sorted(cnts, key=cv.contourArea, reverse=True)
        for c in cnts:
            x, y, w, h = cv.boundingRect(c)
            thresh_img = img[y:y+h, x:x+w]
            break

        img = cv.resize(thresh_img, (WIDTH, HEIGHT))
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        self.coocurrence_matrix = skf.graycomatrix(img, [DISTANCE], [THETA], 256,
                                                   symmetric=True, normed=True)
        self.features[0] = self.glcm_feature('contrast')
        self.features[1] = self.glcm_feature('homogeneity')
        self.features[2] = self.glcm_feature('energy')
        self.features[3] = self.glcm_feature('correlation')
        self.features[4] = self.glcm_feature('dissimilarity')

    def scale_data(self):
        scaler = joblib.load('scaler1.joblib')
        data = pd.DataFrame([self.features], columns=[
                            'contrast', 'homogeneity', 'energy', 'correlation', 'dissimilarity'])
        self.scaled_features = scaler.transform(data)

    def predict(self):
        model = joblib.load('model1.joblib')
        result = model.predict_proba(self.scaled_features)
        return result

    def run(self):
        self.preprocess()
        self.scale_data()
        result = self.predict()
        return round(result[0][1]*100, 4)
