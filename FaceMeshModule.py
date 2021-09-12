import cv2
import mediapipe as mp
import time
import mouse

class FaceMeshDetector():
    def __init__(self, staticMode=False, maxFaces=1, minDetectionCon=0.5, minTrackCon=0.5):

        self.staticMode = staticMode
        self.maxFaces = maxFaces
        self.minDetectionCon = minDetectionCon
        self.minTrackCon = minTrackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(self.staticMode, self.maxFaces, self.minDetectionCon, self.minTrackCon)
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=2)

    def findFaceMesh(self, img, draw=True):
        self.imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(self.imgRGB)

        faces = []
        if self.results.multi_face_landmarks:
            faces = []
            for faceLms in self.results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, faceLms, self.mpFaceMesh.FACE_CONNECTIONS,
                                               landmark_drawing_spec=self.drawSpec,
                                               connection_drawing_spec=self.drawSpec)
                    drawEyeRegions(img, faceLms.landmark)

                face = []
                for id, lm in enumerate(faceLms.landmark):
                    ih, iw, ic = img.shape
                    x, y = int(lm.x * iw), int(lm.y * ih)
                    # print(id,x,y)
                    # if id % 2 == 0:
                    #    cv2.cv2.putText(img, str(id), (x, y), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
                    # if id == 176:
                    #    cv2.cv2.putText(img, str(id), (x, y), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)

                    face.append([x, y])
                faces.append(face)
        return img, faces


# Crop the right eye region
def getRightEye(img, lm):
    eye_top = int(lm[263].y * img.shape[0])
    eye_left = int(lm[362].x * img.shape[1])
    eye_bottom = int(lm[374].y * img.shape[0])
    eye_right = int(lm[263].x * img.shape[1])
    right_eye = img[eye_top:eye_bottom, eye_left:eye_right]
    return right_eye


# Get the right eye coordinates on the actual -> to visualize the bbox
def getRightEyeRect(img, lm):
    eye_top = int(lm[257].y * img.shape[0])
    eye_left = int(lm[362].x * img.shape[1])
    eye_bottom = int(lm[374].y * img.shape[0])
    eye_right = int(lm[263].x * img.shape[1])

    cloned_img = img.copy()
    cropped_right_eye = cloned_img[eye_top:eye_bottom, eye_left:eye_right]
    h, w, _ = cropped_right_eye.shape
    x = eye_left
    y = eye_top
    return x, y, w, h


def getLeftEye(img, lm):
    eye_top = int(lm[159].y * img.shape[0])
    eye_left = int(lm[33].x * img.shape[1])
    eye_bottom = int(lm[145].y * img.shape[0])
    eye_right = int(lm[133].x * img.shape[1])
    left_eye = img[eye_top:eye_bottom, eye_left:eye_right]
    return left_eye


def getLeftEyeRect(img, lm):
    # eye_left lm (27, 23, 130, 133) ->? how to utilize z info
    eye_top = int(lm[159].y * img.shape[0])
    eye_left = int(lm[33].x * img.shape[1])
    eye_bottom = int(lm[145].y * img.shape[0])
    eye_right = int(lm[133].x * img.shape[1])

    cloned_img = img.copy()
    cropped_left_eye = cloned_img[eye_top:eye_bottom, eye_left:eye_right]
    h, w, _ = cropped_left_eye.shape

    x = eye_left
    y = eye_top
    return x, y, w, h


def drawEyeRegions(img, lm):
    # Visualize the Left and Region by drawing a rectangle on it on the actual image.
    # RIGH EYE
    # rightEyeImg = getRightEye(img, lm)
    # rightEyeHeight, rightEyeWidth, _ = rightEyeImg.shape

    xRightEye, yRightEye, rightEyeWidth, rightEyeHeight = getRightEyeRect(img, lm)
    cv2.rectangle(img, (xRightEye, yRightEye),
                  (xRightEye + rightEyeWidth, yRightEye + rightEyeHeight), (200, 21, 36), 2)

    # LEFT EYE
    # leftEyeImg = getLeftEye(img, lm)
    # leftEyeHeight, leftEyeWidth, _ = leftEyeImg.shape

    xLeftEye, yLeftEye, leftEyeWidth, leftEyeHeight = getLeftEyeRect(img, lm)
    cv2.rectangle(img, (xLeftEye, yLeftEye),
                  (xLeftEye + leftEyeWidth, yLeftEye + leftEyeHeight), (200, 21, 36), 2)


def main():
    cap = cv2.VideoCapture(2)
    # Set properties. Each returns === True on success (i.e. correct resolution)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    c_time = time.time()
    p_time = 0

    #p_w, p_h = mouse.size()
    #p_w = p_w / 2
    #p_h = p_h / 2
    detector = FaceMeshDetector()

    while True:
        success, img = cap.read()
        scale_percent = 220  # percent of original size
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)

        # resize image
        img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

        img, faces = detector.findFaceMesh(img)
        # print(img.shape)
        if len(faces) != 0:
            if c_time % 30:
                # print(faces[0][0]-p_coords)
                mouse.move(faces[0][0][0],faces[0][0][1],absolute=True,duration=0)

        c_time = time.time()
        fps = 1 / (c_time - p_time)

        cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)

        cv2.imshow("Image", img)
        p_time = c_time
        key = cv2.waitKey(1)
        if key == 27:
            break


if __name__ == "__main__":
    main()
