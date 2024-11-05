import tensorflow as tf
import numpy as np
import cv2
import mediapipe as mp
from tensorflow import keras
import gtts
from playsound import playsound
from scipy import stats


mp_holistic = mp.solutions.holistic  # Holistic model
mp_drawing = mp.solutions.drawing_utils  # Drawing utilities

gpus = tf.config.experimental.list_physical_devices("GPU")
# print(gpus)
tf.config.experimental.set_memory_growth(gpus[0], True)

# load do modelo
model = keras.models.load_model(r'C:\Users\andre\Desktop\WBG-words_by_gestures\all_models\lingua_gestual(BI)')
model.summary()

colors = [(245, 117, 16), (117, 245, 16), (16, 117, 245)]


def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0, 60 + num * 40), (int(prob * 100), 90 + num * 40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85 + num * 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                    cv2.LINE_AA)

    return output_frame


def play(name):
    playsound(name)


sequence = []
predictions = []
sentence = []
threshold = 0.5

cam = cv2.VideoCapture(0)

actions = ['Ele está inconsciente', 'O meu pneu explodiu', 'O elevador está preso', 'Houve um acidente',
           'Gestua devagar, por favor', 'Tenho dificuldades a respirar', 'Ele está gravemente ferido',
           'Estou com dores no estômago', 'Tenho uma erupção cutânea']

action = np.array(actions)

with mp_holistic.Holistic(min_detection_confidence=0.6, min_tracking_confidence=0.6,
                          static_image_mode=True) as holistic:
    while cam.isOpened:
        ret, frame = cam.read()

        result = holistic.process(frame)

        # draw keypoints
        # 1. Draw face landmarks
        mp_drawing.draw_landmarks(frame, result.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
                                  mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1,
                                                         circle_radius=1),
                                  mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1,
                                                         circle_radius=1)
                                  )

        # 2. Right hand
        mp_drawing.draw_landmarks(frame, result.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2,
                                                         circle_radius=4),
                                  mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2,
                                                         circle_radius=2)
                                  )

        # 3. Left Hand
        mp_drawing.draw_landmarks(frame, result.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2,
                                                         circle_radius=4),
                                  mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2,
                                                         circle_radius=2)
                                  )

        # 4. Pose Detections
        mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2,
                                                         circle_radius=4),
                                  mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2,
                                                         circle_radius=2)
                                  )

        if result.pose_landmarks:
            pose = np.array([[res.x, res.y, res.z, res.visibility] for res in
                             result.pose_landmarks.landmark]).flatten()
        else:
            pose = np.zeros(33 * 4)
            # print('pose-', pose)

            # 2. Left hand
        if result.left_hand_landmarks:
            left = np.array([[res.x, res.y, res.z] for res in
                             result.left_hand_landmarks.landmark]).flatten()
        else:
            left = np.zeros(21 * 3)
            # print('esquerda-',left)

            # 3. Right hand
        if result.right_hand_landmarks:
            right = np.array([[res.x, res.y, res.z] for res in
                              result.right_hand_landmarks.landmark]).flatten()
        else:
            right = np.zeros(21 * 3)
            # print('direita-' + str(len(right)))

            # 4. Face
        if result.face_landmarks:
            face = np.array([[res.x, res.y, res.z] for res in
                             result.face_landmarks.landmark]).flatten()
        else:
            face = np.zeros(468 * 3)
            # print('cara-'+str(len(face)))

        # concatenar todos os pontos (frame a frame)
        keypoints = np.concatenate([face, pose, right, left])

        # cv2.imshow('Live test', frame)
        sequence.append(keypoints)
        sequence = sequence[-90:]
        x = np.array(sequence, dtype=np.float16)

        print(x.shape)

        if len(sequence) == 90:
            res = model.predict(np.expand_dims(x, axis=0))[0]
            print(action[np.argmax(res)])

            # make request to google to get synthesis
            tts = gtts.gTTS(action[np.argmax(res)], lang="pt", slow=False)

            # save audio_CNN2D file
            tts.save("hola.mp3")
            break

        cv2.imshow('OpenCV Feed', frame)

        # Break gracefully
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    play("hola.mp3")

    cam.release()
    cv2.destroyAllWindows()

    # predictions.append(np.argmax(res))
    #
    # if np.unique(predictions[-10:])[0] == np.argmax(res):
    #     if res[np.argmax(res)] > threshold:
    #
    #         if len(sentence) > 0:
    #             if action[np.argmax(res)] != sentence[-1]:
    #                 sentence.append(action[np.argmax(res)])
    #         else:
    #             sentence.append(action[np.argmax(res)])
    #
    # if len(sentence) > 5:
    #     sentence = sentence[-5:]
    #
    #     # Viz probabilities
    # image = prob_viz(res, actions, frame, colors)
    #
    # cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
    # cv2.putText(image, ' '.join(sentence), (3, 30),
    #             cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    # # break

