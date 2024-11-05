# imports
import tensorflow as tf
import numpy as np
import cv2
import mediapipe as mp
import os

mp_drawing = mp.solutions.drawing_utils  # Drawing helpers


def create_directories():
    path = '/home/andre/Desktop/WBG(linha_de_emergencia_cs)/Dataset/LGP2LP'
    new_folder = 'Keypoints'
    print(os.path.join(path, new_folder))
    principal_folder = os.path.join(path, new_folder)

    if not os.path.exists(principal_folder):
        os.mkdir(principal_folder)

    types = ['Dactilologia', 'Declarativo', 'Exclamação', 'Interrogação']
    folders = ['Hands', 'Face', 'Pose', 'Holistic', 'Movenet']
    for type in types:
        sub_folder1 = os.path.join(path, new_folder, type)
        if not os.path.exists(sub_folder1):
            os.mkdir(sub_folder1)
        for folder in folders:
            sub_folder2 = os.path.join(path, new_folder, type, folder)
            if not os.path.exists(sub_folder2):
                os.mkdir(sub_folder2)
            for root, dirs, filename in os.walk('/home/andre/Desktop/WBG(linha_de_emergencia_cs)/Dataset/LGP2LP/'
                                                'Vídeos/Dactilologia'):
                i = 0
                if len(root.split('/')) != 10:
                    pass
                else:
                    if root.split('/')[8] == type:
                        name = os.path.join(path, new_folder, type, folder, root.split('/')[9])
                        # print(name)
                        # print(os.path.isdir(name))

                        if os.path.isdir(name):
                            pass
                        else:
                            os.mkdir(name)
                        while i < len(filename):
                            number = os.path.join(path, new_folder, type, folder, root.split('/')[9], str(i))
                            print(number)
                            if os.path.isdir(number):
                                pass
                            else:
                                os.mkdir(os.path.join(path, new_folder, type, folder, root.split('/')[9], str(i)))
                                print(os.path.join(path, new_folder, type, folder, root.split('/')[9], str(i)))
                            i += 1


def hands_extractor():
    mp_hands = mp.solutions.hands

    cam = cv2.VideoCapture('/home/andre/Desktop/WBG(linha_de_emergencia_cs)/2.mp4')

    with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.4, static_image_mode=True) as hands:
        while cam.isOpened():
            success, image = cam.read()
            if not success:
                break
            else:
                # Flip the image horizontally for a later selfie-view display, and convert
                # the BGR image to RGB.
                image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
                # To improve performance, optionally mark the image as not writeable to
                # pass by reference.
                image.flags.writeable = False
                results = hands.process(image)

                # Draw the hand annotations on the image.
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                if results.multi_hand_landmarks:

                    for hand_landmarks in results.multi_hand_landmarks:
                        print(hand_landmarks)
                        mp_drawing.draw_landmarks(
                            image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                cv2.imshow('MediaPipe Hands', image)
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
        cam.release()


def face_extractor():
    mp_face_mesh = mp.solutions.face_mesh

    cam = cv2.VideoCapture(
        '/home/andre/Desktop/WBG(linha_de_emergencia)1/LGP_to_LP/1.png')
    with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
        while cam.isOpened():
            ret, frame = cam.read()
            if ret:

                # Detections
                result = face_mesh.process(frame)

                # draw points
                drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

                if result.multi_face_landmarks:
                    for face_landmarks in result.multi_face_landmarks:
                        mp_drawing.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACE_CONNECTIONS,
                                                  drawing_spec, drawing_spec)
                        print(face_landmarks)
                # show result
                cv2.imshow('Live Video', frame)
            else:
                break

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break


def pose_extractor():
    mp_pose = mp.solutions.pose

    cam = cv2.VideoCapture(r'C:\Users\andre\Desktop\WBG\LGP to LP\Dataset\Vídeos\Interrogação\Como te sentes\1.mp4')
    with mp_pose.Pose(min_tracking_confidence=0.6, min_detection_confidence=0.6) as pose:
        while cam.isOpened():
            ret, frame = cam.read()
            if ret:
                # extração da posição das articulações
                result = pose.process(frame)

                # call function to draw
                mp_pose = mp.solutions.pose

                mp_drawing.draw_landmarks(
                    frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                # show video
                cv2.imshow('Live Video', frame)
            else:
                break

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break


def holistic():
    mp_holistic = mp.solutions.holistic  # Mediapipe Solutions
    target = '/home/andre/Desktop/WBG(linha_de_emergencia_cs)/Dataset/LGP2LP/Vídeos/Dactilologia'
    aux = []
    for root, dirs, filenames in os.walk(target):
        if len(root.split('/')) > 9:
            if len(os.listdir(root)) > 1:
                for x in filenames:
                    if x.endswith('.mp4'):
                        # print(root)
                        aux.append(root + '/' + x)
    i = 0
    frame_number = 0
    while i < len(aux):
        print(('faltam: ' + str(len(aux) - i) + ' ' + 'videos'))
        cam = cv2.VideoCapture(aux[i])
        # print(aux[i].split('/'))
        dir = '/home/andre/Desktop/WBG(linha_de_emergencia_cs)/Dataset/LGP2LP/Keypoints'
        video_number = aux[i].split('/')[10].split('.')[0]
        print(video_number)
        new_folder = os.path.join(dir, aux[i].split('/')[8], 'Holistic', aux[i].split('/')[9], video_number)
        print(len(os.listdir(new_folder)))
        if (len(os.listdir(new_folder))) != 0:
            print('já está extraído')
            i += 1
        else:
            print('a extrair...')
            print(int(cam.get(cv2.CAP_PROP_FRAME_COUNT)))
            with mp_holistic.Holistic(min_tracking_confidence=0.6, min_detection_confidence=0.6,
                                      static_image_mode=True) as allbody:
                while cam.isOpened():
                    sucess, frame = cam.read()
                    if sucess:

                        # extração da posição das articulações
                        result = allbody.process(frame)

                        # # draw keypoints
                        # # 1. Draw face landmarks
                        mp_drawing.draw_landmarks(frame, result.face_landmarks, mp_holistic.FACE_CONNECTIONS,
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

                        # 4. Pose Detectionsq
                        mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                                  mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2,
                                                                         circle_radius=4),
                                                  mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2,
                                                                         circle_radius=2)
                                                  )

                        # show video
                        cv2.imshow('Live Video', frame)

                        # --------------------- save results ----------------------
                        # 1. Pose
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
                        # print(keypoints)
                        # print(len(pose)+len(right)+len(left) +len(face))
                        # guardar pontos extraidos
                        new_path = os.path.join(new_folder, str(frame_number))
                        # print()
                        np.save(new_path, keypoints)

                        frame_number += 1
                    else:
                        frame_number = 0
                        i += 1
                        break

                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        break
                cam.release()
        if i == len(aux):
            print('concluido')

        cv2.destroyAllWindows()


def pose_movenet():
    from LGP_to_LP.Keypoints_extractors.Draw_points import draw_keypoints
    from LGP_to_LP.Keypoints_extractors.Draw_points import draw_connections

    # 0 . Load model
    interpreter = tf.lite.Interpreter(
        model_path='../../all_models/movenet/lite-model_movenet_singlepose_lightning_3.tflite')
    interpreter.allocate_tensors()

    # 1. Desenhar contornos
    EDGES = {(0, 1): 'm', (0, 2): 'c', (1, 3): 'm', (2, 4): 'c', (0, 5): 'm', (0, 6): 'c',
             (5, 7): 'm', (7, 9): 'm', (6, 8): 'c', (8, 10): 'c', (5, 6): 'y', (5, 11): 'm',
             (6, 12): 'c', (11, 12): 'y', (11, 13): 'm', (13, 15): 'm', (12, 14): 'c',
             (14, 16): 'c'
             }

    cam = cv2.VideoCapture(
        '/home/andre/Desktop/WBG(linha_de_emergencia)1/Dataset/LGP2LP/Vídeos/Declarativo/A minha esposa está no hospital/1.mp4')
    # cap = cv2.VideoCapture(0)
    while cam.isOpened():
        ret, frame = cam.read()
        if ret:
            # Reshape image
            img = frame.copy()
            img = tf.image.resize_with_pad(np.expand_dims(img, axis=0), 192, 192)
            input_image = tf.cast(img, dtype=tf.float32)

            # Setup input and output
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()

            # Make predictions
            interpreter.set_tensor(input_details[0]['index'], np.array(input_image))
            interpreter.invoke()
            keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])
            # print(keypoints_with_scores)

            # Rendering
            draw_connections(frame, keypoints_with_scores, EDGES, 0.4)
            draw_keypoints(frame, keypoints_with_scores, 0.4)

            cv2.imshow('MoveNet Lightning', frame)
        else:
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # create_directories()
    # hands_extractor()
    # face_extractor()
    # pose_extractor()
    holistic()
    # pose_movenet()
