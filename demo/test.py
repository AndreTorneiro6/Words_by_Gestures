# import cv2
# import mediapipe as mp
#
# mp_holistic = mp.solutions.holistic
# mp_drawing = mp.solutions.drawing_utils  # Drawing helpers
#
# with mp_holistic.Holistic(min_tracking_confidence=0.6, min_detection_confidence=0.6,
#                                       static_image_mode=True) as allbody:
#
#     image = cv2.imread(r'C:\Users\andre\Desktop\IMG_20210826_095000.jpg')
#     # cv2.imshow('', image)
#     image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
#     # To improve performance, optionally mark the image as not writeable to
#     # pass by reference.
#     image.flags.writeable = False
#     result = allbody.process(image)
#
#     # Draw landmark annotation on the image.
#     image.flags.writeable = True
#     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#
#     print(result.face_landmarks)
#     # 1. Draw face landmarks
#     mp_drawing.draw_landmarks(image, result.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
#                               mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1,
#                                                      circle_radius=1),
#                               mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1,
#                                                      circle_radius=1)
#                               )
#
#     # 2. Right hand
#     mp_drawing.draw_landmarks(image, result.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
#                               mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2,
#                                                      circle_radius=2),
#                               mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2,
#                                                      circle_radius=2)
#                               )
#
#     # 3. Left Hand
#     mp_drawing.draw_landmarks(image, result.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
#                               mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2,
#                                                      circle_radius=2),
#                               mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2,
#                                                      circle_radius=2)
#                               )
#
#     # 4. Pose Detectionsq
#     mp_drawing.draw_landmarks(image, result.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
#                               mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2,
#                                                      circle_radius=3),
#                               mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2,
#                                                      circle_radius=2)
#                               )
#     cv2.imwrite('annotated_image'  '.png', image)
#     cv2.waitKey(0)
# cv2.destroyAllWindows()
# import gtts
# from playsound import playsound
# # tts = gtts.gTTS('Eu estou aqui', lang="pt", slow=False)
# # name = 'hola.mp3'
# # # save audio_CNN2D file
# # tts.save("hola.mp3")
# # playsound(r'C:\Users\andre\Desktop\WBG-words_by_gestures\hola.mp3')
#
# import pyttsx3
# engine = pyttsx3.init()
# engine.say("Eu estou aqui");
# voices = engine.getProperty('voices')
# for voice in voices:
#    print(engine.setProperty('voice', voice.id) )
# engine.runAndWait() ;
# print (engine.getProperty('voices'))
import cv2
import mediapipe

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh


drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
with mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    min_detection_confidence=0.5) as face_mesh:
    image = cv2.imread(file)
    # Convert the BGR image to RGB before processing.
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Print and draw face mesh landmarks on the image.
    if not results.multi_face_landmarks:
      continue
    annotated_image = image.copy()
    for face_landmarks in results.multi_face_landmarks:
      print('face_landmarks:', face_landmarks)
      mp_drawing.draw_landmarks(
          image=annotated_image,
          landmark_list=face_landmarks,
          connections=mp_face_mesh.FACEMESH_TESSELATION,
          landmark_drawing_spec=None,
          connection_drawing_spec=mp_drawing_styles
          .get_default_face_mesh_tesselation_style())
      mp_drawing.draw_landmarks(
          image=annotated_image,
          landmark_list=face_landmarks,
          connections=mp_face_mesh.FACEMESH_CONTOURS,
          landmark_drawing_spec=None,
          connection_drawing_spec=mp_drawing_styles
          .get_default_face_mesh_contours_style())
    cv2.imwrite('/tmp/annotated_image' + str(idx) + '.png', annotated_image)
