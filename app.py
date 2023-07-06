# Importation des modules nécessaires
from flask import Flask, render_template, Response # Importation de la classe Flask pour créer l'application Flask
import cv2 # Bibliothèque pour la vision par ordinateur (OpenCV)
import numpy as np # Bibliothèque pour le calcul scientifique
import dlib # Bibliothèque pour la reconnaissance faciale
from imutils import face_utils # Bibliothèque pour les opérations faciales
import time # Bibliothèque pour le temps
from pygame import mixer # Bibliothèque pour les sons

# Initialisation de l'application Flask
app = Flask(__name__)

# Initialisation du module de lecture audio (mixer) de Pygame
mixer.init()

# Chargement des fichiers audio
no_driver_sound = mixer.Sound('nodriver_audio.mp3')
sleep_sound = mixer.Sound('sleep_sound.mp3')
tired_sound = mixer.Sound('rest_audio.mp3')

# Initialisation du détecteur de visage de Dlib
detector = dlib.get_frontal_face_detector()

# Chargement du model prédicteur de points repères du visage de Dlib
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Fonction pour calculer la distance entre deux points
def compute(ptA, ptB):
    dist = np.linalg.norm(ptA - ptB)
    return dist

# Fonction pour détecter le clignement des yeux
def blinked(a, b, c, d, e, f):
    # Calcul de la distance verticale entre les points des paupières supérieure et inférieure
    up = compute(b, d) + compute(c, e)
    # Calcul de la distance horizontale entre les points des coins externes et internes de l'œil
    down = compute(a, f)
    # Calcul du ratio entre la distance verticale et horizontale
    ratio = up / (2.0 * down)
    # Verifier si les yeux sont fermés ou ouverts EAR
    if ratio > 0.22:
        return 'actif'
    else:
        return 'sommeil'


# Fonction pour calculer le ratio d'ouverture de la bouche
def mouth_aspect_ratio(mouth):
    # Calcul des distances entre les points caractéristiques de la bouche
    A = compute(mouth[2], mouth[10])  # 51, 59
    B = compute(mouth[4], mouth[8])  # 53, 57
    C = compute(mouth[0], mouth[6])  # 49, 55
    # Calcul du ratio d'ouverture de la bouche
    mar = (A + B) / (2.0 * C)
    return mar

# Définition des indices de début et de fin pour la bouche
(mStart, mEnd) = (49, 68)

# Fonction asynchrone pour la détection de la fatigue
async def tired():
    start = time.time()  # Récupère le temps de départ
    rest_time_start = start  # Temps de départ pour le temps de repos
    tired_sound.play()  # Joue le son de fatigue
    a = 0

    while time.time() - start < 9:  # Boucle tant que le temps écoulé est inférieur à 9 secondes
        if time.time() - rest_time_start > 3:  # Vérifie si 3 secondes se sont écoulées depuis le dernier temps de repos
            tired_sound.play()  # Joue le son de fatigue

    tired_sound.stop()  # Arrête le son de fatigue
    return


# Fonction principale pour la détection de la fatigue du conducteur
def detech():
    sleep_sound_flag = 0  # Indicateur pour le son de fatigue
    no_driver_sound_flag = 0  # Indicateur pour le son de conducteur absent
    yawning = 0  # Compteur de frame de  bâillements
    no_yawn = 0  # Compteur de bâillements
    sleep = 0  # Compteur de somnolence
    active = 0  # Compteur d'activité
    status = ""  # Statut actuel
    color = (0, 0, 0)  # Couleur pour l'affichage du statut
    no_driver = 0  # Compteur de conducteur absent
    frame_color = (0, 255, 0)  # Couleur du cadre de l'image
    url = "http://192.168.1.33:8080/video"  # URL du flux vidéo
    cap = cv2.VideoCapture(0)  # Initialisation de la capture vidéo

    time.sleep(1)  # Donner un peu de temps à la caméra pour s'initialiser (pas necessaire)
    start = time.time()  # Instant de départ
    no_driver_time = time.time()  # Instant pour le conducteur absent
    no_driver_sound_start = time.time()  # Instant pour le son de conducteur absent

    # Boucle principale pour la détection des visages , des bailliement  et des clignements d'yeux
    while True:
        _, frame = cap.read()  # Lecture d'une image depuis la capture vidéo
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)  # Conversion en espace de couleur YUV
        channels = cv2.split(frame)  # Séparation des canaux de couleur
        cv2.equalizeHist(channels[0], channels[0])  # Égalisation de l'histogramme du canal Y
        frame = cv2.merge(channels)  # Fusion des canaux de couleur
        frame = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR)  # Conversion en espace de couleur BGR
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Conversion en niveaux de gris
        face_frame = frame.copy()  # Copie de l'image du visage
        faces = detector(gray, 0)  # Détection des visages dans l'image

        if faces:
            no_driver_sound_flag = 0  # Réinitialisation de l'indicateur de son de conducteur absent
            no_driver_sound.stop()  # Arrêt du son de conducteur absent
            no_driver = 0  # Réinitialisation du compteur de conducteur absent
            no_driver_time = time.time()  # Mise à jour de l'instant pour le conducteur absent

            face = faces[0]  # Sélection du premier visage détecté
            x1 = face.left()  # Coordonnée x du coin supérieur gauche du visage
            y1 = face.top()  # Coordonnée y du coin supérieur gauche du visage
            x2 = face.right()  # Coordonnée x du coin inférieur droit du visage
            y2 = face.bottom()  # Coordonnée y du coin inférieur droit du visage

            cv2.rectangle(frame, (x1, y1), (x2, y2), frame_color, 2)  # Dessin du rectangle autour du visage

            landmarks = predictor(gray, face)  # Détection des repères faciaux
            landmarks = face_utils.shape_to_np(landmarks)  # Conversion en tableau NumPy

            left_blink = blinked(landmarks[36], landmarks[37], landmarks[38], landmarks[41], landmarks[40], landmarks[39])  # Détection du clignement de l'œil gauche
            right_blink = blinked(landmarks[42], landmarks[43], landmarks[44], landmarks[47], landmarks[46], landmarks[45])  # Détection du clignement de l'œil droit
            mouth = landmarks[mStart:mEnd]  # Région de la bouche
            mouthMAR = mouth_aspect_ratio(mouth)  # Calcul du rapport d'ouverture de la bouche
            mar = mouthMAR  # Alias pour le rapport d'ouverture de la bouche

            if mar > 0.80:
                sleep = 0  # Réinitialisation du compteur de somnolence
                active = 0  # Réinitialisation du compteur d'activité
                yawning += 1  # Incrément du compteur de bâillements
                status = "Fatigue"  # Statut : Fatigue
                color = (255, 0, 0)  # Couleur pour l'affichage du statut : blue
                frame_color = (255, 0, 0)  # Couleur du cadre de l'image : blue
                sleep_sound_flag = 0  # Réinitialisation de l'indicateur de son de fatigue
                sleep_sound.stop()  # Arrêt du son de fatigue
            elif left_blink == 'sommeil' or right_blink == 'sommeil':
                if yawning > 20:
                    no_yawn += 1  # Incrément du compteur de bâillements
                sleep += 1  # Incrément du compteur de somnolence
                yawning = 0  # Réinitialisation du compteur de bâillements
                active = 0  # Réinitialisation du compteur d'activité
                if sleep > 5:
                    status = "Endormi"  # Statut : Endormi
                    color = (0, 0, 255)  # Couleur pour l'affichage du statut : Rouge
                    frame_color = (0, 0, 255)  # Couleur du cadre de l'image : Rouge
                    if sleep_sound_flag == 0:
                        sleep_sound.play()  # Lecture du son de fatigue
                    sleep_sound_flag = 1  # Mise à jour de l'indicateur de son de fatigue
            else:
                if yawning > 20:
                    no_yawn += 1  # Incrément du compteur de bâillements
                yawning = 0  # Réinitialisation du compteur de bâillements
                sleep = 0  # Réinitialisation du compteur de somnolence
                active += 1  # Incrément du compteur d'activité
                status = "Actif"  # Statut : Actif
                color = (0, 255, 0)  # Couleur pour l'affichage du statut : Vert
                frame_color = (0, 255, 0)  # Couleur du cadre de l'image : Vert
                if active > 5:
                    sleep_sound_flag = 0  # Réinitialisation de l'indicateur de son de fatigue
                    sleep_sound.stop()  # Arrêt du son de fatigue

            cv2.putText(frame, status, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)  # Affichage du statut

            if time.time() - start < 60 and no_yawn >= 1:
                no_yawn = 0  # Réinitialisation du compteur de bâillements
                tired_sound.play()  # Lecture du son de fatigue
            elif time.time() - start > 60:
                start = time.time()  # Mise à jour de l'instant de départ

            #for n in range(0, 68):
            #   (x, y) = landmarks[n]
            #   cv2.circle(face_frame, (x, y), 1, (255, 255, 255), -1)  # Dessin des repères faciaux

        else:
            no_driver += 1  # Incrément du compteur de conducteur absent
            sleep_sound_flag = 0  # Réinitialisation de l'indicateur de son de fatigue
            sleep_sound.stop()  # Arrêt du son de fatigue
            if no_driver > 10:
                status = "Pas de conducteur"  # Statut : Pas de conducteur
                color = (0, 0, 0)  # Couleur pour l'affichage du statut : Noir
            if time.time() - no_driver_time > 5:
                if no_driver_sound_flag == 0:
                    no_driver_sound.play()  # Lecture du son de conducteur absent
                    no_driver_sound_start = time.time()  # Mise à jour de l'instant pour le son de conducteur absent
                else:
                    if time.time() - no_driver_sound_start > 3:
                        no_driver_sound.play()  # Lecture du son de conducteur absent
                        no_driver_sound_start = time.time()  # Mise à jour de l'instant pour le son de conducteur absent
                no_driver_sound_flag = 1  # Mise à jour de l'indicateur de son de conducteur absent

        cv2.putText(frame, status, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)  # Affichage du statut

        ret, buffer = cv2.imencode('.jpg', frame)  # Encodage de l'image au format JPEG
        frame = buffer.tobytes()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # Renvoi de l'image encodée





# Route pour le flux vidéo
@app.route("/video_feed")
def video_feed():
    print("Ouverture de la caméra")
    return Response(detech(), mimetype='multipart/x-mixed-replace;boundary=frame') # Renvoi de la réponse HTTP contenant le flux vidéo de détection

# Route pour la page d'accueil
@app.route("/")
def home():
    return render_template("index.html")

# Route pour la page de détection
@app.route("/detection")
def detection():
    return render_template("detection.html")

# Point d'entrée de l'application
if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0')
