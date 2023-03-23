import cv2
import os
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

def procesar(imagen): #funcion para procesar la imagen (con filtros)
    imagen = cv2.resize(imagen, None, fx=0.15, fy=0.15)
    imagen = cv2.GaussianBlur(imagen, (7,7), 0)
    imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    imagen = cv2.Canny(imagen,100, 200)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    imagen = cv2.dilate(imagen, kernel, iterations=1)
    contornos, jerarquia = cv2.findContours(imagen, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(imagen, contornos, -1, (255,255,255), -1)
    imagen = cv2.erode(imagen, kernel, iterations=3)
    return imagen

def visualizar(imagen,imagen_proc,guardar,clase): #funcion para visualizar la imagen original y la imagen procesada
    imagen2 = imagen.copy()
    imagen2 = cv2.resize(imagen, None, fx=0.15, fy=0.15)
    imagen2 = cv2.cvtColor(imagen2, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Imagen original /// Imagen procesada', cv2.hconcat([imagen2, imagen_proc]))
    cv2.waitKey(0)
    if guardar==True:
        cv2.imwrite('IMAGENES_PROC/Imagen_procesada_'+clase+'.jpg', cv2.hconcat([imagen2, imagen_proc]))

def visualizar_full(imagen,prediccion): #funcion para visualizar la imagen con su prediccion y medidas
    imagen = cv2.resize(imagen, None, fx=0.15, fy=0.15)
    imagen_org=imagen.copy()
    imagen = cv2.GaussianBlur(imagen, (7,7), 0)
    imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    imagen = cv2.Canny(imagen,100, 200)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    imagen = cv2.dilate(imagen, kernel, iterations=1)
    contornos, jerarquia = cv2.findContours(imagen, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(imagen, contornos, -1, (255,255,255), -1)
    imagen = cv2.erode(imagen, kernel, iterations=3)
    contornos = sorted(contornos, key=cv2.contourArea, reverse=True)
    contorno = contornos[0]
    cv2.putText(imagen_org, prediccion, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    if prediccion=='CLAVO':
        rect = cv2.minAreaRect(contorno)
        largo = rect[1][0]
        ancho = rect[1][1]
        if largo < ancho:
            largo, ancho = ancho, largo
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        imagen = cv2.drawContours(imagen_org, [box], 0, (0, 255, 0), 2)
        const_clavo=8.659
        cv2.putText(imagen_org, "Largo: " + str(round(largo/const_clavo,2)), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(imagen_org, "Ancho: " + str(round(ancho/const_clavo,2)), (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    elif prediccion=='TORNILLO':
        rect = cv2.minAreaRect(contorno)
        largo = rect[1][0]
        ancho = rect[1][1]
        if largo < ancho:
            largo, ancho = ancho, largo
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        imagen = cv2.drawContours(imagen_org, [box], 0, (0, 255, 0), 2)
        const_torn=9.265
        cv2.putText(imagen_org, "Largo: " + str(round(largo/const_torn,2)), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(imagen_org, "Ancho: " + str(round(ancho/const_torn,2)), (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    elif prediccion=='TUERCA':
        circ = cv2.minEnclosingCircle(contorno)
        diametro = circ[1]
        cir= cv2.circle(imagen_org, (int(circ[0][0]), int(circ[0][1])), int(diametro), (0, 255, 0), 2)
        const_tuer=4.763
        cv2.putText(imagen_org, "Diametro: " + str(round(diametro/const_tuer,2)), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    elif prediccion=='ARANDELA':
        circ = cv2.minEnclosingCircle(contorno)
        diametro = circ[1]
        cir= cv2.circle(imagen_org, (int(circ[0][0]), int(circ[0][1])), int(diametro), (0, 255, 0), 2)
        const_aran=4.34
        cv2.putText(imagen_org, "Diametro: " + str(round(diametro/const_aran,2)), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imshow("Imagen", imagen_org)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def obtener_momentos():
    momentos_hu = []
    clases = []
    for carpeta in os.listdir('FOTOS'):
        for imagen in os.listdir('FOTOS/'+carpeta):
            img = cv2.imread('FOTOS/'+carpeta+'/'+imagen)
            img_proc = procesar(img)
            momentos = cv2.moments(img_proc)
            hu_moments = cv2.HuMoments(momentos)
            momentos_hu.append(hu_moments)
            clases.append(carpeta)
    return momentos_hu, clases

def guardar_datos():
    momentos_hu,clases=obtener_momentos()
    np.savez('datos.npz', momentos_hu=momentos_hu, clases=clases)

def clasificar_kmeans(imagen,graficas):
    datos = np.load('datos.npz')
    datos_d = []
    for i in range(0, len(datos['momentos_hu'])):
        media = np.mean(datos['momentos_hu'][i])
        varianza = np.var(datos['momentos_hu'][i])
        datos_d.append([media, varianza])
        array_datos = np.array(datos_d)

    #SEPARO LOS DATOS SEGUN LA CLASE
    array_aran = []
    array_clav = []
    array_torn = []
    array_tuer = []
    for i in range(0, len(datos['clases'])):
        if datos['clases'][i] == 'ARANDELAS':
            array_aran.append(array_datos[i])
        elif datos['clases'][i] == 'CLAVOS':
            array_clav.append(array_datos[i])
        elif datos['clases'][i] == 'TORNILLOS':
            array_torn.append(array_datos[i])
        elif datos['clases'][i] == 'TUERCAS':
            array_tuer.append(array_datos[i])
    
    #GRAFICO
    if graficas==True:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.scatter(np.array(array_aran)[:,0], np.array(array_aran)[:,1], c="b")
        plt.scatter(np.array(array_clav)[:,0], np.array(array_clav)[:,1], c="g")
        plt.scatter(np.array(array_torn)[:,0], np.array(array_torn)[:,1], c="r")
        plt.scatter(np.array(array_tuer)[:,0], np.array(array_tuer)[:,1], c="y")
        plt.legend(['ARANDELAS', 'CLAVOS', 'TORNILLOS', 'TUERCAS'])
        ax.set_xlabel('Media')
        ax.set_ylabel('Varianza')
        ax.set_title('MEDIA - VARIANZA')
        plt.show()

    #CENTROIDES
    #inicializo los centroides de forma aleatoria con valores entre el minimo y el maximo de cada clase
    centroide_aran = [np.random.uniform(np.min(np.array(array_aran)[:,0]), np.max(np.array(array_aran)[:,0])), np.random.uniform(np.min(np.array(array_aran)[:,1]), np.max(np.array(array_aran)[:,1]))]
    centroide_clav = [np.random.uniform(np.min(np.array(array_clav)[:,0]), np.max(np.array(array_clav)[:,0])), np.random.uniform(np.min(np.array(array_clav)[:,1]), np.max(np.array(array_clav)[:,1]))]
    centroide_torn = [np.random.uniform(np.min(np.array(array_torn)[:,0]), np.max(np.array(array_torn)[:,0])), np.random.uniform(np.min(np.array(array_torn)[:,1]), np.max(np.array(array_torn)[:,1]))]
    centroide_tuer = [np.random.uniform(np.min(np.array(array_tuer)[:,0]), np.max(np.array(array_tuer)[:,0])), np.random.uniform(np.min(np.array(array_tuer)[:,1]), np.max(np.array(array_tuer)[:,1]))]
    
    #GRAFICO
    if graficas==True:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.scatter(centroide_aran[0], centroide_aran[1], c="b", marker="x")
        plt.scatter(centroide_clav[0], centroide_clav[1], c="g", marker="x")
        plt.scatter(centroide_torn[0], centroide_torn[1], c="r", marker="x")
        plt.scatter(centroide_tuer[0], centroide_tuer[1], c="y", marker="x")
        plt.legend(['ARANDELAS', 'CLAVOS', 'TORNILLOS', 'TUERCAS'])
        ax.set_title('CENTROIDES INICIALES')
        plt.show()

    #CLASIFICACION
    dist_aran = []
    dist_clav = []
    dist_torn = []
    dist_tuer = []

    clase_aran = []
    clase_clav = []
    clase_torn = []
    clase_tuer = []
    clusters=[]
    #calculo las distancias de cada elemento a cada centroide (distancia euclideana)
    for i in range(0, len(array_datos)):
        dist_aran.append(np.linalg.norm(array_datos[i] - centroide_aran))
        dist_clav.append(np.linalg.norm(array_datos[i] - centroide_clav))
        dist_torn.append(np.linalg.norm(array_datos[i] - centroide_torn))
        dist_tuer.append(np.linalg.norm(array_datos[i] - centroide_tuer))

    #en base a la distancia calculada asigno cada elemento a su clase (formo los clusters)
    for i in range(0, len(array_datos)):
        if dist_aran[i] < dist_clav[i] and dist_aran[i] < dist_torn[i] and dist_aran[i] < dist_tuer[i]:
            clase_aran.append(array_datos[i])
        elif dist_clav[i] < dist_aran[i] and dist_clav[i] < dist_torn[i] and dist_clav[i] < dist_tuer[i]:
            clase_clav.append(array_datos[i])
        elif dist_torn[i] < dist_aran[i] and dist_torn[i] < dist_clav[i] and dist_torn[i] < dist_tuer[i]:
            clase_torn.append(array_datos[i])
        elif dist_tuer[i] < dist_aran[i] and dist_tuer[i] < dist_clav[i] and dist_tuer[i] < dist_torn[i]:
            clase_tuer.append(array_datos[i])
    #agrego los clusters a una lista
    clusters.append([clase_aran, clase_clav, clase_torn, clase_tuer])
    iter=0
    while iter<15: #itero 15 veces o hasta que los centroides no cambien
        centroide_aran_anterior = centroide_aran
        centroide_clav_anterior = centroide_clav
        centroide_torn_anterior = centroide_torn
        centroide_tuer_anterior = centroide_tuer
        centroide_aran = np.mean(clusters[0][0], axis=0)
        centroide_clav = np.mean(clusters[0][1], axis=0)
        centroide_torn = np.mean(clusters[0][2], axis=0)
        centroide_tuer = np.mean(clusters[0][3], axis=0)
        #si los centroides cambian menos de 0.00001, termino el algoritmo
        if np.linalg.norm(centroide_aran - centroide_aran_anterior) < 0.00001 and np.linalg.norm(centroide_clav - centroide_clav_anterior) < 0.00001 and np.linalg.norm(centroide_torn - centroide_torn_anterior) < 0.00001 and np.linalg.norm(centroide_tuer - centroide_tuer_anterior) < 0.00001:
            break
        for i in range(0, len(array_datos)):
            dist_aran.append(np.linalg.norm(array_datos[i] - centroide_aran))
            dist_clav.append(np.linalg.norm(array_datos[i] - centroide_clav))
            dist_torn.append(np.linalg.norm(array_datos[i] - centroide_torn))
            dist_tuer.append(np.linalg.norm(array_datos[i] - centroide_tuer))
        #limpio los clusteres
        clase_aran = []
        clase_clav = []
        clase_torn = []
        clase_tuer = []
        for i in range(0, len(array_datos)):
            if dist_aran[i] < dist_clav[i] and dist_aran[i] < dist_torn[i] and dist_aran[i] < dist_tuer[i]:
                clase_aran.append(array_datos[i])
            elif dist_clav[i] < dist_aran[i] and dist_clav[i] < dist_torn[i] and dist_clav[i] < dist_tuer[i]:
                clase_clav.append(array_datos[i])
            elif dist_torn[i] < dist_aran[i] and dist_torn[i] < dist_clav[i] and dist_torn[i] < dist_tuer[i]:
                clase_torn.append(array_datos[i])
            elif dist_tuer[i] < dist_aran[i] and dist_tuer[i] < dist_clav[i] and dist_tuer[i] < dist_torn[i]:
                clase_tuer.append(array_datos[i])
        clusters.append([clase_aran, clase_clav, clase_torn, clase_tuer])
        print("iteracion:",iter+1)
        iter=iter+1
    print("----------------------------------------------------------------")
    clase_aran=clusters[iter][0]
    clase_clav=clusters[iter][1]
    clase_torn=clusters[iter][2]
    clase_tuer=clusters[iter][3]
    if graficas==True:
        fig=plt.figure()
        ax=fig.add_subplot(111)
        plt.scatter(np.array(clase_aran)[:,0], np.array(clase_aran)[:,1], c="b")
        plt.scatter(np.array(clase_clav)[:,0], np.array(clase_clav)[:,1], c="g")
        plt.scatter(np.array(clase_torn)[:,0], np.array(clase_torn)[:,1], c="r")
        plt.scatter(np.array(clase_tuer)[:,0], np.array(clase_tuer)[:,1], c="y")
        plt.scatter(centroide_aran[0], centroide_aran[1], c="black", marker="x")
        plt.scatter(centroide_clav[0], centroide_clav[1], c="black", marker="x")
        plt.scatter(centroide_torn[0], centroide_torn[1], c="black", marker="x")
        plt.scatter(centroide_tuer[0], centroide_tuer[1], c="black", marker="x")
        plt.legend(['ARANDELAS', 'CLAVOS', 'TORNILLOS', 'TUERCAS'])
        plt.title('K-MEANS')
        ax.set_title('CENTROIDES Y CLUSTERS')
        plt.show()
    #-------------------------------------------#
    #Se calculan los momentos de Hu de la imagen a predecir
    imagen_procesada = procesar(imagen)
    momentos = cv2.moments(imagen_procesada)
    hu_moments = cv2.HuMoments(momentos)
    media = np.mean(hu_moments)
    varianza = np.var(hu_moments)
    elemento = np.array([media, varianza])

    #Se calcula la distancia euclideana de la imagen a predecir con cada centroide
    dist_aran = np.linalg.norm(elemento - centroide_aran)
    dist_clav = np.linalg.norm(elemento - centroide_clav)
    dist_torn = np.linalg.norm(elemento - centroide_torn)
    dist_tuer = np.linalg.norm(elemento - centroide_tuer)

    #Se determina la clase en base a las distancias calculadas
    if dist_aran < dist_clav and dist_aran < dist_torn and dist_aran < dist_tuer:
        prediccion='ARANDELA'
    elif dist_clav < dist_aran and dist_clav < dist_torn and dist_clav < dist_tuer:
        prediccion='CLAVO'
    elif dist_torn < dist_aran and dist_torn < dist_clav and dist_torn < dist_tuer:
        prediccion='TORNILLO'
    elif dist_tuer < dist_aran and dist_tuer < dist_clav and dist_tuer < dist_torn:
        prediccion='TUERCA'
    print("PREDICCION:",prediccion)
    visualizar_full(imagen,prediccion)

def clasificar_knn(imagen,vecinos,graficas):
    datos = np.load('datos.npz')
    datos_d = []
    for i in range(0, len(datos['momentos_hu'])):
        media = np.mean(datos['momentos_hu'][i])
        varianza = np.var(datos['momentos_hu'][i])
        datos_d.append([media, varianza])
        array_datos = np.array(datos_d)
    imagen_procesada = procesar(imagen)
    momentos = cv2.moments(imagen_procesada)
    hu_moments = cv2.HuMoments(momentos)
    media = np.mean(hu_moments)
    varianza = np.var(hu_moments)
    elemento = np.array([media, varianza])
    distancias = [] 
    for i in range(0,len(array_datos)):
        distancia_eu = np.linalg.norm(elemento - array_datos[i])
        distancias.append([distancia_eu,datos['clases'][i]])
    distancias.sort()
    vecinos_cercanos = distancias[:vecinos]
    contador = Counter([i[1] for i in vecinos_cercanos])
    prediccion = contador.most_common(1)[0][0]
    #mostrar la clase de los vecinos mas cercanos
    print("CLASES VECINOS MAS CERCANOS:",[i[1] for i in vecinos_cercanos])
    if prediccion == 'ARANDELAS':
        prediccion = 'ARANDELA'
    elif prediccion == 'CLAVOS':
        prediccion = 'CLAVO'
    elif prediccion == 'TORNILLOS':
        prediccion = 'TORNILLO'
    elif prediccion == 'TUERCAS':
        prediccion = 'TUERCA'
    print("PREDICCION:",prediccion)
    array_aran = []
    array_clav = []
    array_torn = []
    array_tuer = []
    for i in range(0, len(datos['clases'])):
        if datos['clases'][i] == 'ARANDELAS':
            array_aran.append(array_datos[i])
        elif datos['clases'][i] == 'CLAVOS':
            array_clav.append(array_datos[i])
        elif datos['clases'][i] == 'TORNILLOS':
            array_torn.append(array_datos[i])
        elif datos['clases'][i] == 'TUERCAS':
            array_tuer.append(array_datos[i])
    if graficas == True:
        plt.scatter([i[0] for i in array_aran], [i[1] for i in array_aran], c="violet", marker="o")
        plt.scatter([i[0] for i in array_clav], [i[1] for i in array_clav], c="blue", marker="o")
        plt.scatter([i[0] for i in array_torn], [i[1] for i in array_torn], c="green", marker="o")
        plt.scatter([i[0] for i in array_tuer], [i[1] for i in array_tuer], c="yellow", marker="o")
        plt.scatter(media, varianza, c="red", marker="x")
        plt.legend(['ARANDELAS', 'CLAVOS', 'TORNILLOS', 'TUERCAS', 'ELEMENTO A CLASIFICAR'])
        plt.show()
    visualizar_full(imagen,prediccion)

###########################################################################

#ENTRENAMIENTO
#guardar_datos() #descomentar para crear nueva base de datos de entrenamiento

#CLASIFICACION
imagen = cv2.imread("FINAL/IMG_20221214_083310.jpg") #imagen a clasificar
imagen_procesada = procesar(imagen)
visualizar(imagen,imagen_procesada,guardar=False,clase='CLAVO')
#clasificar_kmeans(imagen,graficas=True)
clasificar_knn(imagen,3,graficas=True)

#Directorio - en caso de querer clasificar un directorio de imagenes
'''
for imagen in os.listdir('FINAL/'):
    imagen = cv2.imread('FINAL/'+'/'+imagen)
    clasificar_knn(imagen,3,graficas=True)
    clasificar_kmeans(imagen,graficas=True)
'''