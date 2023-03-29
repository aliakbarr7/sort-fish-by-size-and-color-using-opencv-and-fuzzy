#Install scipy di CMD
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours

#Import Numpy
#Install Numpy di CMD
import numpy as np
#Install Imutills di CMD
import imutils
#Import OpenCV
import cv2
#Import Time
import time


#Inisialisasi variabel midpoint
#Menentukan titik tengah dari objek yang akan diukur
def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)


def nothing(x):
    pass
cv2.namedWindow("Pengaturan Threshold")
cv2.createTrackbar("Iteration", "Pengaturan Threshold", 0, 16, nothing)


#Menginisialisasikan batas bawah dan batas atas warna 1
lower_range_1=np.array([60, 92, 0])
upper_range_1=np.array([132, 255, 158])
#Menginisialisasikan batas bawah dan batas atas warna 2
lower_range_2=np.array([0, 135, 83])
upper_range_2=np.array([79, 255, 192])


#Mengaktifkan Kamera untuk Menampilkan Video Secara Realtime
cap = cv2.VideoCapture(1)


#Membuat Kondisi
#Apabila Kamera Aktif dan Video telah dimulai, Maka Jalankan Program di Bawah Ini
while (cap.read()):
#Pengukuran Objekkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkk        
        #Mengisi frame sebagai variabel untuk menampikan Video
        ref,frame = cap.read()
        #Mengatur Resolusi frame
        resolusi = cv2.resize(frame,(540,480))
        #Mengatur Ukuran dari frame Video untuk input Pengukuran Objek
        frame1 = cv2.resize(frame, None, fx=1, fy=1, interpolation=cv2.INTER_AREA)
        #Mengisi orig sebagai variabel frame sebagai input Video Pengukuran Objek
        #dan Mengatur Resolusi Video
        orig = frame1[:1080,0:1920]

        

        #Grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (15, 15), 0)
        thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
        kernel = np.ones((3,3),np.uint8)
        closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE,kernel,iterations=7)
        result_img = closing.copy()
        contours,hierachy = cv2.findContours(result_img, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        #Mengkonversi Nilai Pembacaan Pixel ke Dalam Satuan CM
        pixelsPerMetric = None
        
                 
#Identifikasi Warnaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
        #Mengisi frame2 sebagai input Video Pengenalan Warna
        frame2 = resolusi.copy()
        #Mengisi variabel hsv dengan fungsi COLOR_BGR2HSV
        hsv = cv2.cvtColor(frame2, cv2.COLOR_BGR2HSV)

        #Mengisi batas bawah dan batas atas warna dengan nilai dari Trackbars
        mask_1=cv2.inRange(hsv,lower_range_1,upper_range_1)
        mask_2=cv2.inRange(hsv,lower_range_2,upper_range_2)
        _,mask_1=cv2.threshold(mask_1, 254,255,cv2.THRESH_BINARY)
        _,mask_2=cv2.threshold(mask_2, 254,255,cv2.THRESH_BINARY)
        cnts_1,_=cv2.findContours(mask_1,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        cnts_2,_=cv2.findContours(mask_2,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
       
#Menjalankan Fungsi Warnaaaaaaaaaa       
        for c in cnts_1:
            x=50

            if cv2.contourArea(c)>x:
                x,y,w,h=cv2.boundingRect(c)
                cv2.rectangle(frame2,(x,y),(x+w,y+h),(0,255,0),2)
                cv2.putText(frame2, ("Biru"), (5,30), cv2.FONT_HERSHEY_SIMPLEX,1, (0,0,255), 2)
                cv2.putText(frame,"Biru",(160,30),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),2)
                warna = 'Biru'
                

        for d in cnts_2:
            x=50
            
            if cv2.contourArea(d)>x:
                x,y,w,h=cv2.boundingRect(d)
                cv2.rectangle(frame2,(x,y),(x+w,y+h),(0,255,0),2)
                cv2.putText(frame2, ("Merah"), (5,30), cv2.FONT_HERSHEY_SIMPLEX,1, (0,0,255), 2)
                cv2.putText(frame,"Merah",(160,30),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),2)
                warna = 'Merah'

#Menjalakankan Fungsi Pengukran Objekkkkkkkkkkkk        
        rows, cols, _ = orig.shape
        x_medium = int(rows / 2)
        x_maksimum = int(rows / 2)
        minimal = 10
        maksimal = 469
        garis_min = cv2.line(orig, (0, minimal), (720, minimal), (0, 255, 255), 2)
        garis_max = cv2.line(orig, (0, maksimal), (720, maksimal), (0, 255, 255), 2)

        for cnt in contours:
            #Pembacaan Area Objek yang di Ukur
            area = cv2.contourArea(cnt)
            #Jika Area Kurang dari 1000 dan Lebih dari 12000  Pixel
            #Maka Lakukan Pengukuran
            if area < 5000 or area > 120000:
                continue
            #...
            x,y,w,h=cv2.boundingRect(cnt)
            #Menghitung kotak pembatas dari contours Objek

            xtinggi = h / 2
            x_medium = int(((y + y + h) / 2) + xtinggi)
            x_maksimum = int(((y + y + h) / 2) - xtinggi)

            if ((x_maksimum > (minimal+30) and x_maksimum < (maksimal-30)) and (x_medium > (minimal+30) and x_medium < (maksimal-30))):
              
              orig = frame.copy()   
              box = cv2.minAreaRect(cnt)
              box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
              box = np.array(box, dtype="int")
              box = perspective.order_points(box)
              cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 64), 2)

              for (x, y) in box:
                  cv2.circle(orig, (int(x), int(y)), 5, (0, 255, 64), -1)
              
              (tl, tr, br, bl) = box
              (tltrX, tltrY) = midpoint(tl, tr)
              (blbrX, blbrY) = midpoint(bl, br)
              (tlblX, tlblY) = midpoint(tl, bl)
              (trbrX, trbrY) = midpoint(tr, br)

              #Menggambar titik tengah pada objek
              cv2.circle(orig, (int(tltrX), int(tltrY)), 0, (0, 255, 64), 5)
              cv2.circle(orig, (int(blbrX), int(blbrY)), 0, (0, 255, 64), 5)
              cv2.circle(orig, (int(tlblX), int(tlblY)), 0, (0, 255, 64), 5)
              cv2.circle(orig, (int(trbrX), int(trbrY)), 0, (0, 255, 64), 5)

              #Menggambar garis pada titik tengah
              cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
                            (255, 0, 255), 2)
              cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
                            (255, 0, 255), 2)

              #Menghitung jarak Euclidean antara titik tengah
              lebar_pixel = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
              panjang_pixel = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

              #Jika piksel pixelsPerMetric belum diinisialisasi, maka
              #Hitung sebagai rasio piksel terhadap metrik yang disediakan
              #Dalam hal ini CM
              if pixelsPerMetric is None:
                 pixelsPerMetric = lebar_pixel
                 pixelsPerMetric = panjang_pixel
              lebar = float((lebar_pixel / 25.5))
              ukuran = float((panjang_pixel / 25.5))

              #Menggambarkan ukuran benda pada gambar
              #cv2.putText(orig, "L: {:.1f} CM".format(lebar_pixel/25.5),(int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0,0,255), 2)
              cv2.putText(orig, "Lebar", (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0,0,255), 2)
              cv2.putText(orig, "L: {:.1f} CM".format(lebar), (int(5), int(120)), cv2.FONT_HERSHEY_SIMPLEX,0.7, (0,0,255), 2)
              cv2.putText(orig, "Panjang", (int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0,0,255), 2)
              cv2.putText(orig, "P: {:.1f} CM".format(ukuran),(int(5), int(90)), cv2.FONT_HERSHEY_SIMPLEX,0.7, (0,0,255), 2)
              #cv2.putText(orig,str(area),(int(x),int(y)),cv2.FONT_HERSHEY_SIMPLEX, 0.6,(0,0,0),2)
              #hitung_objek+=1
              
              #Proses Fuzzifikasi
              #Fuzzifikasi Ukuran
              if ukuran <= 8 :
                 value_pendek = 1
                 value_sedang = 0
                 value_panjang = 0
              if ukuran > 8 and ukuran <= 13 :
                 value_pendek = (13-ukuran)/(13-8)
                 value_sedang = (ukuran-8)/(13-8)
                 value_panjang = 0
              if ukuran > 13 and ukuran <= 17 :
                 value_pendek = 0
                 value_sedang = 1
                 value_panjang = 0
              if ukuran > 17 and ukuran <= 22 :
                 value_pendek = 0
                 value_sedang = (22-ukuran)/(22-17)
                 value_panjang = (ukuran-17)/(22-17)
              if ukuran > 22 :
                 value_pendek = 0
                 value_sedang = 0
                 value_panjang = 1

              #Proses Inferensi Fuzzy Servo
              servo1=[]
              #Jika ukuran pendek maka hasil keputusan adalah kecil
              def fungsiinferensi_kecil (variabel_ukuran):
                  a1 = (variabel_ukuran)
                  nz1 = a1 * 0
                  servo1.append([a1, nz1])

              servo2=[]
              def fungsiinferensi_sedang (variabel_ukuran):
                  a2 = (variabel_ukuran)
                  nz2 = 60 - a2 * (60 - 50)
                  servo2.append([a2, nz2])

              servo3=[]
              def fungsiinferensi_besar (variabel_ukuran):
                  a3 = (variabel_ukuran)
                  nz3 = 50 - a3 * (50 - 60)
                  servo3.append([a3, nz3])

              fungsiinferensi_kecil(value_pendek)
              fungsiinferensi_sedang(value_sedang)
              fungsiinferensi_besar(value_panjang)

              a_predikat1 = servo1[0][0]
              z1 = servo1[0][1]

              a_predikat2 = servo2[0][0]
              z2 = servo2[0][1]

              a_predikat3 = servo3[0][0]
              z3 = servo3[0][1]

              #Proses Fuzzyfikasi Servo
              a_pred_z = (a_predikat1*z1)+(a_predikat2*z2)+(a_predikat3*z3)
              z = a_predikat1+a_predikat2+a_predikat3
              zTotal = a_pred_z/z ;
              
              if zTotal == 0 :
                 cv2.putText(orig, "Kecil", (int(188), int(60)), cv2.FONT_HERSHEY_SIMPLEX,0.8, (0,0,255), 2)
                 kategori = 'Kecil'
              if zTotal > 0 and zTotal <= 55 :
                 cv2.putText(orig, "Sedang", (int(188), int(60)), cv2.FONT_HERSHEY_SIMPLEX,0.8, (0,0,255), 2)    
                 kategori = 'Sedang'
              if zTotal > 55 :
                 cv2.putText(orig, "Besar", (int(188), int(60)), cv2.FONT_HERSHEY_SIMPLEX,0.8, (0,0,255), 2)
                 kategori = 'Besar'  
            
            
           
        #Menampilkan Jumlah Objek yang terdeteksi
        #cv2.putText(orig, "Jumlah Objek: {}".format(hitung_objek),(5,460),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2, cv2.LINE_AA)  
        cv2.putText(orig,"Warna Ikan:",(5,30),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),2)
        cv2.putText(orig,"Kategori Ikan:",(5,60),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),2)
        cv2.imshow('Kamera',orig)
        #cv2.imshow("Deteksi Warna", frame2)
        #cv2.imshow("Pengenalan Warna 1", cnts_1)
        #cv2.imshow("Pengenalan Warna 2", cnts_2)  
        

        key = cv2.waitKey(1)
        if key == 27:
           break


cap.release()
cv2.destroyAllWindows()
