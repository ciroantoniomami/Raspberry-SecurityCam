import sys 
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2 as cv
from yolo.yolo2 import Yolo
import torch
from torchvision import transforms
import time
import matplotlib.pyplot as plt
import telegram_send
import telegram.bot as tl

if __name__ == "__main__":


    ANCHORS = [[(0.2309375, 0.7936855), (0.05625 , 0.339242), (0.021953 , 0.2478555   )], [(0.02875  , 0.125694 ), (0.004375 , 0.03857 ), (0.005    , 0.075047  )]]
    S = [13, 26]
    scaled_anchors = torch.tensor(ANCHORS) / (
        1 / torch.tensor(S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2))

    model = Yolo()

    

    cap = cv.VideoCapture(0)
  
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    
    start = time.perf_counter()
    count = 0

    prev_frame_time = 0
    new_frame_time = 0

    history_fps = []

    while True:
        r, frame = cap.read()
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        frame = cv.resize(frame, (416, 416))

        font = cv.FONT_HERSHEY_SIMPLEX
        new_frame_time = time.time()

        frame_tensor = transforms.ToTensor()(frame).unsqueeze_(0)
        
         
        frame, strunz= model.detect(frame, frame_tensor, scaled_anchors)
        frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
        #if strunz:
            #cv.imwrite('photo.jpg',frame)
            #telegram_send.send(messages=["ATTENTO ALLO STRUNZZ!"])
            #

        count += 1
        fps = 1/(new_frame_time-prev_frame_time)
        history_fps.append(fps)
        
        prev_frame_time = new_frame_time
        fps = "{:3.4f}".format(fps)
        fps = "FPS: " + fps
        cv.putText(frame, fps, (0, 30), font, 0.5, (255, 0, 0), 1, cv.LINE_AA)
        
        cv.imshow('detecter', frame)

 
        c = cv.waitKey(1)
        if c == 27:
            cap.release()
            cv.destroyAllWindows()
            break
        
        #time.sleep(2)
    
    plt.plot(history_fps)
    plt.ylabel("FPS")
    plt.show()
    end = time.perf_counter()
    print(count/(end-start))
