import pygame
from PIL import Image
import numpy as np
import keras.models

class Models:
    convolutional_neural_network = None
    result = -1
    def load():
        Models.convolutional_neural_network = keras.models.load_model("CNN\Digits_detection_keras")

    def predict(img : Image.Image):
        img = img.resize((28,28))
        arr = np.array(img).reshape(-1,28,28,1)
        
        results = Models.convolutional_neural_network.predict(arr)
        relts = [result.argmax() for result in results]
        print(results)
        Models.result = relts
        return relts

class Rect_crop:
    def __init__(self) -> None:
        self.x1 = 0
        self.y1 = 0
        self.x2 = 0
        self.y2 = 0
        self.mode = 0
    
    def set(self, pos, action = 'None'):
        if self.mode == 0:
            self.x1 = pos[0]
            self.y1 = 600 - pos[1]
            if action == "click":
                self.mode = 1
        elif self.mode == 1:
            self.x2 = pos[0]
            self.y2 = 600 - pos[1]
            if action == "click":
                global img_crop 
                img_crop = pygame.Surface((self.x2-self.x1, self.y1-self.y2))
                img_crop.blit(Main_window_copy, (0,0), (self.x1,600- self.y1, self.x2-self.x1, self.y1-self.y2))
                lists.append(Object_draw(img_crop, (600,0)))

                array = pygame.surfarray.array3d(img_crop)

                img = Image.fromarray(array, mode='RGB').convert(mode='L')
                array = np.array(img).T
                array = np.ones_like(array) * 255 - array
                img = Image.fromarray(array, mode='L')
                img.save("image.png")
                Models.predict(img)
                self.mode = 2
        else:
            if action == "click":
                self.mode = 0
    
    def draw(self):
        if self.mode >= 0:
            rect = pygame.Rect(self.x1,600- self.y1, 10,10)
            rect.center = (self.x1,600- self.y1)
            pygame.draw.rect(Main_window, (0,255,0), rect)
        if self.mode >= 1:
            pygame.draw.line(Main_window, (0,255,0), (self.x1,600- self.y1), (self.x2,600- self.y1))
            pygame.draw.line(Main_window, (0,255,0), (self.x1,600- self.y1), (self.x1,600- self.y2))
            pygame.draw.line(Main_window, (0,255,0), (self.x2,600- self.y2), (self.x1,600- self.y2))
            pygame.draw.line(Main_window, (0,255,0), (self.x2,600- self.y2), (self.x2,600- self.y1))
            rect = pygame.Rect(self.x2,600-  self.y2, 10,10)
            rect.center = (self.x2,600-  self.y2)
            pygame.draw.rect(Main_window, (0,255,0), rect)

class Object_draw:
    def __init__(self, surface, pos) -> None:
        self.surface = surface
        self.pos = pos

Rect = Rect_crop()
lists = []
pygame.init()

Models.load()
Main_window = pygame.display.set_mode((1000,600))
img = pygame.image.load("Detect_Digits\data\images.png")
w,h = img.get_size()
lists.append(Object_draw(img, ((600-w)/2,(600-h)/2)))

img_crop = None

while True:
    for event in pygame.event.get():
        if event.type == 256:
            exit(0)
        if event.type == 1024:
            Rect.set(event.pos)
        if event.type == 1025:
            Rect.set(event.pos, "click")

    Main_window.fill((0,0,0))
    for obj in lists:
        Main_window.blit(obj.surface, obj.pos, (0,0,600,600))
    if Models.result != -1:
        font = pygame.font.Font('freesansbold.ttf', 32)
        text = font.render(str(Models.result), True, (255,255,255))
        Main_window.blit(text, (601,200))
    pygame.draw.line(Main_window, (255,255,255), (600,0), (600,600))
    Main_window_copy = Main_window.copy()
    Rect.draw()

    pygame.display.update()
    
    

    