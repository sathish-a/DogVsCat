from PIL import Image
from os import listdir

dir = "/home/sam/PycharmProjects/classify/datasets/"

def reshape(name):
    print("Reshaping " + name + "...")
    i = 0
    for file in listdir(dir + name):
        if not file.startswith('.') and file != 'Thumbs.db':
            i = i + 1
            print(i)
            images = Image.open(dir+ name + "/" + file)
            imResize = images.resize((200, 200), Image.ANTIALIAS)
            imResize.save(dir + name + "/" + file, 'JPEG', quality=90)

reshape("test")
