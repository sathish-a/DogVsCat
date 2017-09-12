from PIL import Image
from os import listdir


def reshape(name):
    print("Reshaping " + name + "...")
    i = 0
    for file in listdir("datasets/" + name):
        if not file.startswith('.') and file != 'Thumbs.db':
            i = i + 1
            print(i)
            images = Image.open("datasets/" + name + "/" + file)
            imResize = images.resize((200, 200), Image.ANTIALIAS)
            imResize.save("datasets/" + name + "/" + file, 'JPEG', quality=90)


def mov_file():
    i = 0
    for file in listdir("train/"):
        if not file.startswith('.') and file != 'Thumbs.db':
            i = i + 1
            file_name_split = file.split('.')
            file_name = file_name_split[0]
            file_lab = file_name_split[1]
            print(file_name, file_lab, i)
            if file_name == 'cat':
                save_dir = "cat/" + file_lab + ".jpg"
            elif file_name == 'dog':
                save_dir = "dog/" + file_lab + ".jpg"
            images = Image.open("train/" + file)
            imResize = images.resize((200, 200), Image.ANTIALIAS)
            imResize.save(save_dir, 'JPEG', quality=90)


mov_file()
