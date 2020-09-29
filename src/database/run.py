#CONVERTENDO IMAGEM BASE64 P/ PNG
import base64
from PIL import Image
from io import BytesIO
#CLASSIFICANDO IMAGEM COM BASE NO MODELO
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model

def run_convert():
    f = open('base64.txt', 'r')
    data = f.read()
    f.closed
    im = Image.open(BytesIO(base64.b64decode(data)))
    im.save('base64.png', 'PNG')

# Import da imagem e remodelagem
def load_image(filename):
	# Import da imagem
	img = load_img(filename, grayscale=True, target_size=(28, 28))
	# converção p/ array
	img = img_to_array(img)
	# Remodelagem dos dados para um único canal
	img = img.reshape(1, 28, 28, 1)
	# preparação dos pixels
	img = img.astype('float32')
	img = img / 255.0
	return img
 
# Import da imagem e do modelo de predição
def run_example():
	# load: imagem
	img = load_image('base64.png')
	# load: modelo
	model = load_model('final_model.h5')
	# Predição da classe
	digit = model.predict_classes(img)
	print(digit[0])
 
# MAIN, rodar "run_example()"
run_convert()
run_example()