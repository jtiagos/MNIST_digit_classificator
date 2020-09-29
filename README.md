####IMPORTANTE
MINICONDA3 como prompt
VIRTUALENV VENV para criação de "Containers" isolados dentro do PC, para evitar problemas de incompatibilidade de libs
conda install virtualenv
REF:https://www.youtube.com/watch?v=Pz9rayiDHW0&list=PL3BqW_m3m6a05ALSBW02qDXmfDKIip2KX&index=2
requirements.txt possui todas as informações de libs usadas nos códigos dentro do VENV
O Modelo Classificador foi compilado dentro do Google Colab para acelerar o processo, mas roda tranquilo dentro do VENV se todas as libs estiverem corretas.

####Usabilidade:

executar "app.py"
Abrir navegador preferencialmente chrome em localhost http://127.0.0.1:4555/
Fazer UPLOAD de arquivo base64. OBS: Usar nome "base64.txt" para o arquivo.
Verificar tela de UPLOAD COMPLETO
Abrir pasta DATABASE
Executar "run.py"
O modelo irá converter o base64 para ".png" e então classificá-lo usando o "final_model.h5"
OBS: como o modelo foi treinado usando MNIST com escrita em branco e fundo preto, seguir o padrão.
OBS: O modelo assume que as novas imagens estão em tons de cinza, que foram alinhadas de forma que uma imagem contenha um dígito manuscrito centralizado e 
que o tamanho da imagem é quadrado com 28 × 28 pixels. Evitar fazer upload de fotos com maior resolução para evitar problemas de reshape e/ou classificador.
Verificar a saída do terminal para o apontamento do resultado!!

####MODELS
Dentro da pasta models você encontrará todos os recursos empregados no conjunto da obra.
Caso precise converter alguma imagem colorida para binarizada inversa, para que o modelo compile e classifique, utilizar "binary_inv.py"

####TEST FILES
Em "test files" você encontrará alguns arquivos que podem ser úteis para a validação do modelo. Apenas como sugestão!!
Atenção a regra de nomes para o caso de utilização....
####REFS

Adaptado dos seguintes autores referenciais:

https://machinelearningmastery.com/how-to-develop-a-convolutional-neural-network-from-scratch-for-mnist-handwritten-digit-classification/

https://www.youtube.com/watch?v=Pz9rayiDHW0&list=PL3BqW_m3m6a05ALSBW02qDXmfDKIip2KX&index=2

https://github.com/ibrahimokdadov/upload_file_python





