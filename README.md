# snakeNaRuchyReka
## Ewa Krzaczkowska

Celem projektu było stworzenie sieci neuronowej służącej do rozpoznawania wybranych gestów ręki. Aplikacja przy użyciu nauczonej sieci powinna zidentyfikować gest i na tej podstawie sterować ruchem węża w zaimplementowanej grze.  

Aplikacje zaimplementowano w Pythonie w środowisku Debian. Podczas budowania i uczenia modelu korzystano z biblioteki TensorFlow keras 2.0. Natomiast do przetwarzania obrazów skorzystano z OpenCV oraz NumPy. A przy tworzeniu gry użyto Pygame.  

W stworzonym projekcie skupiono się na wykryciu 6 gestów:
1. [x] Fist 
2. [x] Hand 
3. [x] Ok 
4. [x] Turn left
5. [x] Turn right
6. [x] One finer  

Dane utworzono samodzielnie poprzez wykonanie dla każdego gestu 600 zdjęć treningowych i 240 zdjęć weryfikacyjnych. Przydatna okazała się do tworzenia danych uczących biblioteka ImageDataGenerator. W celu poprawy dokładności wykrywania za każdym kolejnym uczeniem dane wejściowe przy jej pomocy poddawano losowym transformacjom obrazów   

