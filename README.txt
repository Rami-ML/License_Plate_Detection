Installation
*Make sure to install the requirements with pip install -r requirements.txt . 
Note : TensorFlow needs a 64 bit version of python (OpenCV requirement)


Features / description : 
*The programm enables the user to make detection on a wished mp4 video 
and creates a mp4 file in src/stabilization back 


Usage and programm start :
1-You need to launch the gui in src/gui/gui.py 
2- Write the name of the video you want to detect without ".mp4" extension
!!! Note : it needs to be in src/gui 
3- if you want to display the video press play on the lefter under the labeltext "before"
4-check a frames number that you want to cut the video with and press set
5-wait till video is extracted
6-click exectue so stb.py launches
7-if you want to check the license plate numbers press ocr launch 
	- in this window plate numbers will be displayed
	- to view a cropped image : enter your file name without extension 
	as ocr create both extension and press the button 
8- in order to receive your video press on the right play and wait
	to set output fps change it in write_video function under fps
	-ignore the callback warning as it is due to compatibility with mp4
9- Enjoy your video

Note ! press q to exit video



