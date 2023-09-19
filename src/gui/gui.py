"""gui"""
import sys
from sys import argv
from pathlib import Path
import os
from os.path import isfile, join
import cv2  # type: ignore
from PySide2 import QtCore
from PySide2 import QtGui
from PySide2 import QtWidgets
from src.stabilisierung.stb import execute


class MainWind:  # pylint: disable=R0902
    """setting gui mainwindow which compose of different pushbuttons and lineedits that
    triggers and call functions once they are pressed
    Submit : triggers to get the name of the file for the lineedit where the user enter
    his file name and launches videotoframe where the video is cut to frames
    Set : triggers checkbox_fps once a Qradiobutton is clicked and set is pressed
    Play 1 : Triggers just show function where the video is displayed before edited
    Play 2 : Start creating an mp 4 file in src/stabilisierung from the frames after
    the execution of stb.py
    execute : triggers and gives the start signal for stb
    """

    def __init__(self):  # pylint: disable=R0915
        """ before was setupUi and with a call """
        self.mainwindow = QtWidgets.QMainWindow()
        self.mainwindow.setObjectName("mainwindow")
        self.mainwindow.resize(800, 333)
        self.mainwindow.setWindowTitle("Pyt Detector")
        path_icon = Path(__file__).parent
        self.path_vid = Path(__file__).parent
        self.pathwrite = Path(__file__).parent.parent / 'stabilisierung'
        self.inpath = str(self.pathwrite) + '\\' + 'input_frames\\'
        self.editedframes = str(self.pathwrite) + '\\' + 'output_frames\\'
        self.filecreate = str(self.pathwrite) + '\\' + "street.mp4"
        self.mainwindow.setWindowIcon(QtGui.QIcon(str(path_icon) + "\\Icons\\" + 'ai.png'))
        self.centralwidget = QtWidgets.QWidget(self.mainwindow)
        self.centralwidget.setObjectName("centralwidget")
        self.before = QtWidgets.QLabel(self.centralwidget)
        self.before.setGeometry(QtCore.QRect(210, 40, 71, 21))
        self.mainwindow.setCentralWidget(self.centralwidget)
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(170, 230, 111, 20))
        self.label.setObjectName("label")
        self.label.setText("Enter your file name ")
        self.lineedit = QtWidgets.QLineEdit(self.centralwidget)
        self.lineedit.setGeometry(QtCore.QRect(290, 230, 281, 20))
        self.lineedit.setObjectName("lineEdit")
        self.pushbutton = QtWidgets.QPushButton(self.centralwidget)
        self.pushbutton.setGeometry(QtCore.QRect(600, 230, 75, 23))
        self.pushbutton.setObjectName("Submit")
        self.pushbutton.setText("Submit")
        self.pushbutton.clicked.connect(self.get_userfilename)
        self.lineedit.returnPressed.connect(self.just_show_read_video)
        self.lineedit.returnPressed.connect(self.videotoframes)
        self.pushbutton.clicked.connect(self.videotoframes)
        self.pushbutton.clicked.connect(self.videotoframes)
        self.secondwindow = QtWidgets
        QtCore.QMetaObject.connectSlotsByName(self.mainwindow)
        font = QtGui.QFont()
        font.setPointSize(16)
        font.setBold(True)
        font.setWeight(75)
        self.before.setFont(font)
        self.before.setObjectName("Before")
        self.before.setText("Before")
        self.after = QtWidgets.QLabel(self.centralwidget)
        self.after.setGeometry(QtCore.QRect(540, 30, 101, 41))
        self.after.setText("After")
        font = QtGui.QFont()
        font.setPointSize(16)
        font.setBold(True)
        font.setWeight(75)
        self.after.setFont(font)
        self.after.setObjectName("After")
        self.radio_button = QtWidgets.QRadioButton(self.centralwidget)
        self.radio_button.setGeometry(QtCore.QRect(400, 270, 82, 17))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.radio_button.setFont(font)
        self.radio_button.setObjectName("X1.25")
        self.radio_button_2 = QtWidgets.QRadioButton(self.centralwidget)
        self.radio_button_2.setGeometry(QtCore.QRect(520, 270, 82, 17))
        font = QtGui.QFont()
        font.setPointSize(8)
        font.setBold(True)
        font.setWeight(75)
        self.radio_button_2.setFont(font)
        self.radio_button_2.setObjectName("X1.5")
        self.radio_button_3 = QtWidgets.QRadioButton(self.centralwidget)
        self.radio_button_3.setGeometry(QtCore.QRect(640, 270, 82, 17))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.radio_button_3.setFont(font)
        self.radio_button_3.setObjectName("X1.75")
        self.radio_button_4 = QtWidgets.QRadioButton(self.centralwidget)
        self.radio_button_4.setGeometry(QtCore.QRect(280, 270, 82, 17))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.radio_button_4.setFont(font)
        self.radio_button_4.setObjectName("X1")
        self.radio_button_5 = QtWidgets.QRadioButton(self.centralwidget)
        self.radio_button_5.setGeometry(QtCore.QRect(130, 270, 82, 17))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.radio_button_5.setFont(font)
        self.radio_button_5.setObjectName("X0.8")
        self.radio_button.setText("20fps")
        self.radio_button_2.setText("30fps")
        self.radio_button_3.setText("60fps")
        self.radio_button_4.setText("10fps")
        self.radio_button_5.setText("5fps")
        self.play1 = QtWidgets.QPushButton(self.centralwidget)
        self.play1.setGeometry(QtCore.QRect(180, 90, 131, 121))
        self.play1.setObjectName("Play")
        self.play2 = QtWidgets.QPushButton(self.centralwidget)
        self.play2.setGeometry(QtCore.QRect(510, 90, 131, 121))
        self.play2.setObjectName("Play")
        self.play1.setText("Play")
        self.play2.setText("Play")
        self.play1.setIcon(QtGui.QIcon(str(path_icon) + "\\Icons\\play button.png"))
        self.play1.setIconSize(QtCore.QSize(24, 24))
        self.play2.setIcon(QtGui.QIcon(str(path_icon) + "\\Icons\\play button.png"))
        self.play2.setIconSize(QtCore.QSize(24, 24))
        self.execute = QtWidgets.QPushButton(self.centralwidget)
        self.execute.setGeometry(QtCore.QRect(370, 110, 81, 23))
        self.execute.setObjectName("Execute")
        self.execute.setText("Execute")
        self.launch = QtWidgets.QPushButton(self.centralwidget)
        self.launch.setGeometry(QtCore.QRect(370, 180, 81, 23))
        self.launch.setObjectName("Launch")
        self.launch.setText("Launch OCR")
        self.setfps = QtWidgets.QPushButton(self.centralwidget)
        self.setfps.setGeometry(QtCore.QRect(730, 270, 41, 21))
        self.setfps.setObjectName("set")
        self.setfps.setText("Set")
        self.play1.clicked.connect(self.just_show_read_video)
        self.launch.clicked.connect(self.start_window2)
        self.play2.clicked.connect(self.write_video)
        self.execute.clicked.connect(execute)
        self.setfps.clicked.connect(self.checkbox_fps)

    def start_window2(self):
        """Start the display of Ocr_detections """
        self.secondwindow = OCRDisplay()
        self.secondwindow.show()

    def checkbox_fps(self):
        """give the factor parameter in  vid_cap.set
        (cv2.CAP_PROP_POS_MSE(frame_numbers * factor * 1000))
        which refers indirectly to the fps wished
        Returns:
            float: passes the fps value to the videotoframe function called in
            the variable factor
            as vid_cap.set(cv2.CAP_PROP_POS_MSE(frame_numbers * factor * 1000))
            needs factor to be 1 second divided by frames per second wanted

        """
        if self.radio_button.isChecked():
            print("20fps checked")
            return 0.05
        if self.radio_button_2.isChecked():
            print("30fps checked")
            return 0.03
        if self.radio_button_3.isChecked():
            print("60fps checked")
            return 0.016
        if self.radio_button_4.isChecked():
            print("10fps checked")
            return 0.1
        if self.radio_button_5.isChecked():
            print("5fps checked")
            return 0.2
        return 0.5  # this value is for the test in order to have 2fps

    def get_userfilename(self):
        """get file name of the video from the user

        Returns:
            str: a string that contains the value of text entered by
            the user in lineedit
        """
        text = self.lineedit.text()
        print(text)
        return text

    def just_show_read_video(self):
        """"Show video before it gets modified
            Resized video so the display will not be huge
            not please place the video that you want to be tested
            in src/gui
            Returns:
            int: check if frames are read"""

        path_int = str(self.path_vid) + "/" + self.lineedit.text() + ".mp4"
        print(path_int)
        cap = cv2.VideoCapture(path_int)
        counter_test = 0
        while True:
            ret, frame = cap.read()  # pylint: disable=W0612
            cv2.namedWindow('vid', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('vid', 600, 600)
            try:
                # so video will not crash and wait for a q at the end of the video
                cv2.imshow('vid', frame)
            except Exception:  # pylint: disable=W0703
                # Do not use bare exept in this case counter_test is just to surpress
                counter_test = counter_test + 0
            counter_test = counter_test + 1
            if cv2.waitKey(1) & 0xFF == ord('q'):  # Should check for zero
                break
        cap.release()
        cv2.destroyAllWindows()
        return counter_test

    def videotoframes(self):
        """divide the video into frames and store them in src/stabiliesierung/input_frames
            so Stabilisierung can execute all modules and produce images with detections
            in outputframes . The number of frames is set by the checkboxes""

        Returns:
            int: return number of frames with will allow us to track if the function is exectued

        """
        frame_numbers = 0
        path_read = str(self.path_vid) + "/" + self.lineedit.text() + ".mp4"
        vid_cap = cv2.VideoCapture(path_read)
        ret, image = vid_cap.read()
        ret = True
        factor = self.checkbox_fps()
        while ret:
            vid_cap.set(cv2.CAP_PROP_POS_MSEC, (frame_numbers * factor * 1000))
            ret, image = vid_cap.read()
            if ret:
                print('new frame: ', ret)
                cv2.imwrite(str(self.inpath) + "frame%d.jpg" % frame_numbers, image)
                frame_numbers = frame_numbers + 1
        print(frame_numbers)
        message = QtWidgets.QMessageBox()
        if frame_numbers != 0 and factor != 0.5:
            message.setWindowTitle("Video extraction")
            message.setText(str(frame_numbers) + " frames have been succesfully extracted")
            message.setWindowIcon(QtGui.QIcon(str(self.path_vid) + "/Icons/greentick.png"))
            message.exec_()
        if frame_numbers == 0 and factor != 0.5:
            message.setWindowTitle("Video extraction")
            text_error_line1 = "Please make sure you entered a valid name that exists in gui "
            text_error_line2 = "do not write .mp4 and set an fps before"
            text_error = text_error_line1 + text_error_line2
            message.setText(text_error)
            message.setWindowIcon(QtGui.QIcon(str(self.path_vid) + "/Icons/error.png"))
            message.exec_()
        return frame_numbers

    def write_video(self):
        """Convert the frames stored with boundary in boxes drawn to a video and store it
            please wait for the video to be created it will take time
            Note : Callback that is displaying in the terminal while executing
            it is due to videoWriter as it is forced to creat a mp4 file """
        image_array = []
        files = [i for i in os.listdir(self.editedframes) if isfile(join(self.editedframes, i))]
        files.sort(key=lambda x: int(x[5:-4]))
        frames_number = len(files)
        frame_written = 0
        fps = 10  # set fps from here for better detection
        for i in range(frames_number):
            img = cv2.imread(self.editedframes + files[i])
            size = (img.shape[1], img.shape[0])
            img = cv2.resize(img, size)
            image_array.append(img)
            fourcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')
            out = cv2.VideoWriter(self.filecreate, fourcc, fps, size)
            indirectly_frames = len(image_array)
            frame_written = frame_written + 1
            for k in range(indirectly_frames):
                out.write(image_array[k])
            out.release()
        print("it is done open your video in stabilisierung under name street")
        return frame_written


class OCRDisplay(QtWidgets.QWidget):  # pylint: disable=R0902
    """
    This window display the output of ocr.py which creates files .jpg or .jpeg that contains
    the liecense plate number .
    This will be shown and the user will be asked to enter one of the license plate number to
    show a cropped image of the license plate
    Note : Ocr provides 2 types of files that's why there is a need of specification of file type
    Args:
        QtWidgets ([Qtwidget]): in this window another style will be demonstrated all Widgets
        elemets are created with a virtual box layout that dispay them with a descinding order
        of the declaration
        QVBoxLayout : Constructs a new top-level vertical box with parent parent.
    """

    def __init__(self, parent=None):
        super(OCRDisplay, self).__init__(parent)
        self.path_ocr = Path(__file__).parent.parent / 'ocr/Licenseplates'
        self.path_getlicence = self.path_ocr
        layout = QtWidgets.QVBoxLayout()
        self.label = QtWidgets.QLabel()
        self.label.setGeometry(QtCore.QRect(10, 10, 271, 21))
        self.label.setObjectName("label")
        self.label.setText("Welcome !! Plates will be shown now")
        self.textbrowser = QtWidgets.QTextBrowser()
        self.textbrowser.setGeometry(QtCore.QRect(70, 30, 256, 192))
        self.textbrowser.setObjectName("textBrowser")
        self.label_2 = QtWidgets.QLabel()
        self.label_2.setGeometry(QtCore.QRect(20, 230, 361, 21))
        self.label_2.setObjectName("label_2")
        self.label_2.setText("For a cropped image of the ls paste ls number")
        self.horizontallayoutwidget = QtWidgets.QWidget()
        self.horizontallayoutwidget.setGeometry(QtCore.QRect(100, 260, 221, 31))
        self.horizontallayoutwidget.setObjectName("horizontallayoutwidget")
        self.horizontallayout = QtWidgets.QHBoxLayout(self.horizontallayoutwidget)
        self.horizontallayout.setContentsMargins(0, 0, 0, 0)
        self.horizontallayout.setObjectName("horizontalLayout")
        self.lineocr = QtWidgets.QLineEdit(self.horizontallayoutwidget)
        self.lineocr.setObjectName("lineocr")
        self.horizontallayout.addWidget(self.lineocr)
        self.ocrcheck = QtWidgets.QPushButton("show snipped ls")
        self.horizontallayout.addWidget(self.ocrcheck)
        layout.addWidget(self.label)
        layout.addWidget(self.textbrowser)
        layout.addWidget(self.label_2)
        layout.addWidget(self.lineocr)
        layout.addWidget(self.ocrcheck)
        layout.addWidget(self.horizontallayoutwidget)
        self.setLayout(layout)
        self.setWindowTitle("Det")
        # connect is with a () to indirectly launch license plate
        self.ocrcheck.clicked.connect(self.license_plate())
        self.ocrcheck.clicked.connect(self.show_plate)

    def license_plate(self):
        """Iteriate the folder ocr/licenseplates as the name reference to the plate name
        Notes:
            some files are stored as .jpeg and some are stored as .jpg so it needs to be
            removed while displayed
        """
        plates = [i for i in os.listdir(self.path_ocr) if isfile(join(self.path_ocr, i))]
        number_of_plates = len(plates)
        for j in range(number_of_plates):
            text_origin = plates[j]
            if plates[j].endswith(".jpg"):
                text_modified = text_origin[:-4]
                self.textbrowser.append(text_modified)
            if plates[j].endswith(".jpeg"):
                text_modified = text_origin[:-5]
                self.textbrowser.append(text_modified)
            print(text_modified)

    def show_plate(self):
        """Shows a cropped image of the license plate
        in order to execute right the name and press enter
        write the extension type because ocr make either jpg or jpeg"""
        path_get = str(self.path_getlicence) + "\\" + self.lineocr.text() + ".jpg"
        plate_snip = cv2.imread(path_get)
        cv2.imshow(self.lineocr.text(), plate_snip)
        cv2.waitKey(0)
        return 5


if __name__ == "__main__":
    PATH = Path(__file__).parent
    app = QtWidgets.QApplication(argv)
    mainwindow = QtWidgets.QMainWindow()
    ui = MainWind()
    # print(videotoframes())
    """setupUi(mainwindow)"""
    ui.mainwindow.show()
    sys.exit(app.exec_())
