"testing user's interaction and opencv function"
from pathlib import Path
from PySide2 import QtCore
from src.gui.gui import MainWind, OCRDisplay


def test_mainwindow(qtbot):
    """Check if the mainwindow is successfuly constructed

    Args:
        qtbot : makes it possible to stimulate the user interaction
        in ordder to test by adding widget and controling mouse and
        keyboards effect
    Note
    for example we have in a geometry of a button
    """
    widget = MainWind()
    qtbot.addWidget(widget)
    assert widget.mainwindow.isAnimated() == 1


def test_label(qtbot):
    "test label"
    widget = MainWind()
    qtbot.addWidget(widget)
    assert widget.label.text() == "Enter your file name "


def test_after(qtbot):
    "test label after text"
    widget = MainWind()
    qtbot.addWidget(widget)
    assert widget.after.text() == "After"


def test_before(qtbot):
    "test label after text"
    widget = MainWind()
    qtbot.addWidget(widget)
    assert widget.before.text() == "Before"


def test_radiobutton_a(qtbot):
    "tetst first radio button"
    widget = MainWind()
    qtbot.addWidget(widget)
    qtbot.mouseClick(widget.mainwindow, QtCore.Qt.NoButton, pos=QtCore.QPoint(318, 252))
    assert widget.radio_button.isChecked() == 0
    assert widget.radio_button.text() == "20fps"


def test_radiobutton_b(qtbot):
    "test 2nd radio button"
    widget = MainWind()
    qtbot.addWidget(widget)
    qtbot.mouseClick(widget.mainwindow, QtCore.Qt.NoButton, pos=QtCore.QPoint(438, 253))
    assert widget.radio_button_2.isChecked() == 0
    assert widget.radio_button_2.text() == "30fps"


def test_radiobutton_c(qtbot):
    "test 3rd radio button"
    widget = MainWind()
    qtbot.addWidget(widget)
    qtbot.mouseClick(widget.mainwindow, QtCore.Qt.NoButton, pos=QtCore.QPoint(558, 253))
    assert widget.radio_button_3.isChecked() == 0
    assert widget.radio_button_3.text() == "60fps"


def test_radiobutton_d(qtbot):
    "test radio 4th radio button "
    widget = MainWind()
    qtbot.addWidget(widget)
    qtbot.mouseClick(widget.mainwindow, QtCore.Qt.NoButton, pos=QtCore.QPoint(198, 253))
    assert widget.radio_button_4.isChecked() == 0
    assert widget.radio_button_4.text() == "10fps"


def test_radiobutton_e(qtbot):
    "test radio 5th radio button "
    widget = MainWind()
    qtbot.addWidget(widget)
    qtbot.mouseClick(widget.mainwindow, QtCore.Qt.NoButton, pos=QtCore.QPoint(48, 253))
    assert widget.radio_button_5.isChecked() == 0
    assert widget.radio_button_5.text() == "5fps"


def test_submit(qtbot):
    """test if value is passed by the widget once the pushbutton submit in position
    (525,207) is clicked"""
    widget = MainWind()
    qtbot.addWidget(widget)
    user_interaction = "street"
    qtbot.keyClicks(widget.lineedit, user_interaction)
    qtbot.mouseClick(widget.mainwindow, QtCore.Qt.NoButton, pos=QtCore.QPoint(525, 207))
    assert widget.pushbutton.sender() is None
    assert widget.lineedit.sender() is None
    assert widget.lineedit.text() == "street"


def get_userfilename(qtbot):
    """test the connection between pushbutton and get userfilename method"""
    widget = MainWind()
    qtbot.addWidget(widget)
    user_interaction = "street"
    user_interaction2 = "street2"
    qtbot.keyClicks(widget.lineedit, user_interaction)
    qtbot.keyClicks(widget.lineedit, user_interaction2)
    assert widget.get_userfilename() == "street"
    assert widget.get_userfilename() == "street2"


def test_connectionbutton(qtbot):
    """check if buttons are successfully constructed

    Args:
        qtbot ([type]): [description]
    """
    widget = MainWind()
    qtbot.addWidget(widget)
    assert widget.launch.text() == 'Launch OCR'
    assert widget.execute.text() == "Execute"
    assert widget.pushbutton.text() == "Submit"


def test_ocr_standard(qtbot):
    """
    Test ocr window and check if values are passed right and textbrowser display output
    textbrowser cleared and inserted as there are files in the folder
    """
    widget = OCRDisplay()
    qtbot.addWidget(widget)
    ocrfile = "lol"
    qtbot.keyClicks(widget.lineocr, ocrfile)
    widget.textbrowser.clear()
    widget.textbrowser.append("2")
    text = widget.textbrowser.toPlainText()
    assert widget.label.text() == "Welcome !! Plates will be shown now"
    assert widget.label_2.text() == "For a cropped image of the ls paste ls number"
    assert widget.lineocr.text() == "lol"
    assert text == "2"


def test_frametovideo(qtbot):
    """ the execution of this module requires a lot of time as it divides the video into frames
    and takes time to be executed and should check if exactly n frames have been extracted
    testing with a forced 2fps"""
    widget = MainWind()
    qtbot.addWidget(widget)
    widget.path_vid = Path(__file__).parent / 'Tests'
    widget.inpath = str(Path(__file__).parent) + "/Tests/test extraction/"
    user_interaction = "vid1"
    qtbot.keyClicks(widget.lineedit, user_interaction)
    qtbot.mouseClick(widget.mainwindow, QtCore.Qt.NoButton, pos=QtCore.QPoint(198, 253))
    qtbot.mouseClick(widget.mainwindow, QtCore.Qt.NoButton, pos=QtCore.QPoint(689, 249))
    assert widget.videotoframes() == 15


def test_write(qtbot):
    """ the execution of this module requires a lot of time as it divides the video into frames
    and takes time to be executed and should check if exactly n frames have been extracted
    testing with a forced 2fps"""
    widget = MainWind()
    qtbot.addWidget(widget)
    widget.editedframes = str(Path(__file__).parent) + "/Tests/test extraction/"
    widget.filecreate = str(Path(__file__).parent) + "/Tests/test2/test.mp4"
    assert widget.write_video() == 15


def test_license_plate(qtbot):
    """test if the license plates are in textbrowser displayed
    the \n represents a new line"""
    widget = OCRDisplay()
    qtbot.addWidget(widget)
    widget.path_ocr = Path(__file__).parent / 'Tests/test3'
    widget.textbrowser.clear()  # make sure textborwser clear
    widget.license_plate()
    license_test = widget.textbrowser.toPlainText()
    assert license_test == "AICRM424\nIII"


def test_guiexit(qtbot):
    "check if the mainwindow is closed "
    widget = MainWind()
    qtbot.addWidget(widget)
    assert widget.mainwindow.close()
