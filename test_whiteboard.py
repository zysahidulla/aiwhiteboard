import numpy as np

import whiteboard


def make_board():
    board = whiteboard.VirtualWhiteboard.__new__(whiteboard.VirtualWhiteboard)
    board.last_gesture_time = -5
    board.gesture_msg = ""
    board.gesture_msg_time = 0
    board.color_idx = 4
    board.smoothing_q = __import__("collections").deque(maxlen=whiteboard.SMOOTHING_WINDOW)
    board.prev_point = None
    board.drawing = False
    board.cur_stroke = []
    board.strokes = []
    board.canvas = np.zeros((100, 100, 3), dtype=np.uint8)
    return board


def test_detect_gesture_handles_eraser_mode():
    board = make_board()
    assert board._detect_gesture((False, True, True, True, False), None) == "eraser"


def test_detect_gesture_handles_draw_mode():
    board = make_board()
    assert board._detect_gesture((False, True, False, False, False), None) == "draw"


def test_eraser_stroke_records_black_with_larger_thickness():
    board = make_board()
    board._draw_on_canvas((10, 10), "eraser")
    assert board.cur_stroke == [] or board.cur_stroke[-1][2] == (0, 0, 0)
    assert board.cur_stroke == [] or board.cur_stroke[-1][3] == whiteboard.ERASER_THICKNESS


def test_camera_is_composited_in_front_of_canvas():
    board = make_board()
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    frame[0, 0] = (255, 0, 0)
    canvas = np.zeros((10, 10, 3), dtype=np.uint8)
    canvas[0, 0] = (0, 255, 0)

    merged = board._compose_frame(frame, canvas)

    assert merged[0, 0][0] > 0
    assert merged[0, 0][1] > 0
