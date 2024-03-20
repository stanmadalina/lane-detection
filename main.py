# ex1
import cv2
import numpy as np

cam = cv2.VideoCapture('Lane Detection Test Video-01.mp4')
print(cv2.VideoCapture.read(cam))
nr = 0
left_top = 0
left_bottom = 0
right_top = 0
right_bottom = 0
prev=[0, 0, 0, 0]
while True:
    nr = nr+1
    ret, frame = cam.read()

    if ret is False:
        break

    cv2.imshow('Original', frame)

    arr = frame.shape
    # print("initial size:", arr)

# ex2
    res_frame = cv2.resize(frame, (500, 300))
    [h, w, dim] = res_frame.shape
    cv2.imshow("small", res_frame)


# ex3
    grayscale = cv2.cvtColor(res_frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow("grayscale", grayscale)

# ex4
    # a
    # i
    upper_left = (int(500*0.53), int(300*0.76))
    lower_left = (0, 300)
    upper_right = (int(500*0.47), int(300*0.76))
    lower_right = (500, 300)

    # ii
    trapezoid_points = np.array([upper_right, upper_left, lower_right, lower_left], dtype=np.int32)

    # iii
    black_frame = np.zeros((300, 500), dtype=np.uint8)
    trapezoid = cv2.fillConvexPoly(black_frame, trapezoid_points, 1)

    # cv2.imshow("trapezoid", trapezoid*255)

# b
    road = grayscale*trapezoid
    cv2.imshow("Road", road)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# ex5
    # a
    trapezoid_points = np.float32(trapezoid_points)
    fr_ul = (500, 0)
    fr_ur = (0, 0)
    fr_lr = (500, 300)
    fr_ll = (0, 300)
    frame_bounds = np.array([fr_ur, fr_ul, fr_lr, fr_ll], dtype=np.float32)
    # b
    stretch_matrix = cv2.getPerspectiveTransform(trapezoid_points, frame_bounds)
    # c
    stretched_frame = cv2.warpPerspective(road, stretch_matrix, (500, 300))
    # cv2.imshow("Stretched image", stretched_frame)

# ex6
    blur = cv2.blur(stretched_frame, (7, 7))
    cv2.imshow("Blur", blur)
# ex7
    # a
    sobel_vertical = np.float32([[-1, -2, -1],
                                [0, 0, 0],
                                [1, 2, 1]])
    sobel_horizontal = np.transpose(sobel_vertical)
    # b
    sob_ver = np.float32(blur)
    sob_hor = np.float32(blur)

    sob_ver = cv2.filter2D(sob_ver, -1, sobel_vertical)
    sob_hor = cv2.filter2D(sob_hor, -1, sobel_horizontal)

    # c
    sobel = np.sqrt(sob_ver*sob_ver+sob_hor*sob_hor)
    sobel = cv2.convertScaleAbs(sobel)
    cv2.imshow("Filter", sobel)
# ex8

    _, bi = cv2.threshold(sobel, 100, 255, cv2.THRESH_BINARY, None)

    cv2.imshow("binary", bi)

# ex9
    # a
    bi1 = bi.copy()
    wid = int(w*0.1)
    bi1[:, :wid] = 0
    bi1[:, (w-wid):wid] = 0
    cv2.imshow("binary sliced", bi1)
    # b
    h1 = bi1[:, :(w//2)]
    h2 = bi1[:, (w//2):]
    # cv2.imshow("left side", h1)
    # cv2.imshow("right side", h2)

    coord_1 = np.argwhere(h1 > 0)
    coord_2 = np.argwhere(h2 > 0)

    left_xs = coord_1[:, 1]
    left_ys = coord_1[:, 0]

    right_xs = coord_2[: , 1]+w//2
    right_ys = coord_2[:, 0]

# ex10
    # a
    line1 = np.polynomial.polynomial.polyfit(left_xs, left_ys, deg=1)
    line2 = np.polynomial.polynomial.polyfit(right_xs, right_ys, deg=1)
    # b
    left_top_y = 0
    left_top_x = (left_top_y-line1[0])/line1[1]

    left_bottom_y = h
    left_bottom_x = (left_bottom_y-line1[0])/line1[1]

    right_top_y = 0
    right_top_x = (right_top_y-line2[0])/line2[1]

    right_bottom_y = h
    right_bottom_x = (right_bottom_y - line2[0]) / line2[1]

    # c
    ok=1

    coords=[left_top_x, left_bottom_x, right_top_x, right_bottom_x]
    for x in coords:
        if -10**8 > x or x > 10**8:
            ok = 0
    # d
    if ok==1:
        lt = int(left_top_x), int(left_top_y)
        lb = int(left_bottom_x), int(left_bottom_y)

        rt = int(right_top_x), int(right_top_y)
        rb = int(right_bottom_x), int(right_bottom_y)

        prev = [lt, lb, rt, rb]

        l1 = cv2.line(bi1, lt, lb, (200, 0, 0), 5)
        l2 = cv2.line(bi1, rt, rb, (100, 0, 0), 5)
    else:
        l1 = cv2.line(bi1,  prev[0],  prev[1], (200, 0, 0), 5)
        l2 = cv2.line(bi1, prev[2], prev[3], (100, 0, 0), 5)

    cv2.imshow("line1", l1)
# ex11
    # a
    new_frame1 = np.zeros((300, 500), dtype=np.uint8)

    # b
    cv2.line(new_frame1,lt, lb, (255, 0, 0), 5)

    # c
    mag = cv2.getPerspectiveTransform(frame_bounds, trapezoid_points)
    # d
    lines1 = cv2.warpPerspective(new_frame1, mag, (500, 300))

    # cv2.imshow("lines1", lines1)

    # e
    coords1 = np.argwhere(lines1 > 0)
    left_y = coords1[:, 0]
    left_x = coords1[:, 1]

    # f
    new_frame2 = np.zeros((300, 500), dtype=np.uint8)

    cv2.line(new_frame2, rt, rb, (255, 0, 0), 5)

    lines2 = cv2.warpPerspective(new_frame2, mag, (500, 300))

    # cv2.imshow("lines2", lines2)

    coords2 = np.argwhere(lines2 > 0)

    right_x = coords2[:, 1]
    right_y = coords2[:, 0]

    # g
    final = res_frame.copy()
    final[left_y, left_x] = (50, 50, 250)
    final[right_y, right_x] = (50, 250, 50)

    cv2.imshow("Final", final)

cam.release()
cv2.destroyAllWindows()
