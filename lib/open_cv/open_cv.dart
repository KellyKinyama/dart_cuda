import 'package:dartcv4/dartcv.dart' as cv;

void main() {
  final img = cv.imread("image.png", flags: cv.IMREAD_COLOR);
  final gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY);
  print("${img.rows}, ${img.cols}");

  cv.imwrite("test_cvtcolor.png", gray);
}
