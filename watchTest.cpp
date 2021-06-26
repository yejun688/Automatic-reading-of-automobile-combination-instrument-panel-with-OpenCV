#include <cv.h>
#include <highgui.h>
#include "pa_file.h"
using namespace cv;
using namespace std;
#define  OPENSCALAR 0.1

void  findMinMax(const Mat& mask, Point centerBase, Point& top, Point& center  )
{
	vector<Point> vec_keypoint;
	for (int i = 0; i < mask.rows; i++)	
	{
		for (int j = 0; j < mask.cols; j++)
		{
			if (mask.at<unsigned char>(i, j) == 255)
			 {
				vec_keypoint.push_back(Point(j, i));
			 }
	    }
    }
   sort(vec_keypoint.begin(), vec_keypoint.end(), [](const Point& p1, const Point& p2)
	{return p1.x < p2.x; });//����x��������

	//����ŷʽ����
	float tmpx = (vec_keypoint[0] - centerBase).x;
	float tmpy = (vec_keypoint[0] - centerBase).y;
	float distance_1 = pow(tmpx, 2) + pow(tmpy, 2);
	tmpx = (vec_keypoint[vec_keypoint.size() - 1] - centerBase).x;
	tmpy = (vec_keypoint[vec_keypoint.size() - 1] - centerBase).y;
	float distance_2 = pow(tmpx, 2) + pow(tmpy, 2);

	if (distance_1 < distance_2){
		center = vec_keypoint[0];
		top = vec_keypoint[vec_keypoint.size() - 1];
	}
	else{
		top = vec_keypoint[0];
		center = vec_keypoint[vec_keypoint.size() - 1];
	}
}

Mat getBg(const Mat& mask, const Mat& frame){
	Mat backGround = frame.clone();
	for (int i = 0; i < frame.rows; i++){
		for (int j = 0; j < frame.cols; j++){
			if (mask.at<unsigned char>(i, j) == 255){
				backGround.at<Vec3b>(i, j)[0] = 0;
				backGround.at<Vec3b>(i, j)[1] = 0;
				backGround.at<Vec3b>(i, j)[2] = 0;
			}
		}
	}
	return backGround;
}

void diffMask(Mat& mask){

	vector<vector<Point>>contours;
	vector<Vec4i>hierarchy;
	findContours(mask, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);

	int maxArea = 0;
	int maxIndex = 0;
	mask.setTo(0);
	for (int i = 0; i<contours.size(); i++){
		if (contours[i].size() > 10)
		{
			drawContours(mask, contours, i, Scalar(255), CV_FILLED);
		}

	}
}

vector<Point> findMaxArea(Mat& mask){

	vector<vector<Point>>contours;
	vector<Vec4i>hierarchy;
	findContours(mask, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);

	int maxArea = 0;
	int maxIndex = 0;
	mask.setTo(0);
	for (int i = 0; i<contours.size(); i++){
		if (contours[i].size() > maxArea){
			maxArea = contours[i].size();
			maxIndex = i;
		}
	}
	drawContours(mask, contours, maxIndex, Scalar(255), CV_FILLED);
	return contours[maxIndex];
}


Point getCenter(const Mat& mask, Mat& centerMask, int& radio){

	Mat tmpMask = mask.clone();
	vector<Point> vec_keypoint = findMaxArea(tmpMask);
	Rect box = boundingRect(vec_keypoint);
	int base_length = box.width < box.height ? box.width : box.height;
	radio = base_length;
	int dstLenght = OPENSCALAR*base_length;
	if (dstLenght % 2 != 0){
		dstLenght += 1;
	}

	Mat element = getStructuringElement(MORPH_RECT, Size(dstLenght, dstLenght));
	morphologyEx(tmpMask, tmpMask, MORPH_OPEN, element);  //������
	centerMask = tmpMask.clone();
	//Ѱ�����ĵ�
	Point p;
	int loop = 0;
	for (int i = 0; i < tmpMask.rows; i++){
		for (int j = 0; j < tmpMask.cols; j++){
			if (tmpMask.at<unsigned char>(i, j) == 255){
				p.y = p.y + i;
				p.x = p.x + j;
				loop++;
			}
		}
	}
	p.x = p.x / loop;
	p.y = p.y / loop;
	return p;
}

Point getPointCenter(vector<Point>vec_point){
	Point out(0, 0);
	for (int i = 0; i < vec_point.size(); i++){
		if (out == Point(0, 0)){
			out = vec_point[i];
		}
		else{
			out.x = out.x*0.5 + vec_point[i].x*0.5;
			out.y = out.y*0.5 + vec_point[i].y*0.5;
		}
	}
	return out;
}

float pointDistance(const Point& p1, const Point& p2){
	Point diff = p1 - p2;
	float tmpx = diff.x;
	float tmpy = diff.y;
	return sqrt(pow(tmpx, 2) + pow(tmpy, 2));
}

//������,�Ҽ�����
bool drawAtm = false;
vector<Point> roiArea;
void drawPoint(int event, int x, int y, int flags, void *param)
{
	switch (event)
	{
	case CV_EVENT_MOUSEMOVE:
		break;

	case CV_EVENT_LBUTTONDOWN:
		break;

	case CV_EVENT_LBUTTONUP:
		roiArea.push_back(Point(x, y));
		cout << Point(x, y) << endl;
		break;
	case CV_EVENT_RBUTTONDOWN:
		drawAtm = true;
		break;
	}
}


void drawcircle(Mat& frame, const RotatedRect& box){
	vector<vector<Point>>contours;
	vector<Vec4i>hierarchy;
	ellipse(frame, box, Scalar(255), 1, CV_AA);
	Mat mask = frame.clone();
	findContours(mask, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
	vector<Point>vec_point;
	//��Բ����ͼ���ʱ�򣬲�����ͨ��������ǿ�����ߣ��ҵ�����
	for (int i = 0; i < contours[0].size(); i++){
		if (contours[0][i].y == 2 || contours[0][i].y == frame.rows - 2){
			vec_point.push_back(contours[0][i]);
		}
	}
	if (vec_point.size()>1)
	{
		for (int i = 0; i < vec_point.size(); i++){
			if (i != vec_point.size() - 1){
				line(frame, vec_point[i], vec_point[i + 1], Scalar(255));
			}
			else{
				line(frame, vec_point[i], vec_point[0], Scalar(255));
			}
		}
		mask = frame.clone();
		findContours(mask, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
	}
	frame.setTo(0);
	drawContours(frame, contours, 0, Scalar(255), CV_FILLED);
}

float calAngle(Point center, Point basePoint, Point curP){
	//���Ҷ�����cosֵ
	float a1 = pointDistance(center, basePoint);
	float a2 = pointDistance(center, curP);
	float a3 = pointDistance(curP, basePoint);    //c
	float  dst = (a1*a1 + a2*a2 - a3*a3) / (2 * a1*a2);
	//�����Ǻ�����Ƕ�
	return acos(dst) * 180 / 3.14;;
}


float  getkeyArea(const RotatedRect& box_in, const RotatedRect& box_out, const Mat& frame, Point& leftPoint, Point& rightPoint){
	vector<vector<Point>>contours;
	vector<Vec4i>hierarchy;
	Mat mask = Mat::zeros(frame.size(), CV_8UC1);
	Mat mask_in = mask.clone();
	Mat mask_out = mask.clone();

	drawcircle(mask_in, box_in);
	drawcircle(mask_out, box_out);

	Mat mask_dst = mask_out - mask_in;

	//ȥ������ɫ���� Ҳ��������rg�ռ�
	Mat frame_tmp = frame.clone();
	vector<Mat> vec_rgbsplit;
	split(frame_tmp, vec_rgbsplit);
	Mat rg = vec_rgbsplit[2] - vec_rgbsplit[1];
	rg = rg > 100;

	cvtColor(frame_tmp, frame_tmp, CV_BGR2GRAY);
	frame_tmp = frame_tmp > 50;
	frame_tmp &= mask_dst;
	frame_tmp &= ~rg;
	imshow("��ȡ�̶ȱ���", frame_tmp);
	waitKey();
	findContours(frame_tmp, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
	frame_tmp.setTo(0);
	vector<pair<int, Point>> keymap;
	for (int i = 0; i < contours.size(); i++){
		drawContours(frame_tmp, contours, i, Scalar(255));
		Point center = getPointCenter(contours[i]);
		int  area = contours[i].size();
		keymap.push_back(pair<int, Point>(area, center));
	}

	sort(keymap.begin(), keymap.end(), [](const pair<int, Point>& a, const pair<int, Point>& b){return a.first>b.first; });
	keymap.erase(keymap.end() - keymap.size()*0.5, keymap.end());
	sort(keymap.begin(), keymap.end(), [](const pair<int, Point>& a, const pair<int, Point>& b){return a.second.y > b.second.y; });
	leftPoint = keymap[0].second;
	rightPoint = keymap[1].second;
	circle(frame_tmp, leftPoint, 5, Scalar(255));
	circle(frame_tmp, rightPoint, 5, Scalar(255));
	float angle = calAngle(box_out.center, leftPoint, rightPoint);
	return angle;
}

void main(){
	vector<string> vec_jpg;
	paFindFiles("./data/�Ǳ�����ͼ��/2/", vec_jpg, "*.png;*.jpg");
	if (vec_jpg.size() < 1)
		return;
//��Ĥ����  ��һ�������ԭ����һ��ͼ����������������¼�������ֵ
	Mat mask;
	vector<pair<Mat, Mat>> vec_pair;
	for (int i = 0; i < vec_jpg.size() - 1; i++){
		Mat frame_cur = imread(vec_jpg[i]);
		Mat frame_after = imread(vec_jpg[i + 1]);
		resize(frame_cur, frame_cur, Size(frame_cur.cols*0.5, frame_cur.rows*0.5));
		resize(frame_after, frame_after, Size(frame_after.cols*0.5, frame_after.rows*0.5));

		//��ȡ��������
		Mat diff;
		absdiff(frame_cur, frame_after, diff);
		cvtColor(diff, diff, CV_BGR2GRAY);
		diff = diff > 10;
		diffMask(diff);

		Mat tmp = frame_cur.clone();

		//��ȡ����
		//Mat bg  = getBackGround(frame_cur, frame_after);
		Mat bg = getBg(diff, frame_cur);
		vec_pair.push_back(pair<Mat, Mat>(bg, diff));

		//copyTo �������ع�
		if (mask.empty()){
			diff.copyTo(mask);
		}
		else{
			bitwise_or(mask, diff, mask);  //bitwise_or�ǶԶ��������ݽ��С��򡱲���
		}
		imshow("��ȡ����", frame_cur);
		imshow("��ȡ��������", bg);
		waitKey(1);
	}
	Mat center_mask;
	int ratio = 0;
	Point centerBase = getCenter(mask, center_mask, ratio);//��Ϊ�������ĵ�,�����Ϳ���Ԥ֧���λ��

	//�Ա��������ں�
 	Mat base_background = vec_pair[0].first.clone();
	Mat base_mask = vec_pair[0].second.clone();
	for (int i = 1; i < vec_pair.size() - 1; i++){
		Mat cur_background = vec_pair[i].first;
		Mat cur_mask = vec_pair[i].second;
		for (int r = 0; r < base_background.rows; r++){
			for (int c = 0; c < base_background.cols; c++){
				//���������ú� ������ȡָ��

				if (base_mask.at<unsigned char>(r, c) == 255 && cur_mask.at<unsigned char>(r, c) == 0){
					base_background.at<Vec3b>(r, c)[0] = base_background.at<Vec3b>(r, c)[0] * 0.5 + cur_background.at<Vec3b>(r, c)[0] * 0.5;
					base_background.at<Vec3b>(r, c)[1] = base_background.at<Vec3b>(r, c)[1] * 0.5 + cur_background.at<Vec3b>(r, c)[1] * 0.5;
					base_background.at<Vec3b>(r, c)[2] = base_background.at<Vec3b>(r, c)[2] * 0.5 + cur_background.at<Vec3b>(r, c)[2] * 0.5;
				}

				if (center_mask.at<unsigned char>(r, c) == 255){
					base_background.at<Vec3b>(r, c)[0] = 0;
					base_background.at<Vec3b>(r, c)[1] = 0;
					base_background.at<Vec3b>(r, c)[2] = 0;
				}
			}
		}

	}
	imshow("�����ںϽ��", base_background);
	waitKey(1);




	//��ȡǰ��, ��ȡָ��Զ���㲢���浽vector��
	FILE*fp = fopen("savePoint.txt", "wb");
	vector<Point>vec_centerPoint;
	vector<Point>vec_topPoint;
	for (int i = 0; i < vec_jpg.size(); i++){
		Mat frame_cur = imread(vec_jpg[i]);
		resize(frame_cur, frame_cur, Size(frame_cur.cols*0.5, frame_cur.rows*0.5));

		Mat diff;
		absdiff(frame_cur, base_background, diff);
		imshow("��ȡǰ��", diff);
		cvtColor(diff, diff, CV_BGR2GRAY);
		diff = diff > 10;
		findMaxArea(diff);
		//diff &= vec_pair[i].second;
		Point top, center;
		findMinMax(diff, centerBase, top, center);

		circle(frame_cur, center, 10, Scalar(0, 255, 255), CV_FILLED);
		circle(frame_cur, top, 10, Scalar(0, 0, 255), CV_FILLED);
		imshow("��ȡԶ���˵�", frame_cur);
		waitKey(1);
		vec_centerPoint.push_back(center);
		vec_topPoint.push_back(top);

		fprintf(fp, "%s %d %d %d %d\n", vec_jpg[i].c_str(), center.x, center.y, top.x, top.y);
	}
	fclose(fp);
	waitKey();
	destroyAllWindows();
	//�ֶ�ѡ��,���������Բ
	RotatedRect box_in, box_out;
	Mat frame_cur = imread(vec_jpg[0]);
	resize(frame_cur, frame_cur, Size(frame_cur.cols*0.5, frame_cur.rows*0.5));
	Mat frame_src = frame_cur.clone();
	namedWindow("ѡ����ȦĿ���");
	setMouseCallback("ѡ����ȦĿ���", drawPoint);
	vector<Point> in_keyPoint;
	vector<Point> out_keyPoint;

	while (!drawAtm){
		imshow("ѡ����ȦĿ���", frame_cur);
		waitKey();
	}
	destroyWindow("ѡ����ȦĿ���");
	in_keyPoint = roiArea;
	drawAtm = false;
	roiArea.clear();


	box_in = fitEllipse(in_keyPoint);
	ellipse(frame_cur, box_in, Scalar(0, 0, 255), 1, CV_AA);


	namedWindow("ѡ����ȦĿ���");
	setMouseCallback("ѡ����ȦĿ���", drawPoint);
	while (!drawAtm){
		imshow("ѡ����ȦĿ���", frame_cur);
		waitKey();
	}
	destroyWindow("ѡ����ȦĿ���");
	out_keyPoint = roiArea;
	roiArea.clear();
	box_out = fitEllipse(out_keyPoint);
	ellipse(frame_cur, box_out, Scalar(0, 0, 255), 1, CV_AA);
	circle(frame_cur, box_out.center, 10, Scalar(255, 255, 255), CV_FILLED);
	imshow("��Ȧ+��Ȧ��Բ��Ͻ��", frame_cur);
	waitKey();


	Point legtPoint, rightPoint;
	float base_angle = getkeyArea(box_in, box_out, frame_src, legtPoint, rightPoint); //���̱�ΪԲ�Σ�����ֻ�д��4/5�����̣������ȼ���������ռ����Բ�ĽǶ�
	float total_angle = 360 - base_angle;
	float base_val = 7; //�������� 
	for (int i = 0; i < vec_jpg.size(); i++){
		Mat frame_cur = imread(vec_jpg[i]);
		resize(frame_cur, frame_cur, Size(frame_cur.cols*0.5, frame_cur.rows*0.5));
		circle(frame_cur, vec_topPoint[i], 7, Scalar(0, 255, 255), CV_FILLED);
		Point pcur = vec_topPoint[i];
		Point pbase;
		if (pcur.x<box_out.center.x)
		{
			pbase = legtPoint;
		}
		else{
			pbase = rightPoint;
		}
		float angle = calAngle(box_out.center, pbase, pcur);
		if (pcur.x > box_out.center.x)
			angle = total_angle - angle;

		float val = base_val*angle / total_angle;
		cout << "��ǰԤ��ֵ=0" << val << endl;
		putText(frame_cur, format("val=%.3f", val), Point(30, 30), 1, 3, Scalar(0, 0, 255), 2);
		imshow("ֵԤ��", frame_cur);
		waitKey();
	}
}