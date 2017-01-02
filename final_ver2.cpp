#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <Eigen/Dense>
#include <Eigen/SVD>
#include <math.h>
#include <vector>
#include <string>
#include <sstream>
#include <time.h>

using namespace std;
using Eigen::MatrixXf;
using Eigen::JacobiSVD;
/*template<typename T>
std::string to_string(T value){

std::ostream os;
os << value;
return os.str();
}*/

//to read parameter from CMD
struct parameter {

	string filename;
	string result;
	int header_size;
	int w;
	int h;
	int point;
	int data_size;
	int sample;

};

int sample = 4;
int image_ID = 49;
//int block_num = 2 * 2;//2^k * 2^k  (from 1x1 to 128x128)
int dct_point_defalt = 16;
int min_dct_point = 4;
int data_width = 512;
int data_height = 512;
int data_size = data_width * data_height;
int header_size = 1078;

int **block_dct_point = new int*[(int)log2(data_width / min_dct_point) + 1];

typedef struct node* nodeptr;

typedef struct node {
	int symbol; //-256/sampele-1 ~ 256/sample+1 or 0 ~ 256/sample
	int count;  // probability
	nodeptr left;
	nodeptr right;
};


// a priority queue implemented as a binary heap !
typedef struct PQ {
	int	heap_size;
	//nodeptr* A = new nodeptr[256 / sample];
	nodeptr* A;
};

int parent(int i); // find heap node's parent 
int leftchild(int i); // find heap node's left kid 
int rightchild(int i);// find heap node's right kid  
void heap_make(PQ *p, int i);
void insert_pq(PQ *p, nodeptr r);//insert node to pq
nodeptr extract_min_pq(PQ *p);//extract min node in pq
nodeptr build_huffman(unsigned int freqs[], int size);
void postorder(nodeptr currentcode, FILE *f);//postorder traversal 
void traverse(nodeptr r, 	// root of this subtree 
	int level, 	// current level  
	char code_now[], // code string up 
	char *codes[],// array of codes 
	int length[]
); //array 


int main(int argc, char *argv[])
{
//for (int ID = 1; ID<=image_ID;ID++){//calculate all image in bmp_file
	parameter para;

	/////////////////////////////////////////////////parameter//////////////////////////////////////////////
	
	para.filename = std::string("bmp_file/")+ std::to_string(image_ID)+std::string(".bmp");
	//para.filename = std::string("bmp_file/") + std::to_string(ID) + std::string(".bmp");
	para.header_size = header_size;
	para.w = data_width;
	para.h = data_height;
	para.point = dct_point_defalt;
	para.data_size = para.w * para.h;
	para.result = "baboon_4.csv";
	para.sample = sample;

	//uncommand this line to execute by CMD
	//para = setInfoFromCMD(argc, argv, para);
	//cout << "file : " << para.filename.c_str() << "size : " << para.w << "x" << para.h << endl;
	//cout << para.point << "-point DCT" << endl;

	//////////////////////////////////////////////read file//////////////////////////////////////////////////

	fstream read_f;
	const char* filename_origin = para.filename.c_str();
	read_f.open(filename_origin, ios::in | ios::binary);

	unsigned char int8buffer;
	unsigned char *header = new unsigned char[para.header_size];
	unsigned char *image = new unsigned char[para.data_size];

	if (!read_f) {
		cout << "NO file";
		return 1;
	}

	for (int i = 0; i < para.header_size; i++) {
		read_f.read((char*)&int8buffer, sizeof(unsigned char));
		header[i] = int8buffer;
	}

	for (int i = 0; i < para.h; i++) {
		for (int j = 0; j < para.w; j++) {
			read_f.read((char*)&int8buffer, sizeof(unsigned char));
			image[(para.h - 1 - i)*para.w + j] = int8buffer;
			//image[(para.h - 1 - i)*para.w + j] = int8buffer/para.sample;
			
		}
	}

	/////////////////////////////////////////////DCT decorrelation/////////////////////////////////////////////
	//cout << "***DCT***" << endl;
	MatrixXf  A(para.point, para.point);

	for (int i = 0; i < para.point; i++)
		for (int j = 0; j < para.point; j++) {
			if (i == 0)
				A(i, j) = sqrt((float)2 / para.point)*sqrt((float)1 / 2)*cos(2 * EIGEN_PI*(2 * j + 1)*i / (4 * para.point));
			else
				A(i, j) = sqrt((float)2 / para.point) * 1 * cos(2 * EIGEN_PI*(2 * j + 1)*i / (4 * para.point));
		}

	MatrixXf  A_4x4(4, 4);

	for (int i = 0; i < 4; i++)
		for (int j = 0; j < 4; j++) {
			if (i == 0)
				A_4x4(i, j) = sqrt((float)2 / 4)*sqrt((float)1 / 2)*cos(2 * EIGEN_PI*(2 * j + 1)*i / (4 * 4));
			else
				A_4x4(i, j) = sqrt((float)2 / 4) * 1 * cos(2 * EIGEN_PI*(2 * j + 1)*i / (4 * 4));
		}

	MatrixXf  A_8x8(8, 8);

	for (int i = 0; i < 8; i++)
		for (int j = 0; j < 8; j++) {
			if (i == 0)
				A_8x8(i, j) = sqrt((float)2 / 8)*sqrt((float)1 / 2)*cos(2 * EIGEN_PI*(2 * j + 1)*i / (4 * 8));
			else
				A_8x8(i, j) = sqrt((float)2 / 8) * 1 * cos(2 * EIGEN_PI*(2 * j + 1)*i / (4 * 8));
		}

	MatrixXf  A_16x16(16, 16);

	for (int i = 0; i < 16; i++)
		for (int j = 0; j < 16; j++) {
			if (i == 0)
				A_16x16(i, j) = sqrt((float)2 / 16)*sqrt((float)1 / 2)*cos(2 * EIGEN_PI*(2 * j + 1)*i / (4 * 16));
			else
				A_16x16(i, j) = sqrt((float)2 / 16) * 1 * cos(2 * EIGEN_PI*(2 * j + 1)*i / (4 * 16));
		}


	////////////////////////////////////////////////////DCT image///////////////////////////////////////////////
    
	
	//parameter for calculate bitrate each loop
	float entropy_Q = 0.0;
	float bitrate_Q = 0.0;	
	
	//MatrixXf  block(para.point, para.point);
	//MatrixXf  dct_block(para.point, para.point);
	int *image_dct = new int[para.data_size];
	int start_i ,start_j;

	double current_max_eff = 0;
	double avg_eff=0;
	double var_temp = 0;
	double avg_var = 0;
	double current_min_var= 0;
	int point_index_max = 0;
	int block_num = 0;
	
	//double *bitrate_record = new double[(int)log2(para.w / 4) + 1];
	double best_bitrate = 100;
	double best_block_num = 1;

	for (int block_num_count = 0; pow(2, block_num_count) <= para.w / min_dct_point; block_num_count++) {
		
		
		
		entropy_Q = 0.0;
		bitrate_Q = 0.0;

		block_num = pow(2, block_num_count)*pow(2, block_num_count);
		cout << "The program is computing the bitrate of block num = " << sqrt(block_num)<<" x "<< sqrt(block_num) << " now, please wait!!!" << endl;
		block_dct_point[block_num_count] = new int[block_num];
		for (int i = 0; i < block_num; i++)
			block_dct_point[block_num_count][i] = 4;//default
		if (para.w / sqrt(block_num) >= 16)
			point_index_max = 5;
		else if (para.w / sqrt(block_num) == 8)
			point_index_max = 4;
		else
			point_index_max = 3;

		//calculate the best efficency on which size (4x4,8x8,or 16x16)of DCT per block 
		for (int block_count = 0; block_count < block_num; block_count++) {

			//cout << "block ID:" << block_count << endl;
			start_i = (block_count / (int)sqrt(block_num))*(para.h / (int)sqrt(block_num));
			start_j = (block_count % (int)sqrt(block_num))*(para.w / (int)sqrt(block_num));
			current_max_eff = 0;
			current_min_var = 100000;

			for (int point_index = 2; point_index < point_index_max; point_index++) {

				int dct_point = pow(2, point_index);
				double *avg = new double[dct_point];
				double **X = new double*[dct_point];
				for (int i = 0; i < dct_point; i++)
					X[i] = new double[(para.w / (int)sqrt(block_num)) / dct_point * (para.h / (int)sqrt(block_num))];
				double temp = 0;
				MatrixXf cov(dct_point, dct_point);
				MatrixXf dct(dct_point, dct_point);
				double eff = 0.0;
				double XX = 0.0;
				double YY = 0.0;
				unsigned char *image_block = new unsigned char[para.data_size / block_num];
				int image_block_h = sqrt(para.data_size / block_num);
				int image_block_w = sqrt(para.data_size / block_num);
				for (int i = 0; i < image_block_h; i++)
					for (int j = 0; j < image_block_w; j++)
						image_block[i*image_block_w + j] = image[(i + start_i)*para.w + (j + start_j)];



				for (int i = 0; i < dct_point; i++)
					avg[i] = 0;

				for (int i = 0; i < dct_point; i++)
					for (int j = 0; j <(para.w / (int)sqrt(block_num)) / dct_point * (para.h / (int)sqrt(block_num)); j++) {
						X[i][j] = image_block[j * dct_point + i];
						avg[i] += X[i][j];
					}
				for (int i = 0; i < dct_point; i++)
					avg[i] /= ((para.data_size / block_num) / dct_point);

				//count covariance


				for (int i = 0; i < dct_point; i++)
					for (int j = 0; j < dct_point; j++) {
						for (int k = 0; k < (para.w / (int)sqrt(block_num)) / dct_point * (para.h / (int)sqrt(block_num)); k++)
							temp += (X[i][k] - avg[i])*(X[j][k] - avg[j]);
						cov(i, j) = temp / ((para.data_size / block_num) / dct_point);
						temp = 0;
					}
				//cout << "dct_point:" << dct_point << endl;
				if (dct_point == 4)
					dct = A_4x4*cov*A_4x4.transpose();
				else if (dct_point == 8)
					dct = A_8x8*cov*A_8x8.transpose();
				else
					dct = A_16x16*cov*A_16x16.transpose();

				//count for efficiency

				for (int i = 0; i < dct_point; i++)
					for (int j = 0; j < dct_point; j++)
						if (i != j)
							XX += abs(cov(i, j));
				for (int i = 0; i < dct_point; i++)
					for (int j = 0; j < dct_point; j++)
						if (i != j)
							YY += abs(dct(i, j));
				eff = 1 - YY / XX;
				//cout << "efficiency:" << eff << endl;
				if (eff > current_max_eff) {
					current_max_eff = eff;
					block_dct_point[block_num_count][block_count] = dct_point;
					//var_temp = abs(dct(0, 0));

				}
				//cout << "var:" << abs(dct(0, 0))/ dct_point  << endl;
				if (abs(cov(0, 0)) < current_min_var) {
					//current_min_var = abs(cov(0, 0)) ;
					//block_dct_point[block_count] = dct_point;
					//current_max_eff = eff;
				}

				delete[]avg;
				for (int i = 0; i < dct_point; i++)
					delete[]X[i];
				delete[]X;
				delete[]image_block;

			}
			//cout << "best efficiency at :" << block_dct_point[block_count] <<"x"<< block_dct_point[block_count] << "dct maxtrix" << endl;
			//cout << "min var at :" << block_dct_point[block_count] <<"x"<< block_dct_point[block_count] << "dct maxtrix" << endl;
			avg_eff += current_max_eff / block_num;
			//avg_var += current_min_var / block_num;
		}


	




		//avg_eff /= block_num;
		//avg_var /= block_num;
		//cout << avg_eff << endl;
		//cout << avg_var << endl;
		
		for (int block_count = 0; block_count < block_num; block_count++) {
			MatrixXf  block(block_dct_point[block_num_count][block_count], block_dct_point[block_num_count][block_count]);
			MatrixXf  dct_block(block_dct_point[block_num_count][block_count], block_dct_point[block_num_count][block_count]);
			start_i = (block_count / (int)sqrt(block_num))*(para.h / (int)sqrt(block_num));
			start_j = (block_count % (int)sqrt(block_num))*(para.w / (int)sqrt(block_num));
			//cout << block_count << endl;
			//cout << start_j << endl;
			//cout << start_i << endl;
			//cout << start_j + para.w / sqrt(block_num) - para.point + 1 << endl;
			//cout << start_i + para.h / sqrt(block_num) - para.point + 1 << endl;

			for (int i = start_i; i < start_i+para.h/sqrt(block_num) - block_dct_point[block_num_count][block_count] + 1; i += block_dct_point[block_num_count][block_count])
				for (int j = start_j; j < start_j+para.w / sqrt(block_num) - block_dct_point[block_num_count][block_count] + 1; j += block_dct_point[block_num_count][block_count]) {
					for (int y = 0; y < block_dct_point[block_num_count][block_count]; y++)
						for (int x = 0; x < block_dct_point[block_num_count][block_count]; x++)
							block(x, y) = image[(j + y) * para.w + (i + x)]/para.sample - 128/para.sample;//Because the DCT is designed to work on pixel values ranging from -128 to 127
						if(block_dct_point[block_num_count][block_count]==4)
							dct_block = A_4x4*block*A_4x4.transpose();
						else if(block_dct_point[block_num_count][block_count]==8)
							dct_block = A_8x8*block*A_8x8.transpose();
						else
							dct_block = A_16x16*block*A_16x16.transpose();
						for (int y = 0; y < block_dct_point[block_num_count][block_count]; y++)
							for (int x = 0; x < block_dct_point[block_num_count][block_count]; x++)
								image_dct[(j + y) * para.w + (i + x)] = dct_block(x, y);
			}
		}

		
		///////////////////////////////////////////////calculate bitrate///////////////////////////////////////////////////////
		int max_d50 = 0;
		for (int i = 0; i < para.data_size; i++)
			if (abs(image_dct[i]) > max_d50)
				max_d50 = abs(image_dct[i]);

		int *imageQ = new int[para.data_size];
		//unsigned char temp = 0;
		int counter_range = 2 * max_d50 + 1;//-max_d50~max_d50
		unsigned int *counter = new unsigned int[counter_range];
		nodeptr r; // root of Huffman tree  

		for (int i = 0; i < para.data_size; i++)
			imageQ[i] = 0;

		for (int i = 0; i < counter_range; i++)
			counter[i] = 0;


		for (int i = 0; i < para.data_size; i++)
			//imageQ[i] = image[i] ;
			imageQ[i] = image_dct[i] ;
			//imageQ[i] = image_dct_Q50[i];

		for (int i = 0; i < para.data_size; i++) {

			for (int j = 0; j < counter_range; j++)
				if (imageQ[i] == -max_d50 + j)
					counter[j]++;

		}



		for (int i = 0; i < counter_range; i++) {

			if (!counter[i])
				entropy_Q += 0;
			else
				entropy_Q += (-1)*((float)counter[i] / para.data_size)*log2f((float)counter[i] / para.data_size);
		}

		char **bitcodes = new char*[counter_range]; // array of codes, 1 bit per char 
		char *bitcode = new char[1000];   // a place to hold 1 code each time
		int * length = new int[counter_range];
		r = build_huffman(counter, counter_range);

		for (int i = 0; i < counter_range; i++)
			bitcodes[i] = 0;

		traverse(r, 0, bitcode, bitcodes, length);

		for (int i = 0; i < counter_range; i++) {


			if (counter[i] == para.data_size)
				bitrate_Q += 0;
			else
				bitrate_Q += ((float)counter[i] / para.data_size)*length[i];
		}


			
		delete[]counter;
		delete[]r;
		delete[]bitcode;
		delete[]bitcodes;
		delete[]imageQ;

		//bitrate_record[block_num_count] = bitrate_Q;
		cout << "bitrate_Q: " << bitrate_Q << " for block num=" << sqrt(block_num) << " x " << sqrt(block_num) << endl;
		if (best_bitrate > bitrate_Q) {
			best_bitrate = bitrate_Q;
			best_block_num = block_num;

		}
				
	}		


	cout << "Finally, we find the best bitrate_Q: " << best_bitrate<< " at block num=" << sqrt(best_block_num) << " x " << sqrt(best_block_num) << endl;
	cout << "Its first block has the best efficiency at DCT point = "<< block_dct_point[(int)log2(sqrt(best_block_num))][0] << endl;



	////////////////////////////////////////////////////Best DCT image///////////////////////////////////////////////

	int *image_dct_best = new int[para.data_size];

	for (int block_count = 0; block_count < best_block_num; block_count++) {
		MatrixXf  block(block_dct_point[(int)log2(sqrt(best_block_num))][block_count], block_dct_point[(int)log2(sqrt(best_block_num))][block_count]);
		MatrixXf  dct_block(block_dct_point[(int)log2(sqrt(best_block_num))][block_count], block_dct_point[(int)log2(sqrt(best_block_num))][block_count]);
		start_i = (block_count / (int)sqrt(best_block_num))*(para.h / (int)sqrt(best_block_num));
		start_j = (block_count % (int)sqrt(best_block_num))*(para.w / (int)sqrt(best_block_num));
		
		//cout << block_count << endl;
		//cout << start_j << endl;
		//cout << start_i << endl;
		//cout << start_j + para.w / sqrt(block_num) - para.point + 1 << endl;
		//cout << start_i + para.h / sqrt(block_num) - para.point + 1 << endl;

		for (int i = start_i; i < start_i + para.h / sqrt(best_block_num) - block_dct_point[(int)log2(sqrt(best_block_num))][block_count] + 1; i += block_dct_point[(int)log2(sqrt(best_block_num))][block_count])
			for (int j = start_j; j < start_j + para.w / sqrt(best_block_num) - block_dct_point[(int)log2(sqrt(best_block_num))][block_count] + 1; j += block_dct_point[(int)log2(sqrt(best_block_num))][block_count]){
				for (int y = 0; y < block_dct_point[(int)log2(sqrt(best_block_num))][block_count]; y++)
					for (int x = 0; x < block_dct_point[(int)log2(sqrt(best_block_num))][block_count]; x++)
						block(x, y) = image[(j + y) * para.w + (i + x)] / para.sample - 128 / para.sample;//Because the DCT is designed to work on pixel values ranging from -128 to 127
				if (block_dct_point[(int)log2(sqrt(best_block_num))][block_count] == 4)
					dct_block = A_4x4*block*A_4x4.transpose();
				else if (block_dct_point[(int)log2(sqrt(best_block_num))][block_count] == 8)
					dct_block = A_8x8*block*A_8x8.transpose();
				else
					dct_block = A_16x16*block*A_16x16.transpose();
				for (int y = 0; y < block_dct_point[(int)log2(sqrt(best_block_num))][block_count]; y++)
					for (int x = 0; x < block_dct_point[(int)log2(sqrt(best_block_num))][block_count]; x++)
						image_dct_best[(j + y) * para.w + (i + x)] = dct_block(x, y);
			}
	}

	///////////////////////////////////////////////output DCT image///////////////////////////////////////////////
	string dct_file;
	//if (para.filename == "bmp_file/baboon.bmp")
	dct_file = std::string("result/") + std::to_string(image_ID) + std::string("_dct.bmp");
	//dct_file = std::string("result/") + std::to_string(ID) + std::string("_dct.bmp");
	//else
	//	dct_file = "lout.bmp";
	ofstream fbmp(dct_file, ios::out | ios::binary);

	for (int i = 0; i < para.header_size; i++) {
		int8buffer = header[i];
		fbmp.write((char*)&int8buffer, sizeof(char));

	}

	for (int i = 0; i < para.h; i++) {
		for (int j = 0; j < para.w; j++) {
			int8buffer = image_dct_best[(para.h - 1 - i)*para.w + j];
			//int8buffer = image_dct[i*para.w + j];
			fbmp.write((char*)&int8buffer, sizeof(char));
		}
	}
	fbmp.close();

	///////////////////////////////////////////////quantization///////////////////////////////////////////////////
	//unsigned char* Q50_table_8x8 = new unsigned char[64];
	unsigned char* Q50_table_8x8 = new unsigned char[para.point*para.point];
	int *image_dct_Q50 = new int[para.data_size];
	FILE *Q_in = NULL;
	Q_in = fopen("Q_table_8x8.txt", "r");

	for (int i = 0; i < para.point*para.point; i++) {
		if (para.point == 8)
			fscanf(Q_in, "%d", &Q50_table_8x8[i]);
		else
			Q50_table_8x8[i] = 1;
		//cout << (unsigned)Q_table_8x8[i] << endl;
		//system("pause");
	}
	////C=round(D_block/Q)
	for (int i = 0; i < para.h - para.point + 1; i += para.point)
		for (int j = 0; j < para.w - para.point + 1; j += para.point) {
			for (int y = 0; y < para.point; y++)
				for (int x = 0; x < para.point; x++)
					image_dct_Q50[(j + y) * para.w + (i + x)] = round((float)image_dct_best[(j + y) * para.w + (i + x)] / Q50_table_8x8[y*para.point + x]);
		}
	fclose(Q_in);
	///////////////////////////////////////////////output DCT Q50 image////////////////////////////////////////////
	string dct_Q50_file;
	//if (para.filename == "bmp_file/baboon.bmp")
	dct_Q50_file = std::string("result/") + std::to_string(image_ID) + std::string("_dct_quantization.bmp");
	//dct_Q50_file = std::string("result/") + std::to_string(ID) + std::string("_dct_quantization.bmp");
	//else
	//	dct_file = "lout.bmp";

	ofstream fbmp2(dct_Q50_file, ios::out | ios::binary);


	for (int i = 0; i < para.header_size; i++) {
		int8buffer = header[i];
		fbmp2.write((char*)&int8buffer, sizeof(unsigned char));

	}

	for (int i = 0; i < para.h; i++) {
		for (int j = 0; j < para.w; j++) {
			int8buffer = image_dct_Q50[(para.h - 1 - i)*para.w + j];
			//int8buffer = image_dct[i*para.w + j];
			fbmp2.write((char*)&int8buffer, sizeof(unsigned char));
		}
	}

	fbmp2.close();



	///////////////////////////////////////////////encoding///////////////////////////////////////////////////////

	clock_t start_en, finish_en, start_de, finish_de;
	start_en = clock();
	float duration_en, duration_de;
	entropy_Q = 0.0;
	bitrate_Q = 0.0;
	float entropy_dQ = 0.0;
    float bitrate_dQ = 0.0;

	int max_d50 = 0;
	for (int i = 0; i < para.data_size; i++)
		if (abs(image_dct_Q50[i]) > max_d50)
			max_d50 = abs(image_dct_Q50[i]);


	int *imageQ = new int[para.data_size];
	int *imageQQ = new int[para.data_size];
	//unsigned char temp = 0;
	int counter_range = 2 * max_d50 + 1;//-max_d50~max_d50
	int counter_range2 = 4 * max_d50 + 1;	 //-2*max_d50~2*max_d50
	unsigned int *counter = new unsigned int[counter_range];
	unsigned int *counter2 = new unsigned int[counter_range2];
	nodeptr r; // root of Huffman tree  
	nodeptr r2;

	for (int i = 0; i < para.data_size; i++)
		imageQ[i] = 0;

	for (int i = 0; i < para.data_size; i++)
		imageQQ[i] = 0;


	for (int i = 0; i < counter_range; i++)
		counter[i] = 0;




	for (int i = 0; i < counter_range2; i++)
		counter2[i] = 0;


	for (int i = 0; i < para.data_size; i++)
		//imageQ[i] = image[i] ;
		//imageQ[i] = image_dct[i] ;
		imageQ[i] = image_dct_Q50[i];

	for (int i = 0; i < para.data_size; i++) {

		if (i == 0 && para.sample == 4)
			imageQQ[i] = imageQ[i] - 32;
		else if (i == 0 && para.sample == 2)
			imageQQ[i] = imageQ[i] - 64;
		else if (i == 0 && para.sample == 1)
			imageQQ[i] = imageQ[i] - 128;
		else
			imageQQ[i] = imageQ[i] - imageQ[i - 1];

		for (int j = 0; j < counter_range; j++)
			if (imageQ[i] == -max_d50 + j)
				counter[j]++;

	}

	for (int i = 0; i < para.data_size; i++)
		for (int j = 0; j < counter_range2; j++)
			if (imageQQ[i] == -2 * max_d50 + j)
				counter2[j]++;



	for (int i = 0; i < counter_range; i++) {

		if (!counter[i])
			entropy_Q += 0;
		else
			entropy_Q += (-1)*((float)counter[i] / para.data_size)*log2f((float)counter[i] / para.data_size);
	}

	char **bitcodes = new char*[counter_range]; // array of codes, 1 bit per char 
	char *bitcode = new char[1000];   // a place to hold 1 code each time
	int * length = new int[counter_range];
	r = build_huffman(counter, counter_range);

	for (int i = 0; i < counter_range; i++)
		bitcodes[i] = 0;

	traverse(r, 0, bitcode, bitcodes, length);

	for (int i = 0; i < counter_range; i++) {


		if (counter[i] == para.data_size)
			bitrate_Q += 0;
		else
			bitrate_Q += ((float)counter[i] / para.data_size)*length[i];
	}



	for (int i = 0; i < counter_range2; i++) {

		if (!counter2[i])
			entropy_dQ += 0;
		else
			entropy_dQ += (-1)*((float)counter2[i] / para.data_size)*log2f((float)counter2[i] / para.data_size);

	}

	char **bitcodes2 = new char*[counter_range2]; // array of codes, 1 bit per char 
	char *bitcode2 = new char[1000];   // a place to hold 1 code each time
	int *length2 = new int[counter_range2];
	r2 = build_huffman(counter2, counter_range2);
	int maxlength = 0;//offset_size
	int codenum = 0;
	//int offset_num = 0;
	int *offset_codenum;
	int *offset_range;
	int symbol_size = 0;
	int *symbol_order;
	int *codenum_now = new int[para.data_size];
	int *encode = new int[para.data_size];
	unsigned char bitstream = 0;
	int out_count = 0;
	
	
	for (int i = 0; i < counter_range2; i++)
		bitcodes2[i] = 0;

	traverse(r2, 0, bitcode2, bitcodes2, length2);


	for (int i = 0; i < counter_range2; i++) {

		if (counter2[i] == para.data_size)
			bitrate_dQ += 0;
		else
			bitrate_dQ += ((float)counter2[i] / para.data_size)*length2[i];
	}


	//cout << bitrate_Q << endl;
	//cout << bitrate_dQ << endl;
	//system("pause");

	if (bitrate_Q < bitrate_dQ) {
		for (int i = 0; i < counter_range; i++)                    //by the result of  traverse for bitcodes     
			if (bitcodes[i] != 0)
				symbol_size++;
	}

	else {
		for (int i = 0; i < counter_range2; i++)                    //by the result of  traverse for bitcodes     
			if (bitcodes2[i] != 0)
				symbol_size++;
	}


	symbol_order = new int[symbol_size];
	for (int i = 0; i < symbol_size; i++)
		symbol_order[i] = 0;

	if (bitrate_Q < bitrate_dQ) {
		for (int i = 0; i < counter_range; i++)
			if (length[i] > maxlength)
				maxlength = length[i];
	}

	else {
		for (int i = 0; i < counter_range2; i++)
			if (length2[i] > maxlength)
				maxlength = length2[i];
	}


	//cout << maxlength<< endl;
	offset_codenum = new int[maxlength];
	offset_range = new int[maxlength];


	for (int i = 0; i < maxlength; i++) {

		offset_codenum[i] = 0;
		offset_range[i] = 0;
	}

	int* lengthcount2 = new int[maxlength];

	for (int i = 0; i < maxlength; i++)
		lengthcount2[i] = 0;

	if (bitrate_Q < bitrate_dQ) {
		for (int i = 0; i < maxlength; i++)
			for (int j = 0; j < counter_range; j++)
				if (length[j] == i + 1) {
					lengthcount2[i]++;
					symbol_order[codenum] = -max_d50 + j;
					codenum++;

				}

	}
	else {
		for (int i = 0; i < maxlength; i++)
			for (int j = 0; j < counter_range2; j++)
				if (length2[j] == i + 1) {
					lengthcount2[i]++;
					symbol_order[codenum] = -2 * max_d50 + j;
					codenum++;

				}

	}


	for (int i = 0; i < maxlength; i++)
		if (i == 0) {
			offset_codenum[0] = 0;
			offset_range[0] = 0;
		}

		else {
			for (int j = 0; j < i; j++)
				offset_codenum[i] += lengthcount2[j];
			offset_range[i] = (offset_range[i - 1] + offset_codenum[i] - offset_codenum[i - 1]) * 2;
		}


	string bin_file;
	bin_file =  std::string("result/") + std::to_string(image_ID) + std::string("_dct_encode.bin");
	//bin_file = std::string("result/") + std::to_string(ID) + std::string("_dct_encode.bin");
	ofstream fenco(bin_file, ios::out | ios::binary);

	if (bitrate_Q < bitrate_dQ) {
		for (int i = 0; i < para.data_size; i++)
			for (int j = 0; j < symbol_size; j++)
				if (imageQ[i] == symbol_order[j])
					codenum_now[i] = j;
	
	}
	
	else {
		for (int i = 0; i < para.data_size; i++)
			for (int j = 0; j < symbol_size; j++)
				if (imageQQ[i] == symbol_order[j])
					codenum_now[i] = j;
	}



	for (int i = 0; i < para.data_size; i++)
		for (int j = 0; j < maxlength; j++)
			if (codenum_now[i] > offset_codenum[j] || codenum_now[i] == offset_codenum[j])
				encode[i] = offset_range[j] + (codenum_now[i] - offset_codenum[j]);

		
	if (bitrate_Q < bitrate_dQ) {
		for (int i = 0; i < para.data_size; i++) {

			int *encode_bit = new int[length[imageQ[i] + max_d50]];

			for (int j = 0; j < length[imageQ[i] + max_d50]; j++) {
				encode_bit[length[imageQ[i] + max_d50] - j - 1] = encode[i] % 2;
				encode[i] /= 2;
			}
			for (int j = 0; j < length[imageQ[i] + max_d50]; j++) {
				bitstream *= 2;
				out_count++;
				bitstream += encode_bit[j];
				if (out_count == 8) {
					fenco.write((char*)&bitstream, sizeof(bitstream));
					bitstream = 0;
					out_count = 0;
				}

			}
			delete[]encode_bit;
		}

	}
	else {
		for (int i = 0; i < para.data_size; i++) {

			int *encode_bit = new int[length2[imageQQ[i] + 2 * max_d50]];

			for (int j = 0; j < length2[imageQQ[i] + 2 * max_d50]; j++) {
				encode_bit[length2[imageQQ[i] + 2 * max_d50] - j - 1] = encode[i] % 2;
				encode[i] /= 2;
			}
			for (int j = 0; j < length2[imageQQ[i] + 2 * max_d50]; j++) {
				bitstream *= 2;
				out_count++;
				bitstream += encode_bit[j];
				if (out_count == 8) {
					fenco.write((char*)&bitstream, sizeof(bitstream));
					bitstream = 0;
					out_count = 0;
				}

			}
			delete[]encode_bit;
		}

	}



	fenco.close();
	finish_en = clock();



	///////////////////////////////////////////////decoding///////////////////////////////////////////////////////
	start_de = clock();

	out_count = 0;
	int bit_count = 0;
	ifstream fdeco(bin_file, ios::in | ios::binary);
	long long temp_decode = 0;
	char decode_bit;
	unsigned char tmp;
	int *decode = new int[para.data_size];
	int *decodeQQ = new int[para.data_size];
	int *decodeQ = new int[para.data_size];
	int data_count = 0;

	while (data_count < para.data_size) {
	
		temp_decode *= 2;
		if (out_count == 0)
			fdeco.read((char *)&tmp, sizeof(tmp));
		decode_bit = (char)(tmp / pow(2, 7 - out_count));

		temp_decode += decode_bit;

		tmp -= (char)(tmp / pow(2, 7 - out_count))*pow(2, 7 - out_count);

		out_count++;
		bit_count++;

		if (out_count == 8)
			out_count = 0;

		for (int j = 0; j < bit_count; j++) {


			if (j<maxlength - 1 && temp_decode >(offset_range[j] - 1) && temp_decode < (offset_range[j] + offset_codenum[j + 1] - offset_codenum[j])) {
				decode[data_count] = offset_codenum[j] + (temp_decode - offset_range[j]);
				bit_count = 0;
				data_count++;
				temp_decode = 0;
				continue;
			}
			if (j == maxlength - 1 && temp_decode >(offset_range[j] - 1)) {
				decode[data_count] = offset_codenum[j] + (temp_decode - offset_range[j]);
				bit_count = 0;
				data_count++;
				temp_decode = 0;
				continue;
			}

		}


	}

	if (bitrate_Q < bitrate_dQ) {
		for (int i = 0; i < para.data_size; i++)
			decodeQ[i] = symbol_order[decode[i]];
	}
	else {
		for (int i = 0; i < para.data_size; i++)
			decodeQQ[i] = symbol_order[decode[i]];

		if (para.sample == 4)
			decodeQ[0] = decodeQQ[0] + 32;
		else if (para.sample == 2)
			decodeQ[0] = decodeQQ[0] + 64;
		else
			decodeQ[0] = decodeQQ[0] + 128;

		for (int i = 1; i < para.data_size; i++)
			decodeQ[i] = (decodeQQ[i] + decodeQ[i - 1]);
	}





    string out_file;
	out_file = std::string("result/") + std::to_string(image_ID) + std::string("_dct_quantization_decode.bmp"); 
	//out_file = std::string("result/") + std::to_string(ID) + std::string("_dct_quantization_decode.bmp");

	ofstream fbmp3(out_file, ios::out | ios::binary);



	for (int i = 0; i < para.header_size; i++) {
		int8buffer = header[i];
		fbmp3.write((char*)&int8buffer, sizeof(unsigned char));

	}

	for (int i = 0; i < para.h; i++) {
		for (int j = 0; j < para.w; j++) {
			int8buffer = decodeQ[(para.h - 1 - i)*para.w + j];// *para.sample;
			//int8buffer = decodeQ[i*data_width + j] * sample;
			fbmp3.write((char*)&int8buffer, sizeof(unsigned char));
		}
	}

	finish_de = clock();
	

	///////////////////////////////////////////// output information /////////////////////////////////////////////
	string info_file = std::string("result/info_") + std::to_string(image_ID) + std::string("_dct.txt");
	//string info_file = std::string("result/info_") + std::to_string(ID) + std::string("_dct.txt");
	ofstream fdata(info_file, ios::out | ios::binary);
	duration_en = (float)(finish_en - start_en) / CLOCKS_PER_SEC;
	duration_de = (float)(finish_de - start_de) / CLOCKS_PER_SEC;

	fdata << "Data_width = " << para.w << " , Data_height = " << para.h << "\r\n";
	fdata << "Block num = " << sqrt(best_block_num) << " x " << sqrt(best_block_num) << "\r\n";
	fdata << "Sampling rate = " << para.sample << "\r\n";
	fdata << "Entropy_Q  = " << entropy_Q << " , Bitrate_Q  = " << bitrate_Q << "\r\n";
	fdata << "Entropy_dQ = " << entropy_dQ << " , Bitrate_dQ = " << bitrate_dQ << "\r\n";
	fdata << "Encode time : " << duration_en << " seconds\r\n";
	fdata << "Decode time : " << duration_de << " seconds\r\n";
	fdata.close();

	///////////////////////////////////////////////de-quantization////////////////////////////////////////////////

	////R=Q*C
	int *image_dct_decode = new int[para.data_size];
	for (int i = 0; i<para.h - para.point + 1; i += para.point)
		for (int j = 0; j < para.w - para.point + 1; j += para.point) {
			for (int y = 0; y<para.point; y++)
				for (int x = 0; x<para.point; x++)
					image_dct_decode[(j + y) * para.w + (i + x)] = Q50_table_8x8[y*para.point + x] *decodeQ[(j + y) * para.w + (i + x)];
		}

	//////////////////////////////////////output de-quantization image////////////////////////////////////////////
	string dct_file_decode;
	//if (para.filename == "bmp_file/baboon.bmp")
	//dct_file_decode = std::string("result/") + std::to_string(ID) + std::string("_dct_decode.bmp");
	dct_file_decode = std::string("result/") + std::to_string(image_ID) + std::string("_dct_decode.bmp");
	//else
	//	dct_file = "lout.bmp";

	ofstream fbmp4(dct_file_decode, ios::out | ios::binary);


	for (int i = 0; i < para.header_size; i++) {
		int8buffer = header[i];
		fbmp4.write((char*)&int8buffer, sizeof(unsigned char));

	}

	for (int i = 0; i < para.h; i++) {
		for (int j = 0; j < para.w; j++) {
			int8buffer = image_dct_decode[(para.h - 1 - i)*para.w + j];
			//int8buffer = image_dct[i*para.w + j];
			fbmp4.write((char*)&int8buffer, sizeof(unsigned char));
		}
	}

	///////////////////////////////////////////////iDCT///////////////////////////////////////////////////////////
	int*image_decode = new int[para.data_size];
	////decode_block=round(A'*decode_dct_block*A)+128
	for (int block_count = 0; block_count < best_block_num; block_count++) {
		MatrixXf  decode_block(block_dct_point[(int)log2(sqrt(best_block_num))][block_count], block_dct_point[(int)log2(sqrt(best_block_num))][block_count]);
		MatrixXf  decode_dct_block(block_dct_point[(int)log2(sqrt(best_block_num))][block_count], block_dct_point[(int)log2(sqrt(best_block_num))][block_count]);
		start_i = (block_count / (int)sqrt(best_block_num))*(para.h / (int)sqrt(best_block_num));
		start_j = (block_count % (int)sqrt(best_block_num))*(para.w / (int)sqrt(best_block_num));

		for (int i = start_i; i < start_i + para.h / sqrt(best_block_num) - block_dct_point[(int)log2(sqrt(best_block_num))][block_count] + 1; i += block_dct_point[(int)log2(sqrt(best_block_num))][block_count])
			for (int j = start_j; j < start_j + para.w / sqrt(best_block_num) - block_dct_point[(int)log2(sqrt(best_block_num))][block_count] + 1; j += block_dct_point[(int)log2(sqrt(best_block_num))][block_count]) {
				for (int y = 0; y < block_dct_point[(int)log2(sqrt(best_block_num))][block_count]; y++)
					for (int x = 0; x < block_dct_point[(int)log2(sqrt(best_block_num))][block_count]; x++)
						//decode_dct_block(x, y) = image_dct[(j + y) * para.w + (i + x)];
						decode_dct_block(x, y) = image_dct_decode[(j + y) * para.w + (i + x)];
				//decode_block = A.transpose()*decode_dct_block*A;
				if (block_dct_point[(int)log2(sqrt(best_block_num))][block_count] == 4)
					decode_block = A_4x4.transpose()*decode_dct_block*A_4x4;
				else if (block_dct_point[(int)log2(sqrt(best_block_num))][block_count] == 8)
					decode_block = A_8x8.transpose()*decode_dct_block*A_8x8;
				else
					decode_block = A_16x16.transpose()*decode_dct_block*A_16x16;
				for (int y = 0; y < block_dct_point[(int)log2(sqrt(best_block_num))][block_count]; y++)
					for (int x = 0; x < block_dct_point[(int)log2(sqrt(best_block_num))][block_count]; x++)
						image_decode[(j + y) * para.w + (i + x)] = round(decode_block(x, y)) + 128 / para.sample;
			
			}
	}



	///////////////////////////////////////////////output iDCT image//////////////////////////////////////////////
	string final_decode;
	//if (para.filename == "bmp_file/baboon.bmp")
	final_decode = std::string("result/") + std::to_string(image_ID) + std::string("_decode.bmp");
	//final_decode = std::string("result/") + std::to_string(ID) + std::string("_decode.bmp");
	//else
	//	dct_file = "lout.bmp";

	ofstream fbmp5(final_decode, ios::out | ios::binary);


	for (int i = 0; i < para.header_size; i++) {
		int8buffer = header[i];
		fbmp5.write((char*)&int8buffer, sizeof(unsigned char));

	}

	for (int i = 0; i < para.h; i++) {
		for (int j = 0; j < para.w; j++) {
			int8buffer = image_decode[(para.h - 1 - i)*para.w + j]*para.sample;
			//int8buffer = image_dct[i*para.w + j];
			fbmp5.write((char*)&int8buffer, sizeof(unsigned char));
		}
	}

	//////////////////////////////////////////////free space//////////////////////////////////////////////////////
	delete[]header;
	delete[]image;
	delete[]image_dct;
	delete[]image_dct_Q50;
	delete[]decodeQ;

	delete[]imageQ;
	delete[]imageQQ;
	delete[]counter;
	delete[]counter2;
	delete[]r;
	delete[]r2;
	delete[]bitcode;
	delete[]bitcodes;
	delete[]bitcode2;
	delete[]bitcodes2;
	delete[]lengthcount2;
	delete[]offset_codenum;
	delete[]offset_range;
	delete[]symbol_order;
	delete[]codenum_now;
	delete[]encode;

	delete[]image_dct_decode;
	delete[]image_decode;
	//////////////////////////////////////////////close files/////////////////////////////////////////////////////
	read_f.close();


	fbmp3.close();
	fbmp4.close();
	fbmp5.close();
	//fp.close();
	//fenco.close();
	//fdata.close();
//}
	system("Pause");
	return 0;
}

//function
parameter setInfoFromCMD(int argc, char *argv[], parameter para_default) {

	parameter para;
	// command line
	if (argc >= 2) {
		para.filename = argv[1];
	}
	if (argc >= 3) {
		para.result = argv[2];
	}
	if (argc >= 4) {
		para.header_size = atoi(argv[3]);
	}
	if (argc >= 5) {
		para.w = atoi(argv[4]);
	}
	if (argc >= 6) {
		para.h = atoi(argv[5]);
		para.data_size = para.w * para.h;
	}
	if (argc >= 7) {
		para.point = atoi(argv[6]);
	}
	else {
		para = para_default;
	}
	return para;
};
void write_csv(int w, int h, vector<vector<double>>& klt_basis, vector<vector<double>>& dct, vector<vector<double>>& cov, string filename, double eff) {

	fstream file;
	file.open(filename, ios::out | ios::binary);

	file << "cov" << endl;
	//write file
	for (int j = 0; j< h; j++) {
		for (int i = 0; i< w; i++) {
			file << (float)cov[j][i] << ",";
		}
		file << endl;
	}

	file << "klt basis" << endl;
	//write file
	for (int j = 0; j< h; j++) {
		for (int i = 0; i< w; i++) {
			file << (float)klt_basis[j][i] << ",";
		}
		file << endl;
	}

	file << "Dct" << endl;
	//write file
	for (int j = 0; j< h; j++) {
		for (int i = 0; i< w; i++) {
			file << (float)dct[j][i] << ",";
		}
		file << endl;
	}

	file << endl << "efficiency," << eff << endl;
}


int parent(int i)
{
	return (i - 1) / 2;
}

int leftchild(int i)
{
	return i * 2 + 1;
}

int rightchild(int i)
{
	return i * 2 + 2;
}

void heap_make(PQ *p, int i)
{
	int		leftc, rightc, smallest;
	nodeptr t;
	leftc = leftchild(i);
	rightc = rightchild(i);

	//?H?U?A find the smallest of parent, left, and right 
	if (leftc  < p->heap_size && p->A[leftc]->count < p->A[i]->count)
		smallest = leftc;
	else
		smallest = i;

	if (rightc < p->heap_size && p->A[rightc]->count < p->A[smallest]->count)
		smallest = rightc;

	if (smallest != i) //?ｸ・?A?p?G?e?z?澱c????? 
	{
		t = p->A[i];
		p->A[i] = p->A[smallest];
		p->A[smallest] = t;
		heap_make(p, smallest);
	}
}

void insert_pq(PQ *p, nodeptr r)
{
	int		i;

	p->heap_size++;
	i = p->heap_size - 1;

	while ((i > 0) && (p->A[parent(i)]->count > r->count))
	{
		p->A[i] = p->A[parent(i)];
		i = parent(i);
	}
	p->A[i] = r;
}

nodeptr extract_min_pq(PQ *p)
{
	nodeptr r;

	r = p->A[0];
	p->A[0] = p->A[p->heap_size - 1];// take the last and put it in the root 
	p->heap_size--;// one less thing in queue 
	heap_make(p, 0);// left and right are a heap, make the root a heap 
	return r;
}

nodeptr build_huffman(unsigned int freqs[], int size)
{
	int		i, n;
	nodeptr	x, y, z;
	PQ		p;
	p.A= new nodeptr[size];
	p.heap_size = 0;
	// ?H?U?Afor each character, make a heap/tree node with its value and frequency 

	for (i = 0; i < size; i++)
	{
		if (freqs[i] != 0)
		{   //this condition is important!?A?_?h???????v??????G?A?|?h?@??node 
			x = (nodeptr)malloc(sizeof(node));//this is a leaf of the Huffman tree 
			x->left = NULL;
			x->right = NULL;
			x->count = freqs[i];
			x->symbol = i;
			insert_pq(&p, x);


		}
	}
	while (p.heap_size > 1) {
		//?H?U make a new node z from the two least frequent
		z = (nodeptr)malloc(sizeof(node));
		x = extract_min_pq(&p);
		y = extract_min_pq(&p);

		z->left = x;
		z->right = y;
		z->symbol = z->right->symbol;
		z->count = x->count + y->count;
		insert_pq(&p, z);
	}

	/* return the only thing left in the queue, the whole Huffman tree */
	return extract_min_pq(&p);

}

void postorder(nodeptr currentnode, FILE *f)
{
	if (currentnode)
	{
		postorder(currentnode->left, f);
		postorder(currentnode->right, f);
		if (currentnode->count != 0)
			fprintf(f, "%d ", currentnode->symbol);

	}
}

void traverse(nodeptr r, 	// root of this subtree 
	int level, 	// current level  
	char code_now[], // code string up 
	char *codes[],// array of codes 
	int length[]
)
{
	if ((r->left == NULL) && (r->right == NULL))
	{
		code_now[level] = 0;
		codes[r->symbol] = _strdup(code_now);
		length[r->symbol] = level;
	}
	else
	{
		/*?H?U?Ago left with bit 0 */
		code_now[level] = '1';
		traverse(r->left, level + 1, code_now, codes, length);
		/*?H?U?Ago right with bit 1 */
		code_now[level] = '0';
		traverse(r->right, level + 1, code_now, codes, length);
	}


}
