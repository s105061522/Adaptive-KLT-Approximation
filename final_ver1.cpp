//****hw04****//

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
int data_width = 512;
int data_height = 512;
int data_size = 512 * 512;
int header_size = 1078;
int dct_point = 8;

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
	nodeptr* A = new nodeptr[256 / sample];
};

typedef struct PQ2 {
	int	heap_size;
	nodeptr* A = new nodeptr[256 / sample * 2 - 1];
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


void heap_make2(PQ2 *p, int i);
void insert_pq2(PQ2 *p, nodeptr r);//insert node to pq
nodeptr extract_min_pq2(PQ2 *p);//extract min node in pq

nodeptr sort_for_nodes(node * nodearray, int size);

int main(int argc, char *argv[])
{

	parameter para;

	/////////////////////////////////////////////////parameter//////////////////////////////////////////////
	para.filename = "bmp_file/baboon.bmp";
	para.header_size = header_size;
	para.w = data_width;
	para.h = data_height;
	para.point = dct_point;
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
		}
	}

	/////////////////////////////////////////////DCT decorrelation/////////////////////////////////////////////
	//cout << "***DCT***" << endl;
	MatrixXf  A(para.point, para.point);

	for (int i = 0; i<para.point; i++)
		for (int j = 0; j < para.point; j++) {
			if (i == 0)
				A(i, j) = sqrt((float)2 / para.point)*sqrt((float)1 / 2)*cos(2 * EIGEN_PI*(2 * j + 1)*i / (4 * para.point));
			else
				A(i, j) = sqrt((float)2 / para.point) * 1 * cos(2 * EIGEN_PI*(2 * j + 1)*i / (4 * para.point));
		}

	//write_f << "Tdct" << endl;
	//cout << "Tdct:" << endl;
	//write file
	//for (int i = 0; i< para.point; i++) {
	//	for (int j = 0; j< para.point; j++) {
			//write_f << A(i, j) << ",";
	//		cout << A(i, j) << " ";
	//	}
		//write_f << endl;
	//	cout << endl;
	//}


	////////////////////////////////////////////////////DCT image///////////////////////////////////////////////


	MatrixXf  block(para.point, para.point);
	MatrixXf  dct_block(para.point, para.point);
	int *image_dct = new  int[para.data_size];
	for (int i = 0; i<para.h - para.point + 1; i += para.point)
		for (int j = 0; j < para.w - para.point + 1; j += para.point) {
			for (int y = 0; y<para.point; y++)
				for (int x = 0; x<para.point; x++)
					block(x, y) = image[(j + y) * para.w + (i + x)] - 128;//Because the DCT is designed to work on pixel values ranging from -128 to 127
			dct_block = A*block*A.transpose();
			for (int y = 0; y < para.point; y++)
				for (int x = 0; x < para.point; x++)
					image_dct[(j + y) * para.w + (i + x)] = dct_block(x, y);
		}


	///////////////////////////////////////////////output DCT image///////////////////////////////////////////////
	const char* dct_file;
	//if (para.filename == "bmp_file/baboon.bmp")
		dct_file = "baboon_dct.bmp";
	//else
	//	dct_file = "lout.bmp";

	ofstream fbmp(dct_file, ios::out | ios::binary);


	for (int i = 0; i < para.header_size; i++) {
		int8buffer = header[i];
		fbmp.write((char*)&int8buffer, sizeof(char));

	}

	for (int i = 0; i < para.h; i++) {
		for (int j = 0; j < para.w; j++) {
			int8buffer = image_dct[(para.h - 1 - i)*para.w + j];
			//int8buffer = image_dct[i*para.w + j];
			fbmp.write((char*)&int8buffer, sizeof(char));
		}
	}
	///////////////////////////////////////////////quantization///////////////////////////////////////////////////
	unsigned char* Q50_table_8x8 = new unsigned char[64];
	int *image_dct_Q50 = new int[para.data_size];
	FILE *Q_in = NULL;
	Q_in = fopen("Q_table_8x8.txt", "r");
	
	for (int i = 0; i < 64; i++) {
		fscanf(Q_in, "%d", &Q50_table_8x8[i]);
		//cout << (unsigned)Q_table_8x8[i] << endl;
		//system("pause");
	}
	fclose(Q_in);

	////C=round(D_block/Q)
	for (int i = 0; i<para.h - para.point + 1; i += para.point)
		for (int j = 0; j < para.w - para.point + 1; j += para.point) {
			for (int y = 0; y<para.point; y++)
				for (int x = 0; x<para.point; x++)
					image_dct_Q50[(j + y) * para.w + (i + x)] = round((float)image_dct[(j + y) * para.w + (i + x)]/Q50_table_8x8[y*para.point+x]);
		}

	///////////////////////////////////////////////output DCT Q50 image////////////////////////////////////////////
	const char* dct_Q50_file;
	//if (para.filename == "bmp_file/baboon.bmp")
	dct_Q50_file = "baboon_dct_Q50.bmp";
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

	///////////////////////////////////////////////encoding///////////////////////////////////////////////////////
	clock_t start_en, finish_en, start_de, finish_de;
	start_en = clock();
	float duration_en, duration_de;
	float entropy_Q = 0.0;
	float bitrate_Q = 0.0;
	float entropy_dQ = 0.0;
	float bitrate_dQ = 0.0;

	int *imageQ = new int[para.data_size];
	int *imageQQ = new int[para.data_size];
	unsigned char temp = 0;
	unsigned int *counter = new unsigned int[256 / para.sample];       //0~256/sample
	unsigned int *counter2 = new unsigned int[256 / para.sample * 2 - 1];//-(256 / sample - 1)~(256 / sample - 1)
	nodeptr r; // root of Huffman tree  
	nodeptr r2;

	for (int i = 0; i < para.data_size; i++)
		imageQ[i] = 0;

	for (int i = 0; i < para.data_size; i++)
		imageQQ[i] = 0;


	for (int i = 0; i < 256 / para.sample; i++)
		counter[i] = 0;




	for (int i = 0; i < 256 / para.sample * 2 - 1; i++) 
		counter2[i] = 0;


	for (int i = 0; i < para.data_size; i++)
		//imageQ[i] = image[i] / para.sample;
		//imageQ[i] = image_dct[i] / para.sample;
		imageQ[i] = image_dct_Q50[i]/para.sample;

	for (int i = 0; i < para.data_size; i++) {

		if (i == 0 && para.sample == 4)
			imageQQ[i] = imageQ[i] - 32;
		else if (i == 0 && para.sample == 2)
			imageQQ[i] = imageQ[i] - 64;
		else if (i == 0 && para.sample == 1)
			imageQQ[i] = imageQ[i] - 128;
		else
			imageQQ[i] = imageQ[i] - imageQ[i - 1];

		for (int j = 0; j < 256 / para.sample; j++)
			if (imageQ[i] == j)
				counter[j]++;

	}

	for (int i = 0; i < para.data_size; i++)
		for (int j = 0; j < 256 / para.sample * 2 - 1; j++)
			if (imageQQ[i] == j - (256 / para.sample - 1))
				counter2[j]++;



	for (int i = 0; i < 256 / para.sample; i++) {

		if (!counter[i])
			entropy_Q += 0;
		else
			entropy_Q += (-1)*((float)counter[i] / para.data_size)*log2f((float)counter[i] / para.data_size);
	}

	char **bitcodes = new char*[256 / para.sample]; // array of codes, 1 bit per char 
	char *bitcode = new char[100];   // a place to hold 1 code each time
	int * length = new int[256 / para.sample];
	r = build_huffman(counter, 256 / para.sample);

	for (int i = 0; i < 256 / para.sample; i++)
		bitcodes[i] = 0;

	traverse(r, 0, bitcode, bitcodes, length);
	//delete[]r;
	//delete[]bitcode;
	//delete[]bitcodes;

	for (int i = 0; i < 256 / para.sample; i++) {


		if (counter[i] == para.data_size)
			bitrate_Q += 0;
		else
			bitrate_Q += ((float)counter[i] / para.data_size)*length[i];
	}



	for (int i = 0; i < 256 / para.sample * 2 - 1; i++) {

		if (!counter2[i])
			entropy_dQ += 0;
		else
			entropy_dQ += (-1)*((float)counter2[i] / para.data_size)*log2f((float)counter2[i] / para.data_size);

	}

	char **bitcodes2 = new char*[256 / para.sample * 2 - 1]; // array of codes, 1 bit per char 
	char *bitcode2 = new char[100];   // a place to hold 1 code each time
	int *length2 = new int[256 / para.sample * 2 - 1];
	r2 = build_huffman(counter2, 256 / para.sample * 2 - 1);
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

	for (int i = 0; i < 256 / para.sample * 2 - 1; i++)
		bitcodes2[i] = 0;

	traverse(r2, 0, bitcode2, bitcodes2, length2);

	for (int i = 0; i < 256 / para.sample * 2 - 1; i++)                    //by the result of  traverse for bitcodes     
		if (bitcodes2[i] != 0) {

			symbol_size++;
		}


	symbol_order = new int[symbol_size];
	for (int i = 0; i < symbol_size; i++)
		symbol_order[i] = 0;

	for (int i = 0; i < 256 / para.sample * 2 - 1; i++)
		if (length2[i] > maxlength)
			maxlength = length2[i];

	offset_codenum = new int[maxlength];
	offset_range = new int[maxlength];


	for (int i = 0; i < maxlength; i++) {

		offset_codenum[i] = 0;
		offset_range[i] = 0;
	}

	int* lengthcount2 = new int[maxlength];

	for (int i = 0; i < maxlength; i++)
		lengthcount2[i] = 0;

	for (int i = 0; i < maxlength; i++)
		for (int j = 0; j < 256 / sample * 2 - 1; j++)
			if (length2[j] == i + 1) {
				lengthcount2[i]++;
				symbol_order[codenum] = j - (256 / sample - 1);
				codenum++;

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


	for (int i = 0; i < 256 / para.sample * 2 - 1; i++) {

		if (counter2[i] == para.data_size)
			bitrate_dQ += 0;
		else
			bitrate_dQ += ((float)counter2[i] / para.data_size)*length2[i];
	}

	const char* bin_file ="baboon_dct_encode.bin";
	ofstream fenco(bin_file, ios::out | ios::binary);

	for (int i = 0; i < para.data_size; i++)
		for (int j = 0; j < symbol_size; j++)
			if (imageQQ[i] == symbol_order[j])
				codenum_now[i] = j;

	for (int i = 0; i < para.data_size; i++)
		for (int j = 0; j < maxlength; j++)
			if (codenum_now[i] > offset_codenum[j] || codenum_now[i] == offset_codenum[j])
				encode[i] = offset_range[j] + (codenum_now[i] - offset_codenum[j]);



	int data_count = 0;
	for (int i = 0; i < para.data_size; i++) {

		int *encode_bit = new int[length2[imageQQ[i] + 256 / para.sample - 1]];

		for (int j = 0; j < length2[imageQQ[i] + 256 / para.sample - 1]; j++) {
			encode_bit[length2[imageQQ[i] + 256 / para.sample - 1] - j - 1] = encode[i] % 2;
			encode[i] /= 2;
		}
		for (int j = 0; j < length2[imageQQ[i] + 256 / para.sample - 1]; j++) {
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



	fenco.close();
	finish_en = clock();

	///////////////////////////////////////////////decoding///////////////////////////////////////////////////////
	start_de = clock();

	out_count = 0;
	int bit_count = 0;
	ifstream fdeco(bin_file, ios::in | ios::binary);
	_int64 temp_decode = 0;
	char decode_bit;
	unsigned char tmp;
	int *decode = new int[para.data_size];
	int *decodeQQ = new int[para.data_size];
	int *decodeQ = new int[para.data_size];

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


	const char* out_file;
	out_file = "baboon_dct_Q50_decode.bmp";

	ofstream fbmp3(out_file, ios::out | ios::binary);



	for (int i = 0; i < para.header_size; i++) {
		int8buffer = header[i];
		fbmp3.write((char*)&int8buffer, sizeof(unsigned char));

	}

	for (int i = 0; i < para.h; i++) {
		for (int j = 0; j < para.w; j++) {
			int8buffer = decodeQ[(para.h - 1 - i)*para.w + j] * para.sample;
			//int8buffer = decodeQ[i*data_width + j] * sample;
			fbmp3.write((char*)&int8buffer, sizeof(unsigned char));
		}
	}

	finish_de = clock();


	///////////////////////////////////////////// output information /////////////////////////////////////////////
	const char* info_file = "info_baboon_dct.txt";
	ofstream fdata(info_file, ios::out | ios::binary);
	duration_en = (float)(finish_en - start_en) / CLOCKS_PER_SEC;
	duration_de = (float)(finish_de - start_de) / CLOCKS_PER_SEC;

	fdata << "Data_width = " << para.w << " , Data_height = " << para.h << "\r\n";
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
					image_dct_decode[(j + y) * para.w + (i + x)] = Q50_table_8x8[y*para.point + x] *decodeQ[(j + y) * para.w + (i + x)] * para.sample;
		}

	//////////////////////////////////////output de-quantization image////////////////////////////////////////////
	const char* dct_file_decode;
	//if (para.filename == "bmp_file/baboon.bmp")
	dct_file_decode = "baboon_dct_decode.bmp";
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
	////decode_block=round(A'*decode_dct_block*A)+128
	MatrixXf  decode_block(para.point, para.point);
	MatrixXf  decode_dct_block(para.point, para.point);
	int*image_decode = new int[para.data_size];
	for (int i = 0; i<para.h - para.point + 1; i += para.point)
		for (int j = 0; j < para.w - para.point + 1; j += para.point) {
			for (int y = 0; y<para.point; y++)
				for (int x = 0; x<para.point; x++)
					//decode_dct_block(x, y) = image_dct[(j + y) * para.w + (i + x)];
					decode_dct_block(x, y) = image_dct_decode[(j + y) * para.w + (i + x)];
			decode_block = A.transpose()*decode_dct_block*A;
			for (int y = 0; y < para.point; y++)
				for (int x = 0; x < para.point; x++)
					image_decode[(j + y) * para.w + (i + x)] = round(decode_block(x, y)) +128;
		}

	///////////////////////////////////////////////output iDCT image//////////////////////////////////////////////
	const char* final_decode;
	//if (para.filename == "bmp_file/baboon.bmp")
	final_decode = "baboon_decode.bmp";
	//else
	//	dct_file = "lout.bmp";

	ofstream fbmp5(final_decode, ios::out | ios::binary);


	for (int i = 0; i < para.header_size; i++) {
		int8buffer = header[i];
		fbmp5.write((char*)&int8buffer, sizeof(unsigned char));

	}

	for (int i = 0; i < para.h; i++) {
		for (int j = 0; j < para.w; j++) {
			int8buffer = image_decode[(para.h - 1 - i)*para.w + j];
			//int8buffer = image_dct[i*para.w + j];
			fbmp5.write((char*)&int8buffer, sizeof(unsigned char));
		}
	}
	//////////////////////////////////////////////free space//////////////////////////////////////////////////////
	delete[]image;
	delete[]image_dct;
	delete[]image_dct_Q50;
	delete[]decodeQ;
	delete[]image_dct_decode;
	delete[]image_decode;
	
	delete[]header;
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
	//////////////////////////////////////////////close files/////////////////////////////////////////////////////
	read_f.close();
	fbmp.close();
	fbmp2.close();
	fbmp3.close();
	fbmp4.close();
	fbmp5.close();
	//fp.close();
	//fenco.close();
	//fdata.close();
	//system("Pause");
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

void heap_make2(PQ2 *p, int i)
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
		heap_make2(p, smallest);
	}
}

void insert_pq2(PQ2 *p, nodeptr r)
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

nodeptr extract_min_pq2(PQ2 *p)
{
	nodeptr r;

	r = p->A[0];
	p->A[0] = p->A[p->heap_size - 1];// take the last and put it in the root 
	p->heap_size--;// one less thing in queue 
	heap_make2(p, 0);// left and right are a heap, make the root a heap 
	return r;
}

nodeptr build_huffman(unsigned int freqs[], int size)
{
	int		i, n;
	nodeptr	x, y, z;
	PQ		p;
	PQ2     p2;

	p.heap_size = 0;
	p2.heap_size = 0;
	// ?H?U?Afor each character, make a heap/tree node with its value and frequency 

	for (i = 0; i < size; i++)
	{
		if (freqs[i] != 0)
		{                  //this condition is important!?A?_?h???????v??????G?A?|?h?@??node 
			x = (nodeptr)malloc(sizeof(node));//this is a leaf of the Huffman tree 
			x->left = NULL;
			x->right = NULL;
			x->count = freqs[i];
			if (size == 256 / sample) {
				x->symbol = i;
				insert_pq(&p, x);
			}

			else {
				x->symbol = i;// -(256 / sample - 1);
				insert_pq2(&p2, x);
			}


		}
	}
	if (size == 256 / sample) {
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
	else {
		while (p2.heap_size > 1) {
			//?H?U make a new node z from the two least frequent
			z = (nodeptr)malloc(sizeof(node));
			x = extract_min_pq2(&p2);
			y = extract_min_pq2(&p2);

			z->left = x;
			z->right = y;
			z->symbol = z->right->symbol;
			z->count = x->count + y->count;
			insert_pq2(&p2, z);
		}

		/* return the only thing left in the queue, the whole Huffman tree */
		return extract_min_pq2(&p2);
	}


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
