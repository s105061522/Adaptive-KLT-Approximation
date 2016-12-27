#include <iostream>
#include <Eigen/Dense>
#include <Eigen/SVD>

using Eigen::MatrixXf;
using Eigen::JacobiSVD;

int main(int argc, char* argv[])
{
  //equivalent MATLAB code: m=[3 -1; 2.5 1.5]; [u,s,v] = svd(m)
  MatrixXf m(2,2);
  m(0,0) = 3;
  m(1,0) = 2.5;
  m(0,1) = -1;
  m(1,1) = 1.5;
  std::cout << "Here is the matrix m:" << std::endl << m << std::endl;
  Eigen::JacobiSVD<Eigen::MatrixXf> *svd = new Eigen::JacobiSVD<Eigen::MatrixXf>(m, Eigen::ComputeThinU | Eigen::ComputeThinV);
  std::cout << "Its singular values are:" << std::endl << svd->singularValues() << std::endl;
  std::cout << "Its left singular vectors are the columns of the thin U matrix:" << std::endl << svd->matrixU() << std::endl;
  std::cout << "Its right singular vectors are the columns of the thin V matrix:" << std::endl << svd->matrixV() << std::endl;

  system("Pause");
}
