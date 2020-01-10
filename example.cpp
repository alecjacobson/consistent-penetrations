#include "consistent_penetrations.h"
#include <igl/readDMAT.h>
#include <igl/readMESH.h>

int main(int argc, char * argv[])
{
  Eigen::MatrixXd VA;
  Eigen::MatrixXi TA,FA;
  igl::readMESH("../A.mesh",VA,TA,FA);
  Eigen::MatrixXd VB;
  Eigen::MatrixXi TB,FB;
  igl::readMESH("../B.mesh",VB,TB,FB);

  Eigen::MatrixXd DR;
  igl::embree::consistent_penetrations(VA,TA,VB,FB,DR);


  Eigen::MatrixXd DRm;
  igl::readDMAT("../DR.dmat",DRm);
  std::cout<<"linf with matlab: "<<(DR-DRm).array().abs().maxCoeff()<<std::endl;
}
