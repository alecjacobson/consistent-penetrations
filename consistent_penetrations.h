#include <igl/fast_winding_number.h>
#include <igl/per_vertex_normals.h>
#include <igl/edges.h>
#include <igl/sparse.h>
#include <igl/ismember.h>
#include <igl/max.h>
#include <igl/adjacency_matrix.h>
#include <igl/find.h>
#include <igl/sort.h>
#include <igl/sortrows.h>
#include <igl/matlab_format.h>
#include <igl/slice_mask.h>
#include <igl/slice.h>
#include <igl/embree/EmbreeIntersector.h>
#include <igl/Hit.h>
#include <igl/speye.h>
#include <Eigen/Core>
#include <algorithm>
#include <cassert>

namespace igl
{
namespace embree
{
  // "Consistent Penetration Depth Estimation" Heidelberg 2007.
  //
  // This is a direct translation of gptoolbox's consistent_penetrations.m
  //
  // Inputs:
  //   VA  #VA by 3 list of A's mesh vertex positions
  //   EleA  #EleA by 3 list of A's element indices into rows of VA 
  //   VB  #VB by 3 list of B's mesh vertex positions
  //   FB  #FB by 3 list of B's facet indices into rows of VB
  // Outputs:
  //   DR  #VA by 3 list of penetration depth vectors (forces)
  inline void consistent_penetrations(
    const Eigen::MatrixXd & VA,
    const Eigen::MatrixXi & EleA,
    const Eigen::MatrixXd & VB,
    const Eigen::MatrixXi & FB,
    Eigen::MatrixXd & DR);
}
}

// Implementation

namespace igl
{
namespace embree
{
  inline void ray_mesh_intersect(
      const Eigen::MatrixXf & source,
      const Eigen::MatrixXf & dir,
      const igl::embree::EmbreeIntersector & ei,
      Eigen::VectorXi & id,
      Eigen::VectorXd & t,
      Eigen::MatrixXd & lambda
      )
  {
    using namespace igl;
    using namespace igl::embree;
    const int n = source.rows();
    id.resize(n);
    t.resize(n);
    lambda.resize(n,3);

    for(int si = 0;si<n;si++)
    {
      Eigen::Vector3f s = source.row(si);
      Eigen::Vector3f d = dir.row(si);
      igl::Hit hit;
      const float tnear = 1e-4f;
      if(ei.intersectRay(s,d,hit,tnear))
      {
        id(si) = hit.id;
        t(si) = hit.t;
        lambda(si,0) = 1.0-hit.u-hit.v;
        lambda(si,1) = hit.u;
        lambda(si,2) = hit.v;
      }else
      {
        id(si) = -1;
        t(si) = std::numeric_limits<float>::infinity();
        lambda.row(si).setZero();
      }
    }
  }
  inline void ray_mesh_intersect(
      const Eigen::MatrixXf & source,
      const Eigen::MatrixXf & dir,
      const Eigen::MatrixXf & V,
      const Eigen::MatrixXi & F,
      Eigen::VectorXi & id,
      Eigen::VectorXd & t,
      Eigen::MatrixXd & lambda
      )
  {
    igl::embree::EmbreeIntersector ei;
    ei.init(V,F,true);
    return ray_mesh_intersect(source,dir,ei,id,t,lambda);
  }
inline void consistent_penetrations(
  const Eigen::MatrixXd & VA,
  const Eigen::MatrixXi & EleA,
  // Augh, pass by copy so we can manipulate below
  Eigen::MatrixXi EA,
  const Eigen::SparseMatrix<int>  & AA,
  const Eigen::MatrixXd & VB,
  const Eigen::MatrixXi & FB,
  Eigen::MatrixXd & DR)
{
  typedef Eigen::Array<bool,Eigen::Dynamic,1> ArrayX1b;
  // dimension
  const int dim = VA.cols();
  assert(dim == 3 && "Only 3D supported");
  DR = Eigen::MatrixXd::Zero(VA.rows(),dim);
  // Determine which vertices of A are inside of B
  Eigen::VectorXi I;
  {
    Eigen::VectorXd W;
    igl::fast_winding_number(VB,FB,VA,W);
    I = (W.array().abs()>0.5).cast<int>();
  }
  Eigen::SparseMatrix<int> II;
  igl::speye(AA.rows(),AA.cols(),II);
  Eigen::VectorXi bI = I;
  {
    Eigen::VectorXi NI = 1-I.array();
    Eigen::VectorXi tmp  = (AA+II)*NI;
    for(int i = 0;i<bI.size();i++)
    {
      bI(i) = (bI(i)>0)  && (tmp(i)>0);
    }
  }

  Eigen::MatrixXi IEA(EA.rows(),EA.cols());
  for(int i = 0;i<EA.rows();i++)
  {
    for(int j = 0;j<EA.cols();j++)
    {
      IEA(i,j) = I(EA(i,j));
    }
  }
  Eigen::VectorXi crossing;
  igl::find( (IEA.rowwise().sum().array() == 1).eval(), crossing);
  Eigen::VectorXi HI;
  Eigen::VectorXd T;
  Eigen::MatrixXd B;
  {
    Eigen::MatrixXf source(crossing.size(),dim);
    Eigen::MatrixXf dir(crossing.size(),dim);
    for(int i = 0;i<crossing.size();i++)
    {
      source.row(i) = VA.row(EA(crossing(i),0)).cast<float>();
      dir.row(i) = 
        (VA.row(EA(crossing(i),1))- VA.row(EA(crossing(i),0))).cast<float>();
    }
    ray_mesh_intersect(source,dir,VB.cast<float>(),FB,HI,T,B);
    ArrayX1b valid(crossing.size(),1);
    for(int i = 0;i<crossing.size();i++)
    {
      valid(i) = HI(i)>=0 && (T(i)<1.0);
    }
    igl::slice_mask(Eigen::VectorXi(crossing),valid,1,crossing);
    igl::slice_mask(Eigen::VectorXi(HI),valid,1,HI);
    igl::slice_mask( Eigen::VectorXd(T),valid,1,T);
    igl::slice_mask( Eigen::MatrixXd(B),valid,1,B);
  }
  Eigen::MatrixXi CI(crossing.size(),2);
  CI << crossing,HI;
  ArrayX1b good(CI.rows(),1);
  for(int i =0;i<good.rows();i++)
  {
    good(i) = (IEA(CI(i,0),0)+ IEA(CI(i,0),1)) == 1;
  }
  igl::slice_mask(Eigen::MatrixXi(CI),good,1,CI);
  igl::slice_mask( Eigen::VectorXd(T),good,1,T);
  igl::slice_mask( Eigen::MatrixXd(B),good,1,B);
  if(CI.rows() == 0)
  {
    // DR = 0
    return;
  }
  // sort so that inside vertex comes first
  for(int i = 0;i<CI.rows();i++)
  {
    const bool flip = IEA(CI(i,0),1);
    if(flip)
    {
      // Why doesn't this work?
      // std::swap(EA(CI(i,0),0),EA(CI(i,0),1));
      const int tmp = EA(CI(i,0),0);
      EA(CI(i,0),0) = EA(CI(i,0),1);
      EA(CI(i,0),1) = tmp;
      T(i) = 1-T(i);
    }
  }
  // only keep farthest per-crossing-edge
  Eigen::VectorXi MI;
  {
    Eigen::SparseMatrix<double> M;
    igl::sparse(
        Eigen::VectorXi::LinSpaced(T.rows(),0,T.rows()-1),
        CI.col(0).eval(),
        T,
        T.rows(),
        CI.col(0).maxCoeff()+1,
        M);
    Eigen::VectorXd MV;
    igl::max(M,1,MV,MI);
    igl::slice_mask(Eigen::VectorXi(MI),(MV.array()>0).eval(),1,MI);
  }
  igl::slice(Eigen::MatrixXi(CI),MI,1,CI);
  igl::slice( Eigen::VectorXd(T),MI,1,T);
  igl::slice( Eigen::MatrixXd(B),MI,1,B);
  Eigen::MatrixXd PA(CI.rows(),dim);
  Eigen::MatrixXi EACI;
  igl::slice(EA,CI.col(0).eval(),1,EACI);
  for(int i = 0;i<PA.rows();i++)
  {
    PA.row(i) = VA.row(EACI(i,0))+T(i)*(VA.row(EACI(i,1))-VA.row(EACI(i,0)));
  }
  Eigen::VectorXd D = Eigen::VectorXd::Zero(VA.rows(),1);
  Eigen::MatrixXd R = Eigen::MatrixXd::Zero(VA.rows(),dim);
  {
    Eigen::VectorXi CbJ;
    Eigen::VectorXi find_bI;
    igl::find(bI,find_bI);
    {
      Eigen::VectorXi _;
      igl::ismember(EACI.col(0).eval(),find_bI,_,CbJ);
    }
    Eigen::VectorXi CbI = Eigen::VectorXi::LinSpaced(CI.rows(),0,CI.rows()-1);

    Eigen::MatrixXd VAbI;
    igl::slice_mask(VA,(bI.array()>0).eval(),1,VAbI);
    Eigen::VectorXd W(CbJ.rows(),1);
    for(int i = 0;i<W.rows();i++)
    {
      W(i) = 1. / ( PA.row(CbI(i)) - VAbI.row(CbJ(i))).squaredNorm();
    }
    Eigen::MatrixXd NB;
    igl::per_vertex_normals(VB,FB,NB);
    Eigen::MatrixXd NP(B.rows(),dim);
    for(int i = 0;i<B.rows();i++)
    {
      NP.row(i) = 
        (
         B(i,0) * NB.row(FB(CI(i,1),0)) + 
         B(i,1) * NB.row(FB(CI(i,1),1)) + 
         B(i,2) * NB.row(FB(CI(i,1),2))
         ).normalized();
      for(int j = 0;j<NP.cols();j++)
      {
        assert(NP(i,j) == NP(i,j) && "NP should be a number");
      }
    }
    const int nbI = find_bI.size();
    Eigen::VectorXd WW = Eigen::VectorXd::Zero(nbI,1);
    for(int i = 0;i<CbJ.rows();i++)
    {
      WW(CbJ(i)) += W(i);
    }
    for(int i = 0;i<WW.rows();i++)
    {
      if(WW(i) == 0)
      {
        WW(i) = 1;
      }
    }
    Eigen::VectorXd DbI = Eigen::VectorXd::Zero(nbI,1);
    Eigen::MatrixXd RbI = Eigen::MatrixXd::Zero(nbI,dim);
    for(int i = 0;i<CbJ.rows();i++)
    {
      DbI(CbJ(i)) += W(i)*(PA.row(CbI(i))-VAbI.row(CbJ(i))).dot(NP.row(CbI(i)));
      RbI.row(CbJ(i)) += W(i)*NP.row(CbI(i));
    }
    for(int i = 0;i<nbI;i++)
    {
      DbI(i) /= WW(i);
      if(DbI(i)==0)
      {
        RbI.row(i).setConstant(0);
      }else
      {
        RbI.row(i) = (RbI.row(i)/WW(i)).eval().normalized().eval();
      }
    }
    for(int i = 0;i<nbI;i++)
    {
      D(find_bI(i)) = DbI(i);
      R.row(find_bI(i)) = RbI.row(i);
    }
  }

  // Processed points
  Eigen::VectorXi pI = Eigen::VectorXi::Zero(VA.rows(),1);
  while(true)
  {
    for(int i = 0;i<bI.rows();i++)
    {
      pI(i) = pI(i) || bI(i);
    }
    // new border points
    bool some_new = false;
    {
      Eigen::VectorXi tmp = AA*pI;
      for(int i = 0;i<I.rows();i++)
      {
        bI(i) = (tmp(i)>0) && I(i) && !pI(i);
        if(bI(i))
        {
          some_new = true;
        }
      }
    }
    if(!some_new)
    {
      break;
    }
    Eigen::SparseMatrix<int> AApIbI;
    Eigen::VectorXi find_bI;
    igl::find(bI,find_bI);
    Eigen::VectorXi find_pI;
    igl::find(pI,find_pI);
    igl::slice(AA,find_pI,find_bI,AApIbI);
    Eigen::VectorXi AI,AJ,AV;
    igl::find(AApIbI,AI,AJ,AV);
    Eigen::MatrixXd VAbI;
    igl::slice_mask(VA,(bI.array()>0).eval(),1,VAbI);
    Eigen::MatrixXd VApI;
    igl::slice_mask(VA,(pI.array()>0).eval(),1,VApI);
    Eigen::MatrixXd RpI;
    igl::slice_mask(R, (pI.array()>0).eval(),1,RpI);
    Eigen::VectorXd DpI;
    igl::slice_mask(D, (pI.array()>0).eval(),1,DpI);
    for(int i = 0;i<RpI.rows();i++)
    {
      for(int j = 0;j<RpI.cols();j++)
      {
        assert(RpI(i,j) == RpI(i,j) && " should be a number");
      }
    }
    const int nbI = find_bI.size();
    Eigen::VectorXd mu(AI.rows(),1);
    for(int i = 0;i<mu.rows();i++)
    {
      mu(i) = 1. / ( VApI.row(AI(i)) - VAbI.row(AJ(i))).squaredNorm();
    }
    Eigen::VectorXd mumu = Eigen::VectorXd::Zero(nbI,1);
    for(int i = 0;i<mu.rows();i++)
    {
      mumu(AJ(i)) += mu(i);
    }
    for(int i = 0;i<mumu.rows();i++)
    {
      if(mumu(i) == 0)
      {
        mumu(i) = 1;
      }
    }
    Eigen::VectorXd DbI = Eigen::VectorXd::Zero(nbI,1);
    Eigen::MatrixXd RbI = Eigen::MatrixXd::Zero(nbI,dim);
    for(int i = 0;i<AJ.rows();i++)
    {
      DbI(AJ(i)) += 
        mu(i)*((VApI.row(AI(i))-VAbI.row(AJ(i))).dot(RpI.row(AI(i))) + 
        DpI(AI(i)));
      RbI.row(AJ(i)) += mu(i)*RpI.row(AI(i));
    }
    for(int i = 0;i<nbI;i++)
    {
      DbI(i) /= mumu(i);
      if(DbI(i)==0)
      {
        RbI.row(i).setConstant(0);
      }else
      {
        RbI.row(i) = (RbI.row(i)/mumu(i)).eval().normalized().eval();
      }
    }
    for(int i = 0;i<nbI;i++)
    {
      D(find_bI(i)) = DbI(i);
      R.row(find_bI(i)) = RbI.row(i);
    }

  }

  for(int i = 0;i<DR.rows();i++)
  {
    DR.row(i) = D(i)*R.row(i);
  }

};
inline void consistent_penetrations(
  const Eigen::MatrixXd & VA,
  const Eigen::MatrixXi & EleA,
  const Eigen::MatrixXd & VB,
  const Eigen::MatrixXi & FB,
  Eigen::MatrixXd & DR)
{
  Eigen::MatrixXi EA;
  igl::edges(EleA,EA);
  {
    // Just for consistency with matlab
    Eigen::MatrixXi EA1;
    igl::sort(EA,2,true,EA1);
    igl::sortrows(EA1,true,EA);
  }
  Eigen::SparseMatrix<int> AA;
  igl::adjacency_matrix(EA,AA);
  // EleA might not touch all of VA. It'd be better to restrict further
  // computation to the part that is touched. But this will do for now.
  AA.conservativeResize(VA.rows(),VA.rows());
  return consistent_penetrations(VA,EleA,EA,AA,VB,FB,DR);
}
}
}

