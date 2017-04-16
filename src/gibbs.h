#ifndef GIBBS_H
#define GIBBS_H

#include "RcppArmadillo.h"
#include <omp.h>

using namespace Rcpp;
using namespace arma;


class MCMC
{
  private:
  mat Y;            // the data (n,J)
  vec C;            // the group membership (n,1)
  int K;            // number of mixture components 
  int J;            // the number of groups
  int n;            // number of observations
  int p;            // observation dimension
  int num_iter, num_burnin, num_thin, num_display;   // number of iteration, burnin and thinning
  
  /* --- hyperparameters --- */
  std::string truncation_type; // fixed vs adaptive
  vec epsilon_range; 
  double nu_2;        // nu_2 = p + 2
  double nu_1;   // nu_1 = p + 2
  // Hyperparamter of the Inverse Wishart on Psi_1
  mat Psi_2; //  = eye<mat>(p,p);  
  // mean of the Normal prior on m_1
  vec m_2; // (p);  m_2.zeros();    
  // Covariance of the Normal prior on m_1
  mat inv_S_2; // =  eye<mat>(p,p); S_2 = S_2/1000;   
  // k_0 prior parameters
  vec tau_k0;  //  tau.fill(4);
  // alpha parameters
  vec tau_alpha;  // (2); tau_alpha.fill(1);
  // rho parameters
  vec rho_pm;  //(2); rho_0.fill(0.0);
  vec tau_rho;  // (2)
  // varphi parameters
  vec tau_varphi; // (2)
  vec varphi_pm; // (2); varphi_0.fill(0);  
  bool merge_step;  // 
  double merge_par;
  // latent indicator initial values
  uvec Z_input; 
  // eta parameters
  vec tau_eta;
  
  int length_chain; 
  
  vec saveRho, saveK0, saveEpsilon, saveVarphi;
  vec saveAlpha;
  cube saveW, saveMu, saveMu0, saveOmega;
  umat saveZ, saveS, saveR;
  
  
  void main_loop(Rcpp::List state);
  Rcpp::List InitMuSigma(uvec Z, int k);
  
  Rcpp::List GenerateZetas( mat loglike,
                            arma::mat logW );

  Rcpp::List UpdateZetas(   arma::cube mu, 
                            arma::cube Omega, 
                            arma::mat logW );
                  

  double UpdateAlpha(double alpha, 
                     arma::mat N, 
                     arma::uvec R, 
                     double alpha_par);
    
  double updateRho( arma::mat N, arma::uvec R);
  
  arma::mat UpdateLogWs(   arma::mat N, 
                           arma::uvec R,
                           double rho,
                           double alpha  );
                            
  Rcpp::List UpdateSMuSigma(  arma::uvec Z,
                              int k,  
                              arma::vec mu_0,
                              double varphi,
                              arma::mat Sigma_1, 
                              arma::mat Omega_1, 
                              double k_0, 
                              double epsilon,
                              arma::vec m_1 );      
                              
  double UpdateK0(  arma::cube Omega, 
                    arma::mat mu_0,
                    arma::vec m_1  );        
                    
  arma::mat UpdateSigma1(arma::cube Omega);
  
  arma::vec UpdateM1(  double k_0, 
                        arma::mat mu_0, 
                        arma::cube Omega );
                        
  double UpdateEpsilon( double epsilon,
                        arma::uvec S, 
                        arma::mat mu_0, 
                        arma::cube mu, 
                        arma::cube Omega,
                        double epsilon_par  );                      
                  
  double UpdateVarphi( arma::uvec S );
  
  Rcpp::List PriorSMuSigma(   double varphi,
                            arma::mat Sigma_1, 
                            arma::mat Omega_1, 
                            double k_0, 
                            double epsilon,
                            arma::vec m_1 ); 
                            
  arma::uvec UpdateR(arma::uvec R, arma::mat N, double eta, double alpha);
  double supportR(arma::mat N, arma::uvec R_temp, double eta, double alpha);

    arma::uvec swap_step(   arma::uvec R, 
                          arma::mat N, 
                          vec tau_rho, 
                          double alpha);

                                        

  
  public:
  
  // constructor 
  MCMC( mat Y, 
        vec C, 
        Rcpp::List prior,
        Rcpp::List mcmc,
        Rcpp::List state );
        
  Rcpp::List get_chain();
      
  
};



#endif


