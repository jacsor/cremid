#include <RcppArmadillo.h>
#include "helpers.h"   
#include "gibbs.h"  

using namespace Rcpp;
using namespace arma;
using namespace std;


MCMC::MCMC( mat Y, 
            vec C, 
            Rcpp::List prior,
            Rcpp::List mcmc,
            Rcpp::List state ) :
            Y(Y), 
            C(C)
{
       
      p = Y.n_cols;  
      n = Y.n_rows;  
      J = C.max() + 1;
      K = Rcpp::as<int>(prior["K"]);
      num_iter = Rcpp::as<int>(mcmc["nskip"]) * Rcpp::as<int>(mcmc["nsave"]);
      num_burnin = Rcpp::as<int>(mcmc["nburn"]);
      num_thin = Rcpp::as<int>(mcmc["nskip"]);    
      num_display = Rcpp::as<int>(mcmc["ndisplay"]);    

      epsilon_range = Rcpp::as<vec>(prior["epsilon_range"]);
      m_2 = Rcpp::as<vec>(prior["m_2"]);
      nu_2 = Rcpp::as<double>(prior["nu_2"]);    
      nu_1 = Rcpp::as<double>(prior["nu_1"]);    
      Psi_2 = Rcpp::as<mat>(prior["Psi_2"]);
      inv_S_2 = inv(Rcpp::as<mat>(prior["S_2"]));
      tau_k0 = Rcpp::as<vec>(prior["tau_k0"]);
      tau_alpha = Rcpp::as<vec>(prior["tau_alpha"]);
      tau_rho = Rcpp::as<vec>(prior["tau_rho"]);
      rho_pm = Rcpp::as<vec>(prior["point_masses_rho"]);
      tau_varphi = Rcpp::as<vec>(prior["tau_varphi"]);
      varphi_pm = Rcpp::as<vec>(prior["point_masses_varphi"]);
      merge_step = Rcpp::as<bool>(prior["merge_step"]);
      merge_par = Rcpp::as<double>(prior["merge_par"]);
      Z_input = Rcpp::as<uvec>(state["Z"]);
      truncation_type = Rcpp::as<std::string>(prior["truncation_type"]);
      tau_eta = Rcpp::as<vec>(prior["tau_eta"]);

      length_chain =  num_iter/num_thin;
      saveVarphi.set_size(length_chain);
      saveRho.set_size(length_chain);
      saveAlpha.set_size(length_chain);
      saveW.set_size(J,K,length_chain);
      saveK0.set_size(length_chain);
      saveEpsilon.set_size(length_chain);
      saveS.set_size(length_chain,K);
      saveR.set_size(length_chain,K);
      saveZ.set_size(length_chain,n);
      saveMu.set_size(J,K*p,length_chain);
      saveMu0.set_size(p,K,length_chain);
      saveOmega.set_size(p,K*p,length_chain);     

      main_loop(state);
                                  
}





void MCMC::main_loop(Rcpp::List state)
{    
  
  /* --- support variables --- */
  
  // counter
  int km = 0;
  // latent indicators
  uvec S(K);
  uvec R = Rcpp::as<uvec>(state["R"]);
  // number of obserations per group and per component
  mat N(J,K);
  // used in the swap step
  mat temp;
  vec indices;
  // latent assignemnts
  uvec Z = Z_input;

  /* --- parameters --- */
  
  // probability of misalignment and shift    
  double varphi = tau_varphi(0)/sum(tau_varphi);
  // proportion of the common mixture weights
  double rho = tau_rho(0)/sum(tau_rho);
  double eta = 0.5;
  // link between mean and covariance
  double k_0 = 1;
  // perturbation parameter for the mean
  double epsilon  = mean(epsilon_range); //   epsilon(0) = 0.001; epsilon(1) = 1.0;
  double epsilon_old = epsilon;
  double epsilon_par = sqrt(K);
  int epsilon_count = 0; 
  int epsilon_tot = 100;
  // mass parameter for the dirichlet prior on the mixture weights
  double alpha = 1;
  double alpha_old = alpha;
  double alpha_par  = sqrt(K);
  double alpha_count = 0; 
  int alpha_tot = 100; 
  // mixture weights
  mat logW(J, K); logW.fill( log(1.0/K));
  // mean \mu_{j,k}
  cube mu(J,p, K);
  // centering of mean locations \mu_k
  mat mu_0(p,K);  
  // covariance locations
  cube Sigma(p,p,K);
  // precision locations
  cube Omega(p,p,K);
  // centering for the Wishart prior 
  mat Sigma_1 = 10.0*eye<mat>(p,p); 
  mat Omega_1 = inv(Sigma_1);
  // mean of the mean
  vec m_1 = mean(Y,0).t(); 

  
  /* --- chain initialization --- */
    
  for(int k = 0; k < K; k++ )
  {
    List tempMuSigma = InitMuSigma(Z, k); 
    mu.slice(k) = Rcpp::as<mat>(tempMuSigma["mu"]); 
    mu_0.col(k) = Rcpp::as<vec>(tempMuSigma["mu_0"]);
    Omega.slice(k) = Rcpp::as<mat>(tempMuSigma["Omega"]);   
    Sigma.slice(k) = Rcpp::as<mat>(tempMuSigma["Sigma"]);   
  }
  
  // assign each observation to a mixture component
  
  List tempZ = UpdateZetas(mu, Omega, logW);   
  Z = Rcpp::as<uvec>(tempZ["Z"]);  
  N = Rcpp::as<mat>(tempZ["N"]);  
  
      
  /* --- let the chain run --- */

  for(int it=0; it<(num_iter + num_burnin); it++)
  {          
    
    if((it+1)%num_display==0)
      cout << "Iteration: " << it + 1 << " of " << num_iter + num_burnin << endl;
    
    // MERGE STEP
    if(merge_step)
    {
      for( int k=0; k< K - 1 ; k++ )
      {
        if( sum(N.col(k)) > 0 )
        {
          for(int kk=k+1; kk < K ; kk++)
          {
            if( sum(N.col(kk)) > 0  )
            {
              double kl_div = KL( mu_0.col(k), 
                                  mu_0.col(kk), 
                                  Sigma.slice(k), 
                                  Omega.slice(k), 
                                  Sigma.slice(kk), 
                                  Omega.slice(kk) ) / epsilon ;
              if( kl_div < R::qchisq(merge_par, (double)p, 1, 0) )
              {
                N.col(k) = N.col(k) + N.col(kk);
                N.col(kk) = zeros<vec>(J);
                
                List tempSMuSigma = PriorSMuSigma(   varphi,
                                                     Sigma_1, 
                                                     Omega_1, 
                                                     k_0, 
                                                     epsilon,
                                                     m_1 );                                                    
                S(kk) = Rcpp::as<unsigned>(tempSMuSigma["S"]);
                mu.slice(kk) = Rcpp::as<mat>(tempSMuSigma["mu"]); 
                mu_0.col(kk) = Rcpp::as<vec>(tempSMuSigma["mu_0"]);
                Omega.slice(kk) = Rcpp::as<mat>(tempSMuSigma["Omega"]);   
                Sigma.slice(kk) = Rcpp::as<mat>(tempSMuSigma["Sigma"]);                                        
                        
              }
  
            }
  
          }
        }
  
      }
    }
                    
    alpha_old = alpha;  
    alpha = UpdateAlpha( alpha, N, R, alpha_par );
    if( it <= num_burnin )
    {
      if( alpha != alpha_old )
        alpha_count++;
           
      if( (it+1)  % alpha_tot == 0)
      {
        if( alpha_count < 30 )
          alpha_par *= 1.1;
        if( alpha_count > 50 )  
          alpha_par *= 0.9;
        alpha_count = 0;        
      }      
    }  
    else
    {
        if( alpha != alpha_old )
          alpha_count++;
    }    
    
    if(truncation_type == "adaptive")
      R = UpdateR(R, N, eta, alpha);
    else
      R = swap_step(R, N, tau_rho, alpha);
        
    rho = updateRho( N, R);
    
    logW = UpdateLogWs( N, R, rho, alpha );  
    
    List tempZ = UpdateZetas(mu, Omega, logW);
    Z = Rcpp::as<uvec>(tempZ["Z"]);  
    N = Rcpp::as<mat>(tempZ["N"]);  
    
    for(int k=0; k < K; k++)
    {      
      uvec Z_k = arma::find(Z==k);
      List tempSMuSigma = UpdateSMuSigma( Z,
                                          k,
                                          mu_0.col(k),
                                          varphi,
                                          Sigma_1, 
                                          Omega_1,
                                          k_0, 
                                          epsilon,
                                          m_1 ); 
      S(k) = Rcpp::as<unsigned>(tempSMuSigma["S"]);
      mu.slice(k) = Rcpp::as<mat>(tempSMuSigma["mu"]); 
      mu_0.col(k) = Rcpp::as<vec>(tempSMuSigma["mu_0"]);
      Omega.slice(k) = Rcpp::as<mat>(tempSMuSigma["Omega"]);   
      Sigma.slice(k) = Rcpp::as<mat>(tempSMuSigma["Sigma"]);   
      
    }  
    
    k_0 =  UpdateK0(Omega, mu_0, m_1);
    
    Sigma_1 = UpdateSigma1(Omega);
    Omega_1 = inv_sympd(Sigma_1);
    
    m_1 = UpdateM1( k_0, mu_0, Omega );
    
    epsilon_old = epsilon;
    epsilon = UpdateEpsilon(epsilon, S, mu_0, mu, Omega, epsilon_par);
    if( it <= num_burnin )
    {
      if( epsilon != epsilon_old )
        epsilon_count++;
        
      if( (it+1)  % epsilon_tot == 0)
      {
        if( epsilon_count < 30 )
          epsilon_par *= 1.1;
        if( epsilon_count > 50 )  
          epsilon_par *= 0.9;
        epsilon_count = 0;
      }      
    }  
    else
    {
      if( epsilon != epsilon_old )
        epsilon_count++;
    }
  
    varphi = UpdateVarphi( S );
    
    if( (it+1 > num_burnin) && ((it+1) % num_thin == 0))
    {  
      // save chain
      saveVarphi(km) = varphi;
      saveRho(km) = rho;
      saveK0(km) = k_0;
      saveEpsilon(km) = epsilon;
      saveAlpha(km) = alpha;
      saveW.slice(km) = exp(logW); 
      saveOmega.slice(km) = reshape( mat(Omega.memptr(), Omega.n_elem, 1, false), p, K*p);  
      saveMu.slice(km) = reshape( mat(mu.memptr(), mu.n_elem, 1, false), J, K*p);   
      saveMu0.slice(km) = mu_0;
      saveS.row(km) = S.t();          
      saveZ.row(km) = Z.t();  
      saveR.row(km) = R.t();  
      
      km++;        
    }
      
      
  }
  
  cout << endl << "MH acceptance rate " << endl;
  cout << "epsilon: " << (double)epsilon_count/num_iter << endl;
  cout << "alpha: " << alpha_count / (double)num_iter << endl << endl;
  
}     




Rcpp::List MCMC::InitMuSigma(uvec Z, int k)
{
  mat EmpCov = eye<mat>(p,p) + cov(Y);
  mat tempInit = mvrnormArma(1, zeros<vec>(p) , eye<mat>(p,p));
  mat mu_k(J,p);
  vec mu_0k(p);  
  mat Sigma_k(p,p);
  mat Omega_k(p,p);
  
  uvec Zk = arma::find(Z==k);
  if( Zk.n_elem > 0 )
  {
    mu_0k = mean(Y.rows(Zk),0).t(); 
    
    if( Zk.n_elem > (uint)p + 2 )
      Sigma_k = cov(Y.rows(Zk));  
    else
      Sigma_k = EmpCov;  
      
    Omega_k = inv(Sigma_k);
    for(int j=0; j<J; j++)  
        mu_k.row(j) = mean(Y.rows(Zk),0);           
  }
  else
  {
    mu_0k = mean(Y,0).t() + tempInit.row(k).t();
    Sigma_k = EmpCov;  
    Omega_k = inv(Sigma_k);

    if( k < K / 2 )
    {
      for(int j=0; j<J; j++)  
        mu_k.row(j) = mean(Y,0).t();        
    }
    else
    {
      for(int j=0; j<J; j++)  
        mu_k.row(j) = mean(Y,0) + tempInit.row(k);            
    }
  }                 
  
  return Rcpp::List::create(  
  Rcpp::Named( "mu" ) = mu_k,
  Rcpp::Named( "mu_0" ) = mu_0k,
  Rcpp::Named( "Sigma" ) = Sigma_k,
  Rcpp::Named( "Omega" ) = Omega_k) ;   
  
}





Rcpp::List MCMC::UpdateZetas(   arma::cube mu,
                                arma::cube Omega,
                                arma::mat logW )
{
  mat log_like(n, K);
  uvec C_j;
  int j;  // used as private index of for loop inside omp below
  
  #pragma omp parallel for private(j, C_j)
  for(int k = 0; k < K; k++)
  {
    uvec index(1);
    index(0) = k;
    for(j=0; j < J; j++)
    {
      C_j = arma::find(C==j);
      log_like.submat(C_j,  index) = dmvnrm_arma_precision(
        Y.rows(C_j),
        mu.slice(k).row(j),
        Omega.slice(k)  );
    }
  }  
  
  Rcpp::List zetas_output = GenerateZetas(log_like, logW);
  
  return zetas_output;
}



Rcpp::List MCMC::GenerateZetas( arma::mat log_like,
                                arma::mat logW  )
{
  // J is the number of data sets,
  // K is the number of mixture components
  // So, N has a row for each data set, a column for each component
  mat N(J, K);
  N.fill(0);

  // Zeta vector assigning each data point to a component
  uvec Z(n);
  Z.fill(0);

  // generate a new random uniform distribution for this iteration
  NumericVector U = runif(n);

  // log likelihood
  double tot_log_like = 0.0;
  vec prob;
  vec probsum;
  double x;
  bool not_assigned;

  int i;
  int k;
  #pragma omp parallel for private(k, prob, probsum, x, not_assigned)
  for(i = 0; i < n; i++)
  {
    prob = exp(log_like.row(i).t() + logW.row(C(i)).t());
    probsum = cumsum(prob);
    x = U(i) * sum(prob);
    not_assigned = true;
    for (k = 0; (k < K) && not_assigned; k++)
    {
      if(x <= probsum(k))
      {
        Z(i) = k;
        not_assigned = false;
      }
    }
  }

  for(i = 0; i < n; i++) {
    N(C(i), Z(i))++;
    tot_log_like += log_like(i,Z(i));
  }

  return Rcpp::List::create(  Rcpp::Named( "Z" ) = Z,
                              Rcpp::Named( "N" ) = N,
                              Rcpp::Named( "tot_log_like" ) = tot_log_like ) ;
}



                               
double MCMC::UpdateAlpha(double alpha, arma::mat N, arma::uvec R, double alpha_par)
{
  uvec R0 = find(R==0);
  uvec R1 = find(R==1);
  int K_0 = R0.n_elem;
  int K_1 = R1.n_elem;  
  
  double output = alpha;    
  double log_ratio = 0;
  double temp = rgammaBayes(  pow( alpha, 2 ) * alpha_par, 
                        alpha * alpha_par );                      
  
  log_ratio += R::dgamma(alpha, pow(temp,2)* alpha_par, 1/temp/alpha_par, 1);                          
  log_ratio -= R::dgamma(temp, pow(alpha,2)* alpha_par, 1/alpha/alpha_par, 1);  
  log_ratio += R::dgamma(temp, tau_alpha(0), 1/tau_alpha(1), 1);
  log_ratio -= R::dgamma(alpha, tau_alpha(0), 1/tau_alpha(1), 1);
  
  if(K_0 > 0)
  {
    log_ratio += marginalLikeDirichlet( sum(N.cols(R0),0).t(), (temp/K_0)*ones<vec>(K_0)  );
    log_ratio -= marginalLikeDirichlet( sum(N.cols(R0),0).t(), (alpha/K_0)*ones<vec>(K_0)  );
  }

  if(K_1 > 0)
  {
    mat N1 = N.cols(R1);
    for(int j = 0; j < J; j++)
    {
        log_ratio += marginalLikeDirichlet( N1.row(j).t(), (temp/K_1)*ones<vec>(K_1)  );
        log_ratio -= marginalLikeDirichlet( N1.row(j).t(), (alpha/K_1)*ones<vec>(K_1)  );
    }    
  }

  if( exp(log_ratio) > R::runif(0,1) )
      output = temp;
      
  return output;
}



double MCMC::updateRho( arma::mat N, arma::uvec R)
{
  uvec R_0 = find(R == 0);
  uvec R_1 = find(R == 1);
  double n_0 = accu(N.cols(R_0));
  double n_1 = accu(N.cols(R_1));
  return( rbeta(1, tau_rho(0) + n_0, tau_rho(1) + n_1 )(0) );
}



arma::mat MCMC::UpdateLogWs(   arma::mat N, 
                               arma::uvec R,
                               double rho,
                               double alpha  )
{
  
  mat logW(J,K);  
  uvec R0 = find(R==0);
  uvec R1 = find(R==1);
  int K_0 = R0.n_elem;
  int K_1 = R1.n_elem;  
  uvec jindex(1);
  
  if( K_0 > 0)
  {
    vec pi0 = rDirichlet( sum(N.cols(R0),0).t() +  alpha * ones<vec>(K_0) / K_0  );
    mat N0 = N.cols(R0);
    for(int j=0; j<J; j++)
    {
      jindex(0) = j;
      logW.submat(jindex, R0) = pi0.t() + log(rho);
    }
      
  }
  
  if(K_1 > 0)
  {
    mat N1 = N.cols(R1);
    for(int j=0; j<J; j++)
    {
      jindex(0) = j;
      logW.submat(jindex, R1) = rDirichlet( N1.row(j).t() +  alpha * ones<vec>(K_1) / K_1  ).t() + log(1.0 - rho);
    }
      
  }
  
  return logW ;      
}






Rcpp::List MCMC::UpdateSMuSigma(  arma::uvec Z,
                                  int k, 
                                  arma::vec mu_0,
                                  double varphi,
                                  arma::mat Sigma_1, 
                                  arma::mat Omega_1, 
                                  double k_0, 
                                  double epsilon,
                                  arma::vec m_1 ) 
{ 
  uvec Z_k = arma::find(Z==k);  
  mat data_group = Y.rows(Z_k);
  vec C_k = C(Z_k);
  int p = data_group.n_cols;
  int N_k = data_group.n_rows;   
  unsigned S_k;
  mat Omega(p,p);
  mat Sigma(p,p);
  mat mu(p,J);
  mat mu_0new(p,1);  
  double r = R::runif(0,1);
  
  if(N_k == 0)
  {
    Omega = rWishartArma(Omega_1, nu_1);
    Sigma = inv_sympd( Omega );       
    mu_0new = trans(mvrnormArma(1, m_1, Sigma/k_0));    
    
    if( r < varphi )
    {
      S_k = 0;
      mu = repmat(mu_0, 1, J);
    }
    else 
    {
      S_k = 1;
      for(int j=0; j<J; j++)
        mu.col(j) = trans( mvrnormArma(1, mu_0new,  Sigma*epsilon ));      
    }
  }
  else  // N_k > 0
  {
    double sign_0, sign_1;
    mat Psi_0(p,p), Psi_1(p,p);
    double extra_piece_var_1 = 0;
    vec m1_0(p), m1_1(p);
    double log_extra_piece = 0;
    vec log_det_Psi(2); log_det_Psi.fill(0);
    
    vec n_jk(J);
    mat mean_jk(p,J); mean_jk.fill(0);
    vec mean_k = mean(data_group,0).t();
    mat SS_jk(p,p), ss_jk_1(p,p);
    
    vec log_prob(2); 
    double prob;
    double log_prob_sum;
    
    if( ( varphi > 0  ) && ( varphi < 1) )
    {
      // marginal likelihood under model 0
      mat mean_k_rep = repmat( mean_k.t(), N_k, 1);
      mat SS_k = ( data_group - mean_k_rep ).t() * ( data_group - mean_k_rep );
      mat ss_k = N_k * ( mean_k - mu_0 ) * ( mean_k - mu_0 ).t();        
      Psi_0 = inv_sympd( Sigma_1 + SS_k + ss_k );    
      log_det(log_det_Psi(0), sign_0, Psi_0); 
      m1_0 = ( N_k * mean_k + k_0 * m_1 ) / (N_k + k_0);
      
      log_prob(0) = (nu_1 + N_k)/2.0 * log_det_Psi(0) + log( varphi );

      // marginal likelihood under model 1 
      extra_piece_var_1 = k_0;
      m1_1 = k_0 * m_1;     
      SS_jk.fill(0);
      ss_jk_1.fill(0);

      for(int j=0; j<J; j++)
      {
        uvec indices = find(C_k==j);
        n_jk(j) = indices.n_elem;     
        
        if (n_jk(j) > 0)
        {
            mean_jk.col(j) = mean(data_group.rows(indices),0).t();
            mat mean_jk_rep = repmat( trans(mean_jk.col(j)),(int)n_jk(j), 1);
            SS_jk = SS_jk + (data_group.rows(indices) - mean_jk_rep).t() * ( data_group.rows(indices) - mean_jk_rep );
            ss_jk_1 = ss_jk_1 + (mean_jk.col(j) - mu_0) * (mean_jk.col(j) - mu_0).t() / (epsilon + 1.0/n_jk(j));
            log_extra_piece +=  log( epsilon*n_jk(j) + 1.0 );
            extra_piece_var_1 +=  n_jk(j) / (epsilon * n_jk(j) + 1.0);
            m1_1 = m1_1 +  n_jk(j) / (epsilon * n_jk(j) + 1.0) * mean_jk.col(j);
           
        }
      }          
      Psi_1 = inv_sympd( Sigma_1 + SS_jk + ss_jk_1 );
      log_det(log_det_Psi(1), sign_1, Psi_1);
      
      log_prob(1) = (nu_1 + N_k)/2.0 * log_det_Psi(1) - p/2.0 * log_extra_piece + log( 1.0 - varphi );
      
      log_prob_sum = log_exp_x_plus_exp_y( log_prob(0), log_prob(1) );
      prob = exp( log_prob(0) - log_prob_sum  );
    }
    else
      prob = varphi;
    
    if( r < prob )
    {
      S_k = 0;
      Omega = rWishartArma(Psi_0, nu_1 + N_k);
      Sigma = inv_sympd( Omega ); 
      mu_0new = trans( mvrnormArma(1, m1_0, Sigma/(k_0+N_k)) );      
      mu = repmat(mu_0new, 1, J);
    }
    else 
    {
      S_k = 1;
      Omega = rWishartArma(Psi_1, nu_1 + N_k);
      Sigma = inv_sympd( Omega ); 
      m1_1 = m1_1 / extra_piece_var_1;
      mu_0new = trans( mvrnormArma(1, m1_1, Sigma/extra_piece_var_1));
      for(int j=0; j<J; j++)
      {
        if( n_jk(j) > 0 )
        {
          mu.col(j) = trans(mvrnormArma(1, (n_jk(j)*mean_jk.col(j) + 1.0/epsilon*mu_0new)/(n_jk(j) + 1.0/epsilon), 
            Sigma/(n_jk(j) + 1.0/epsilon)));
        }
        else
          mu.col(j) = trans( mvrnormArma(1, mu_0new,  Sigma*epsilon) );           
      }      
    }
  }
           
  return Rcpp::List::create(  
    Rcpp::Named( "S" ) = S_k,
    Rcpp::Named( "mu" ) = mu.t(),
    Rcpp::Named( "mu_0" ) = mu_0new,
    Rcpp::Named( "Sigma" ) = Sigma,
    Rcpp::Named( "Omega" ) = Omega) ;    
};





double MCMC::UpdateK0(  arma::cube Omega, 
                        arma::mat mu_0,
                        arma::vec m_1  )
{
  double tau_2_tot = tau_k0(1);
  double tau_1_tot = tau_k0(0) + p*K;
  for(int k=0; k < K; k++)
      tau_2_tot += as_scalar( (mu_0.col(k) - m_1).t() * Omega.slice(k) * (mu_0.col(k) - m_1));  

  return rgammaBayes(tau_1_tot/2, tau_2_tot/2);
};






arma::mat MCMC::UpdateSigma1(arma::cube Omega)
{
  mat psi_2_tot = Psi_2;
  for(int k=0; k< K; k++)
      psi_2_tot += Omega.slice(k);

  return( rWishartArma(inv_sympd(psi_2_tot), K*nu_1 + nu_2) );
  
};


arma::vec MCMC::UpdateM1(   double k_0, 
                            arma::mat mu_0, 
                            arma::cube Omega )
{
  mat precision = inv_S_2;
  vec meanM = inv_S_2*m_2;
  for(int k=0; k< K; k++)
  {
      precision += k_0*Omega.slice(k);
      meanM += k_0 * ( Omega.slice(k)*mu_0.col(k) );
  }
  mat variance = inv_sympd(precision);
  mat output = mvrnormArma(1, variance*meanM, variance);
  return( output.row(0).t() );
};



double MCMC::UpdateEpsilon( double epsilon,
                            arma::uvec S, 
                            arma::mat mu_0, 
                            arma::cube mu, 
                            arma::cube Omega,
                            double epsilon_par  )
{
  
  double counts;
  uvec index = find(S==1);
  counts = index.n_elem;
  
  double epsilon_new;
  double numerator = 0;
  double denominator = 0;
  double output = epsilon;
  double eps_diff = epsilon_range(1) - epsilon_range(0);
  double a,b;
  
  if(counts==0)
    output = R::runif( epsilon_range(0), epsilon_range(1) );   // sample from the prior
  else
  {
    a = exp(  log( epsilon_par ) + log( epsilon - epsilon_range(0) ) - log( eps_diff ) ) ;
    b = epsilon_par - a ;
    
    epsilon_new = (eps_diff)*as<double>(rbeta(1, a, b)) + epsilon_range(0);
    
    denominator = dGeneralizedBeta(epsilon_new, a, b , epsilon_range  );
    
    a = exp(  log( epsilon_par ) + log( epsilon_new - epsilon_range(0) ) - log( eps_diff ) );
    b = epsilon_par - a ;
    
    numerator = dGeneralizedBeta(epsilon, a, b , epsilon_range  );
    
    for(int k=0; k< K; k++)
    {
      if( S(k) == 1 )
      {
        numerator += sum(dmvnrm_arma_precision(mu.slice(k),mu_0.col(k).t(), 
          Omega.slice(k)/epsilon_new));
        denominator += sum(dmvnrm_arma_precision(mu.slice(k),mu_0.col(k).t(), 
          Omega.slice(k)/epsilon ));
      }
    }      
    // cout << exp(numerator - denominator) << endl;
    
    if( exp(numerator - denominator) > R::runif(0,1) )
      output = epsilon_new;
    
  }
    
  return output;  
  
  
};







double MCMC::UpdateVarphi( arma::uvec S )
{
  if( varphi_pm(0)==1 )
    return 0.0;
  else if( varphi_pm(1)==1 )  
    return 1.0;
  else
  {
    vec counts(2); 
    uvec index = find(S==0);
    counts(0) = index.n_elem;
    index = find(S==1);
    counts(1) = index.n_elem;
        
    for(int s=0; s<2; s++)
    {      
      if( ( counts(s) == K ) && ( varphi_pm(s) > 0 ) )
      {
         double log_den = log_exp_x_plus_exp_y( log(varphi_pm(s)) , 
          log(1 - sum(varphi_pm)  ) + marginalLikeDirichlet(counts, tau_varphi) );
    
        if( exp( log(varphi_pm(s)) - log_den  ) <   R::runif(0,1) )
          return as<double>(rbeta(1, tau_varphi(0) + counts(0), tau_varphi(1) + counts(1)));
        else  
          return (1-s);         
      }    
    }
    return as<double>(rbeta(1, tau_varphi(0) + counts(0), tau_varphi(1) + counts(1)));
  }      
}



Rcpp::List MCMC::get_chain()
{
  return Rcpp::List::create(  
    Rcpp::Named( "alpha" ) = saveAlpha,
    Rcpp::Named( "epsilon" ) = saveEpsilon,
    Rcpp::Named( "k_0" ) = saveK0,
    Rcpp::Named( "mu" ) = saveMu,
    Rcpp::Named( "mu_0" ) = saveMu0,
    Rcpp::Named( "Omega" ) = saveOmega,
    Rcpp::Named( "rho" ) = saveRho,
    Rcpp::Named( "Z" ) = saveZ,
    Rcpp::Named( "S" ) = saveS,
    Rcpp::Named( "R" ) = saveR,
    Rcpp::Named( "varphi" ) = saveVarphi,
    Rcpp::Named( "w" ) = saveW
    );
};








Rcpp::List MCMC::PriorSMuSigma(   double varphi,
                                  arma::mat Sigma_1, 
                                  arma::mat Omega_1, 
                                  double k_0, 
                                  double epsilon,
                                  arma::vec m_1 ) 
{ 
  unsigned S_k;
  mat Omega(p,p);
  mat Sigma(p,p);
  mat mu(p,J);
  mat mu_0new(p,1);  
  double r = R::runif(0,1);

  Omega = rWishartArma(Omega_1, nu_1);
  Sigma = inv_sympd( Omega );       
  mu_0new = trans(mvrnormArma(1, m_1, Sigma/k_0));    
    
  if( r < varphi )
  {
    S_k = 0;
    mu = repmat(mu_0new, 1, J);
  }
  else 
  {
    S_k = 1;
    for(int j=0; j<J; j++)
      mu.col(j) = trans( mvrnormArma(1, mu_0new,  Sigma*epsilon ));      
  }
  
  return Rcpp::List::create(  
    Rcpp::Named( "S" ) = S_k,
    Rcpp::Named( "mu" ) = mu.t(),
    Rcpp::Named( "mu_0" ) = mu_0new,
    Rcpp::Named( "Sigma" ) = Sigma,
    Rcpp::Named( "Omega" ) = Omega) ;    
};




arma::uvec MCMC::UpdateR(arma::uvec R, arma::mat N, double eta, double alpha)
{
  // the code is written for even K
  vec u = randu(K/2);  
  uvec R_temp = R;
  vec log_like(4); log_like.fill(0); // 00, 10, 01, 11
  vec like(4); like.fill(0);
  int part_sum_R;
  int count;
  int k_1, k_2;
  
  for(int k = 0; k < (K/2); k++)
  {
    k_1 = sampling( ones<vec>(K) );
    k_2 = sampling( ones<vec>(K-1) );
    if(k_2 == k_1)
      k_2 = K-1;
    
    part_sum_R = sum(R) - R(k_1) - R(k_2);
    count = 0;
    for(int ii = 0; ii < 2; ii++)
    {
      for(int jj = 0; jj < 2; jj++)
      {
        R_temp = R;
        R_temp(k_1) = jj;
        R_temp(k_2) = ii;
        if( ii == 0 )
          log_like(count) += log(eta);    
        else
          log_like(count) += log(1.0 - eta);            
        if( jj == 0 )
          log_like(count) += log(eta);    
        else
          log_like(count) += log(1.0 - eta);  
        log_like(count) += supportR(N, R_temp, eta, alpha);
        count++;
      }
    }    
    
    like = exp(log_like - max(log_like));
    if( part_sum_R > 1)
    {
      u(k) *= sum(like);
      if( u(k) < like(0) )
      {
        R(k_1) = 0;
        R(k_2) = 0;
      }
      else if( u(k) - like(0) < like(1) )
      {
        R(k_1) = 1;
        R(k_2) = 0;
      }
      else if( u(k) - like(0) - like(1)  < like(2) )
      {
        R(k_1) = 0;
        R(k_2) = 1;
      }
      else
      {
        R(k_1) = 1;
        R(k_2) = 1;        
      }
        
    }
    else if( part_sum_R == 1 )
    {
      like(0) = 0;
      u(k) *= sum(like);
      if( u(k) < like(1) )
      {
        R(k_1) = 1;
        R(k_2) = 0;
      }
      else if( u(k) - like(1) < like(2) )
      {
        R(k_1) = 0;
        R(k_2) = 1;
      }
      else
      {
        R(k_1) = 1;
        R(k_2) = 1;        
      }      
    }
    else  // part_sum_R == 0
    {
      like(1) = 0;
      like(2) = 0;
      u(k) *= sum(like);
      if( u(k) < like(0)  )
      {
        R(k_1) = 0;
        R(k_2) = 0;
      }
      else
      {
        R(k_1) = 1;
        R(k_2) = 1;        
      }
    }
  }

  return R;
}

double MCMC::supportR(arma::mat N, arma::uvec R_temp, double eta, double alpha)
{
  uvec R_temp_0, R_temp_1;
  int K_temp_0, K_temp_1;
  mat N_temp_0, N_temp_1;
  double output;
  
  R_temp_0 = find(R_temp == 0);
  R_temp_1 = find(R_temp == 1);
  K_temp_0 = R_temp_0.n_elem;
  K_temp_1 = R_temp_1.n_elem;
  N_temp_0 = N.cols(R_temp_0);
  N_temp_1 = N.cols(R_temp_1);
  
  output = lgamma( tau_rho(0) + accu(N.cols(R_temp_0)) ) + lgamma(tau_rho(1) + accu(N.cols(R_temp_1)) );
  if( K_temp_0 > 0)
    output += marginalLikeDirichlet(  sum(N_temp_0, 0).t(), (alpha/K_temp_0)*ones<vec>(K_temp_0) );
  if( K_temp_1 > 0)
  {
    for(int j = 0; j < J; j++)
      output += marginalLikeDirichlet(  N_temp_1.row(j).t(), (alpha/K_temp_1)*ones<vec>(K_temp_1) );
  } 
  return output;
  
}



arma::uvec MCMC::swap_step(arma::uvec R, arma::mat N, vec tau_rho, double alpha)
{
  uvec R_0 = find(R == 0);
  uvec R_1 = find(R == 1);
  int K_0 = R_0.n_elem;
  int K_1 = R_1.n_elem;
  
  vec indices(2);
  indices(0) = R_0(sampling( ones<vec>(K_0)));
  indices(1) = R_1(sampling( ones<vec>(K_1)));
  
  uvec R_new = R;
  R_new(indices(0)) = R(indices(1));
  R_new(indices(1)) = R(indices(0));
  
  uvec R_0_new = find(R_new == 0);
  uvec R_1_new = find(R_new == 1);
  
  vec counts(2), counts_new(2);
  counts_new(0) = accu(N.cols(R_0_new));
  counts_new(1) = accu(N.cols(R_1_new));
  counts(0) = accu(N.cols(R_0));
  counts(1) = accu(N.cols(R_1));
  
  double numerator = marginalLikeDirichlet( counts_new, tau_rho );
  double denominator =  marginalLikeDirichlet( counts, tau_rho );
  
  numerator += marginalLikeDirichlet( sum(N.cols(R_0_new),0).t(), alpha*ones<vec>(K_0)/K_0 );
  denominator += marginalLikeDirichlet( sum(N.cols(R_0),0).t(), alpha*ones<vec>(K_0)/K_0 );
  
  mat N_1_new = N.cols(R_1_new);
  mat N_1 = N.cols(R_1);
  for(int j = 0; j < J; j++)
  {
    numerator += marginalLikeDirichlet( N_1_new.row(j).t(), alpha*ones<vec>(K_1)/K_1 );
    denominator += marginalLikeDirichlet( N_1.row(j).t(), alpha*ones<vec>(K_1)/K_1 );
  }
  
  double acceptance_log_prob = numerator - denominator;
  
  if( exp(acceptance_log_prob) > R::runif(0,1) )
    return R_new;
  else   
    return R;
  
}
