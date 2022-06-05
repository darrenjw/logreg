/*
fit-bayes.c

Random walk MH in C

*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

/* macros */
#define N 200
#define P 8
#define ITERS 10000
#define THIN 1000

/* function headers */
double ll(gsl_vector *beta);
double lprior(gsl_vector *beta);
double lpost(gsl_vector *beta);
double kernel(double ll);

/* global variables! */
gsl_matrix *x;
gsl_vector *y, *tymo, *eta, *beta, *betaP;
gsl_rng *r;

/* main runner function */
int main(int argc, char *argv[])
{
  FILE *s;
  int i,j;
  double tmp, ll;
  char *tmps;

  fprintf(stderr, "RW MH for Bayesian logistic regression in C\n");

  fprintf(stderr, "Reading data\n");
  s=fopen("../pima.data", "r");
  if (s == NULL) {
    perror("error opening data file, pima.data");
    exit(1);
  }
  tmps = malloc(20);
  x = gsl_matrix_calloc(N, P);
  y = gsl_vector_calloc(N);
  for (i=0; i<N; i++) {
    gsl_matrix_set(x, i, 0, 1.0);
    for (j=1; j<P; j++) {
      fscanf(s, "%lf", &tmp);
      gsl_matrix_set(x, i, j, tmp);
    }
    fscanf(s, "%s", tmps);
    if (strcmp(tmps, "Yes") == 0) {
      gsl_vector_set(y, i, 1.0);
	}
    else {
      gsl_vector_set(y, i, 0.0);
    }
  }
  fclose(s);
  fprintf(stderr, "Data read and file closed\n");
  
  fprintf(stderr, "\ny: ( ");
  for (i=0; i<50; i++) {
    fprintf(stderr, "%f ", gsl_vector_get(y, i));
  }
  fprintf(stderr, "... ... )'\n");
  fprintf(stderr, "\nx:\n");
  for (i=0; i<10; i++) {
    for (j=0; j<P; j++) {
      fprintf(stderr, "%f ", gsl_matrix_get(x, i, j));
    }
    fprintf(stderr, "\n");
  }
  fprintf(stderr, "...\n...\n");

  /* tymo = two y minus one = 2*y-1 */
  tymo = gsl_vector_alloc(N);
  gsl_vector_memcpy(tymo, y);
  gsl_vector_scale(tymo, 2.0);
  gsl_vector_add_constant(tymo, -1.0);
  
  fprintf(stderr, "\ntymo: ( ");
  for (i=0; i<50; i++) {
    fprintf(stderr, "%f ", gsl_vector_get(tymo, i));
  }
  fprintf(stderr, "... ... )'\n");

  /* some prep in advance of main MCMC loop */
  r = gsl_rng_alloc (gsl_rng_taus);
  eta = gsl_vector_alloc(N);
  betaP = gsl_vector_alloc(P);
  beta = gsl_vector_calloc(P);
  gsl_vector_set(beta, 0, -10);
  ll = -1e80;
  for (i=0; i<P; i++) {
    printf("beta%d ", i);
  }
  printf("\n");
  /* main mcmc loop */
  for (i=0; i<ITERS; i++) {
    for (j=0; j<THIN; j++) {
      ll = kernel(ll);
    }
    fprintf(stderr, "%d ", i);
    for (j=0; j<P; j++) {
      printf("%f ", gsl_vector_get(beta, j));
    }
    printf("\n");
  }
  fprintf(stderr, ".\n\n");
  
  fprintf(stderr, "\n\nBye...\n");
  return(0);
}

/* helper functions */

double ll(gsl_vector *beta) {
  int i;
  double llik = 0.0;
  gsl_blas_dgemv(CblasNoTrans, 1.0, x, beta, 0.0, eta); /* eta := x * beta */
  for (i=0; i<N; i++) {
    llik -= log(1.0 + exp(-gsl_vector_get(tymo, i)*gsl_vector_get(eta, i)));
  }
  return(llik);
}

double lprior(gsl_vector *beta) {
  int i;
  double lp = 0.0;
  lp += log(gsl_ran_gaussian_pdf(gsl_vector_get(beta, 0), 10.0));
  for (i=1; i<P; i++) {
    lp += log(gsl_ran_gaussian_pdf(gsl_vector_get(beta, i), 1.0));
  }
  return(lp);
}

double lpost(gsl_vector *beta) {
  return( ll(beta) + lprior(beta) );
}

  

double kernel(double ll) {
  int i;
  double llp;
  gsl_vector_set(betaP, 0, gsl_vector_get(beta, 0) + gsl_ran_gaussian(r, 0.2));
  for (i=1; i<P; i++) {
    gsl_vector_set(betaP, i, gsl_vector_get(beta, i) + gsl_ran_gaussian(r, 0.02));
  }
  llp = lpost(betaP);
  if (log(gsl_ran_flat(r, 0, 1)) < llp - ll) {
    gsl_vector_memcpy(beta, betaP);
    ll = llp;
  }
  return(ll);
}









/* eof */

