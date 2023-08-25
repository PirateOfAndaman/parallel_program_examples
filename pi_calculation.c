#include<stdio.h>
#include<stdlib.h>
#include<omp.h>
#define MIN(x,y) ((x<y)?x:y)
static long num_steps=1000;


int main(){
    
    int num_procs = omp_get_num_procs();
    omp_set_num_threads(num_procs);
    double*partial_sum=(double *)malloc(num_procs*sizeof(double));

    
    double step=1.0/(double)num_steps;
    int batch=num_steps/num_procs;
      double res=0;
    #pragma omp parallel
    {

        int id=omp_get_thread_num();
        double part_sum=0;
        int start=batch*id;

        int end=batch*(id+1);
        end=MIN(end,num_steps);

        for(int i=start;i<end;i++)
        {
        double x=i*step;
        part_sum+=(4.0/(1.0+x*x)); //i could either  res+=(4.0/(1.0+x*x))*step but multiplication is expensive and we are multiplying 
                            // each point of function with the same value of dx so better just to multiply at last.
        }
        #pragma omp critical
            res+=part_sum*step;

    }
    
  

    // for(int i=0;i<num_procs;i++){
    //     res+=partial_sum[i];
    // }
    printf("Value of pi is hopefully: %f\n",res);
    

    return 0;
}