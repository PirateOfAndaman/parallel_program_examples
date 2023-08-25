#include<stdio.h>
#include<stdlib.h>
#include<omp.h>
#define MIN(x,y) ((x<y)?x:y)
static long num_steps=1000;


int main(){


    int n=10000;
    // printf("Enter the dimension of image: ");
    // scanf("%d",&n);

    int* img=(int *)malloc(sizeof(int)*n*n);

    // for(int i=0;i<n;i++){
    //     for(int j=0;j<n;j++){
    //         scanf("%d",&img[i*n+j]);
    //     }
    // }

      for(int i=0;i<n;i++){
        for(int j=0;j<n;j++){
           img[i*n+j]=1;
        }
    }


    int m=3;

    // printf("Enter the dimension of convolution matrix :");
    // scanf("%d",&m);

        int* con=(int *)malloc(sizeof(int)*m*m);

    // for(int i=0;i<m;i++){
    //     for(int j=0;j<m;j++){
    //         scanf("%d",&con[i*m+j]);
    //     }
    // }

    for(int i=0;i<m;i++){
        for(int j=0;j<m;j++){
            con[i*m+j]=0;
        }
    }
    int midr=m/2;
    int midc=m/2;

    int num_procs = omp_get_num_procs();
    omp_set_num_threads(num_procs);
    int batchSize=n*n/2;
    long int res=0;
       
  double start_parallel = omp_get_wtime();
    #pragma omp parallel
    {
         int id=omp_get_thread_num();
         int start =batchSize*id;
         int end=MIN(n*n,batchSize*(id+1));
         int part_sum=0;

        
         for(int i=start;i<end;i++){
            int r=i/n;  //index in the img matrix, changed from row major to (x,y)
            int c=i%n;

           
           for (int a=midc;a>=0;a--){
            for(int b=midr;b>=0;b--){
                if(((r-a)>=0&&(r-a<n))&&((c-b)>=0&&(c-b<n)))
                    {
                        //cout<<r-a<<" "<<c-b<<" "<<midr-a<<" "<<midc-b<<endl;
                        part_sum+=img[(r-a)*n+c-b]*con[(midr-a)*m+midc-b];
                    }
                
            }
           }

            for (int a=midr;a>=1;a--){
            for(int b=midc;b>=0;b--){
                if(((r+a)>=0&&(r+a<n))&&((c-b)>=0&&(c-b<n))){
                         //cout<<r+a<<" "<<c-b<<" "<<midr+a<<" "<<midc-b<<endl;
                         part_sum+=img[(r+a)*n+c-b]*con[(midr+a)*m+midc-b];
                }
               
            }
           }

            for (int a=midr;a>=1;a--){
            for(int b=midc;b>=1;b--){
                if(((r+a)>=0&&(r+a<n))&&((c+b)>=0&&(c+b<n))){
                   // cout<<r+a<<" "<<c+b<<" "<<midr+a<<" "<<midc+b<<endl;
                    part_sum+=img[(r+a)*n+c+b]*con[(midr+a)*m+midc+b];
                }
                
           }
            }

            for (int a=midr;a>=0;a--){
            for(int b=midc;b>=1;b--){
                if(((r-a)>=0&&(r-a<n))&&((c+b)>=0&&(c+b<n))){
                       // cout<<r-a<<" "<<c+b<<" "<<midr-a<<" "<<midc+b<<endl;
                        part_sum+=img[(r-a)*n+c+b]*con[(midr-a)*m+midc+b];

                }
                
            }}
           
          
         }
           #pragma omp critical
            {
                res+=part_sum;
            }
           

    }
    double end_parallel = omp_get_wtime(); 
  printf("Parallel Work took %f miliseconds\n", (end_parallel - start_parallel)*1000);
    printf("Result :%ld",res);

    return 0;
}