#include "imgPgm.h"

int main(){
    unsigned char image[1024*1024];
    char image_name[]="./voyager2.pgm";
    int rtn;

    printf("Test imgPgm\r\n");
    rtn = src_read_pgm(image_name,image, 805,623);
    printf("read rtn: %d\r\n", rtn);

    return 0;
}